import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import os
import google.generativeai as genai
from datetime import datetime

# ============================================================================
# UNIT CONVERSION
# ============================================================================

# Conversion factors to ng/mL (base unit)
CONVERSION_FACTORS = {
    'ng/mL': 1,
    'pg/mL': 0.001,
    'μg/mL': 1000,
    'mg/mL': 1000000,
    'ng/L': 0.001,
    'μg/L': 1,
    'mg/L': 1000,
    'ng/dL': 0.01,
    'mg/dL': 10000,
    'ng/μL': 1000,
    'g/mL': 1000000000.0
}

UNITS_LIST = list(CONVERSION_FACTORS.keys())

def convert_to_ng_ml(value, unit):
    """Convert value from given unit to ng/mL"""
    return value * CONVERSION_FACTORS.get(unit, 1)

def format_biomarker_display(biomarker, value, unit):
    """Format biomarker for display"""
    return f"{biomarker}: {value} {unit} ({convert_to_ng_ml(value, unit):.4f} ng/mL)"

# ============================================================================
# GEMINI API CONFIGURATION
# ============================================================================

def load_system_prompt():
    """Load system prompt from prompt.pmt file"""
    try:
        with open('prompt.pmt', 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        st.error("❌ prompt.pmt file not found!")
        return "You are a helpful medical AI assistant specializing in biomarker analysis and infection classification."

def initialize_gemini():
    """Initialize Gemini API with API key from environment variable"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        st.error("❌ GEMINI_API_KEY environment variable not set!")
        st.stop()
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name='gemini-2.5-flash-lite',
        system_instruction=load_system_prompt()
    )

# ============================================================================
# AUTHENTICATION SYSTEM
# ============================================================================

def load_password_hash():
    try:
        with open('passkey.pwd', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        st.error("Password file not found! Please generate 'passkey.pwd' first.")
        st.stop()

def verify_password(entered_password, stored_hash):
    entered_hash = hashlib.sha256(entered_password.encode()).hexdigest()
    return entered_hash == stored_hash

def check_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True
    
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 40px;
            background-color: #f0f2f6;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:        
        password = st.text_input("Enter Password", type="password", key="login_password")
        col_a, col_b = st.columns(2)
        
        with col_a:
            login_button = st.button("🔓 Login", use_container_width=True, type="primary")
        
        with col_b:
            if st.button("❓ Help", use_container_width=True):
                st.info("Please contact the administrator for access credentials.")
        
        if login_button:
            if password:
                stored_hash = load_password_hash()
                if verify_password(password, stored_hash):
                    st.session_state.authenticated = True
                    st.session_state.current_page = "Home"
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Incorrect password. Please try again.")
            else:
                st.warning("⚠️ Please enter a password.")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    return False

# ============================================================================
# CHECK AUTHENTICATION
# ============================================================================

if not check_authentication():
    st.stop()

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_resource
def load_statistics():
    with open("biomarker_statistics.pkl", "rb") as f:
        return pickle.load(f)

stats_dict = load_statistics()
biomarkers = sorted(stats_dict.keys())

# ============================================================================
# CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_infection(biomarker, threshold_value, stats_dict, verbose=True):
    if biomarker not in stats_dict:
        return None

    stats_df = stats_dict[biomarker]
    matches_std_range = []

    for _, row in stats_df.iterrows():
        if row['Mean - Std'] <= threshold_value <= row['Mean + Std']:
            matches_std_range.append({
                'Infection': row['Infection'],
                'Mean': row['Mean'],
                'Distance': abs(threshold_value - row['Mean']),
                'Range_Type': 'Mean ± Std'
            })

    def compute_confidence(matches):
        if len(matches) == 1:
            matches[0]['Confidence'] = 99.0
            return matches
        total = sum([1 / (m['Distance'] + 1e-10) for m in matches])
        for m in matches:
            m['Confidence'] = (1 / (m['Distance'] + 1e-10)) / total * 100
        return sorted(matches, key=lambda x: x['Confidence'], reverse=True)

    if matches_std_range:
        return {
            'Biomarker': biomarker,
            'Threshold': threshold_value,
            'Matches': compute_confidence(matches_std_range),
            'Classification_Method': 'Mean ± Std Range',
            'Total_Matches': len(matches_std_range)
        }

    matches_minmax = []
    for _, row in stats_df.iterrows():
        if row['Min'] <= threshold_value <= row['Max']:
            matches_minmax.append({
                'Infection': row['Infection'],
                'Mean': row['Mean'],
                'Distance': abs(threshold_value - row['Mean']),
                'Range_Type': 'Min-Max'
            })

    if matches_minmax:
        return {
            'Biomarker': biomarker,
            'Threshold': threshold_value,
            'Matches': compute_confidence(matches_minmax),
            'Classification_Method': 'Min-Max Range',
            'Total_Matches': len(matches_minmax)
        }

    return {
        'Biomarker': biomarker,
        'Threshold': threshold_value,
        'Matches': [],
        'Classification_Method': 'No Match',
        'Total_Matches': 0
    }


def classify_infection_bayesian_geometric_smoothed(biomarker_thresholds, stats_dict, verbose=True):
    """
    Bayesian classification with geometric mean and probability smoothing.
    Smoothing prevents complete elimination when biomarkers disagree.
    """
    MIN_PROBABILITY = 0.001

    first_biomarker = list(biomarker_thresholds.keys())[0]
    if first_biomarker not in stats_dict:
        if verbose:
            st.error(f"Error: No statistics found for biomarker: {first_biomarker}")
        return None

    all_infections = stats_dict[first_biomarker]['Infection'].unique()
    infection_probs = {infection: 1.0 for infection in all_infections}

    successful_biomarkers = []
    individual_results = {}

    matched_infections = set()
    temp_results = {}

    # STEP 1: Collect matches
    for biomarker_data in biomarker_thresholds.values():
        biomarker = biomarker_data['biomarker']
        threshold_ng_ml = biomarker_data['value_ng_ml']
        
        if biomarker not in stats_dict:
            if verbose:
                st.warning(f"No statistics for {biomarker}, skipping...")
            continue

        result = classify_infection(biomarker, threshold_ng_ml, stats_dict, verbose=False)

        if result is None or result['Total_Matches'] == 0:
            if verbose:
                st.warning(f"No matches found for {biomarker}={threshold_ng_ml}, skipping...")
            continue

        temp_results[biomarker] = result
        successful_biomarkers.append(biomarker)
        for match in result['Matches']:
            matched_infections.add(match['Infection'])

    # STEP 2: Apply smoothing logic
    for biomarker in successful_biomarkers:
        result = temp_results[biomarker]
        individual_results[biomarker] = result
        biomarker_probs = {}
        for match in result['Matches']:
            biomarker_probs[match['Infection']] = match['Confidence'] / 100.0
        
        for infection in all_infections:
            if infection in biomarker_probs:
                # Matched by this biomarker
                infection_probs[infection] *= biomarker_probs[infection]
            elif infection in matched_infections:
                # Matched by another biomarker (smoothing)
                infection_probs[infection] *= MIN_PROBABILITY
            else:
                # Never matched by any biomarker
                infection_probs[infection] *= 0.0

    # STEP 3: Apply geometric mean
    n_biomarkers = len(successful_biomarkers)

    if n_biomarkers > 0:
        for infection in infection_probs:
            if infection_probs[infection] > 0:
                infection_probs[infection] = infection_probs[infection] ** (1.0 / n_biomarkers)
    
    total_prob = sum(infection_probs.values())

    # STEP 4: Build result
    if total_prob == 0:
        result = {
            'Method': 'Bayesian Probability (Geometric Mean + Smoothing)',
            'Biomarkers_Used': successful_biomarkers,
            'Total_Biomarkers': len(biomarker_thresholds),
            'Classifications': [],
            'Status': 'No Classification',
            'Smoothing_Applied': True,
            'Min_Probability': MIN_PROBABILITY,
            'Matched_Infections': list(matched_infections)
        }
    else:
        normalized_probs = {
            infection: (prob / total_prob) * 100
            for infection, prob in infection_probs.items()
        }
        sorted_infections = sorted(normalized_probs.items(), key=lambda x: x[1], reverse=True)
        classifications = [
            {
                'Infection': infection,
                'Confidence': confidence,
                'Rank': rank + 1
            }
            for rank, (infection, confidence) in enumerate(sorted_infections)
            if confidence > 0
        ]

        result = {
            'Method': 'Bayesian Probability (Geometric Mean + Smoothing)',
            'Biomarkers_Used': successful_biomarkers,
            'Total_Biomarkers': len(biomarker_thresholds),
            'Individual_Results': individual_results,
            'Classifications': classifications,
            'Status': 'Success',
            'Note': f'Geometric mean with smoothing applied (N={n_biomarkers}, Min Prob={MIN_PROBABILITY})',
            'Smoothing_Applied': True,
            'Min_Probability': MIN_PROBABILITY,
            'Matched_Infections': list(matched_infections)
        }

    return result


def plot_classification_ranges(biomarker, threshold_value, stats_dict, classification_result):
    stats_df = stats_dict[biomarker]
    
    # Filter out infections with less than 5 data points
    stats_df = stats_df[stats_df['Count'] >= 5].copy()
    
    if len(stats_df) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No infections with sufficient data points (≥5) to display', 
                ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    infections = stats_df['Infection'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(infections)))
    infection_colors = dict(zip(infections, colors))
    matched = [m['Infection'] for m in classification_result['Matches'] if m['Infection'] in infections]

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)

    for ax, mode in zip(axes, ['Mean ± Std', 'Min-Max']):
        for i, row in stats_df.iterrows():
            inf = row['Infection']
            color = infection_colors[inf]
            alpha = 0.9 if inf in matched else 0.4
            lw = 5 if inf in matched else 3

            if mode == 'Mean ± Std':
                low, high = row['Mean - Std'], row['Mean + Std']
            else:
                low, high = row['Min'], row['Max']

            ax.plot([low, high], [i, i], color=color, linewidth=lw, alpha=alpha)
            ax.scatter([row['Mean']], [i], color=color, edgecolors='black', s=200, alpha=alpha)
            
            label_text = f"{inf} (n={int(row['Count'])})"
            ax.text(low - (high-low)*0.05, i, label_text, ha='right', va='center', fontsize=9)

        ax.axvline(threshold_value, linestyle='--', linewidth=2, color='red', label=f'Lab reading: {threshold_value}')
        ax.set_title(f"{mode} Range", fontsize=11, fontweight='bold')
        ax.set_xlabel('Lab reading (ng/mL)', fontsize=10)
        ax.set_yticks([])
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"{biomarker} Classification", fontsize=13, fontweight='bold')
    return fig

# ============================================================================
# LLM FUNCTIONS
# ============================================================================
def get_llm_response(user_message, chat_history):
    """Get response from Gemini API"""
    try:
        model = initialize_gemini()
        
        # Build conversation history for context
        messages = []
        for msg in chat_history:
            role = 'model' if msg['role'] == 'assistant' else 'user'
            messages.append({
                'role': role,
                'parts': [msg['content']]
            })
        
        # Start chat with history
        chat = model.start_chat(history=messages)
        
        # Send message and get response
        response = chat.send_message(user_message)
        
        return response.text
        
    except Exception as e:
        print("\n" + "="*80)
        print("🔴 GEMINI API ERROR")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"User Message: {user_message}")
        print("="*80 + "\n")
        return "⚠️ System Error: Unable to process your request at this time. Please try again later or contact support."

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

if 'biomarker_thresholds' not in st.session_state:
    st.session_state.biomarker_thresholds = {}

if 'input_counter' not in st.session_state:
    st.session_state.input_counter = 0

if 'llm_chat_history' not in st.session_state:
    st.session_state.llm_chat_history = []

if 'llm_submitted' not in st.session_state:
    st.session_state.llm_submitted = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Biomarker Infection Classifier", layout="wide")

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("### 👤 User Session")
    st.success("✅ Authenticated")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.llm_chat_history = []
        st.session_state.biomarker_thresholds = {}
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📑 Navigation")
    
    if st.button("🏠 Home", use_container_width=True, 
                 type="primary" if st.session_state.current_page == "Home" else "secondary"):
        st.session_state.current_page = "Home"
        st.rerun()
    
    if st.button("📊 Statistical", use_container_width=True,
                 type="primary" if st.session_state.current_page == "Statistical" else "secondary"):
        st.session_state.current_page = "Statistical"
        st.rerun()
    
    if st.button("🤖 LLM-Aided", use_container_width=True,
                 type="primary" if st.session_state.current_page == "LLM-Aided" else "secondary"):
        st.session_state.current_page = "LLM-Aided"
        st.rerun()

# ============================================================================
# HELPER FUNCTION FOR BIOMARKER INPUT
# ============================================================================

def render_biomarker_input(page_prefix=""):
    """Render biomarker input section with unit support"""
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    
    with col1:
        available_biomarkers = [b for b in biomarkers if b not in st.session_state.biomarker_thresholds]
        selected_biomarker = st.selectbox(
            "Select Biomarker", 
            [""] + available_biomarkers, 
            key=f"{page_prefix}biomarker_select_{st.session_state.input_counter}"
        )
    
    with col2:
        threshold_input = st.number_input(
            "Lab reading value", 
            min_value=0.0, 
            value=None,
            step=0.01, 
            format="%.4f", 
            placeholder="Enter value...",
            key=f"{page_prefix}threshold_input_{st.session_state.input_counter}"
        )
    
    with col3:
        selected_unit = st.selectbox(
            "Unit",
            UNITS_LIST,
            key=f"{page_prefix}unit_select_{st.session_state.input_counter}"
        )
    
    with col4:
        st.write("")
        st.write("")
        add_button = st.button("➕ Add", use_container_width=True, key=f"{page_prefix}add_btn")
    
    if add_button and selected_biomarker and threshold_input is not None:
        if selected_biomarker not in st.session_state.biomarker_thresholds:
            # Store both original and converted values
            st.session_state.biomarker_thresholds[selected_biomarker] = {
                'biomarker': selected_biomarker,
                'value': threshold_input,
                'unit': selected_unit,
                'value_ng_ml': convert_to_ng_ml(threshold_input, selected_unit)
            }
            st.session_state.input_counter += 1
            st.rerun()
        else:
            st.warning(f"{selected_biomarker} is already added!")
    elif add_button and threshold_input is None:
        st.warning("Please enter a value!")

def display_added_biomarkers(page_prefix=""):
    """Display currently added biomarkers with edit functionality"""
    if st.session_state.biomarker_thresholds:
        st.markdown("### Currently Added Biomarkers:")
        
        for bio_key in list(st.session_state.biomarker_thresholds.keys()):
            bio_data = st.session_state.biomarker_thresholds[bio_key]
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            
            with col1:
                st.write(f"**{bio_data['biomarker']}**")
            
            with col2:
                new_value = st.number_input(
                    f"Value", 
                    value=bio_data['value'],
                    min_value=0.0,
                    step=0.01,
                    format="%.4f",
                    key=f"{page_prefix}edit_val_{bio_key}",
                    label_visibility="collapsed"
                )
            
            with col3:
                new_unit = st.selectbox(
                    "Unit",
                    UNITS_LIST,
                    index=UNITS_LIST.index(bio_data['unit']),
                    key=f"{page_prefix}edit_unit_{bio_key}",
                    label_visibility="collapsed"
                )
            
            # Update if changed
            if new_value != bio_data['value'] or new_unit != bio_data['unit']:
                st.session_state.biomarker_thresholds[bio_key] = {
                    'biomarker': bio_data['biomarker'],
                    'value': new_value,
                    'unit': new_unit,
                    'value_ng_ml': convert_to_ng_ml(new_value, new_unit)
                }
            
            with col4:
                if st.button("🗑️", key=f"{page_prefix}remove_{bio_key}", use_container_width=True):
                    del st.session_state.biomarker_thresholds[bio_key]
                    st.rerun()
            
            # Show converted value
            st.caption(f"→ {bio_data['value_ng_ml']:.4f} ng/mL")
        
        st.write(f"**Total Biomarkers:** {len(st.session_state.biomarker_thresholds)}")

# ============================================================================
# PAGE ROUTING
# ============================================================================

# HOME PAGE
if st.session_state.current_page == "Home":
    st.title("🧬 Biomarker Infection Classifier")
    st.markdown("---")
    
    st.markdown("## Welcome!")
    st.markdown("This tool helps classify infection types based on biomarker laboratory readings.")
    
    st.markdown("### Choose Your Classification Method:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Statistical Classification")
        st.markdown("""
        Uses Bayesian probability and statistical range analysis to predict infections based on:
        - Mean ± Standard Deviation ranges
        - Min-Max ranges
        - Multi-biomarker fusion with smoothing
        """)
        if st.button("Go to Statistical Classification →", use_container_width=True, type="primary"):
            st.session_state.current_page = "Statistical"
            st.rerun()
    
    with col2:
        st.markdown("#### 🤖 LLM-Aided Classification")
        st.markdown("""
        Uses Large Language Models combined with statistical data to provide:
        - Natural language explanations
        - Context-aware predictions
        - Interactive assistance
        """)
        if st.button("Go to LLM-Aided Classification →", use_container_width=True, type="primary"):
            st.session_state.current_page = "LLM-Aided"
            st.rerun()
    
    st.markdown("---")
    st.markdown("*Powered by ASU*")

# STATISTICAL PAGE
elif st.session_state.current_page == "Statistical":
    st.title("📊 Statistical Classification")
    st.markdown("---")
    
    st.subheader("Add Biomarkers")
    render_biomarker_input("stat_")
    display_added_biomarkers("stat_")
    
    st.markdown("---")
    
    if st.session_state.biomarker_thresholds:
        if st.button("🔬 Classify Infection", type="primary", use_container_width=True):
            
            num_biomarkers = len(st.session_state.biomarker_thresholds)
            
            if num_biomarkers == 1:
                bio_key = list(st.session_state.biomarker_thresholds.keys())[0]
                bio_data = st.session_state.biomarker_thresholds[bio_key]
                biomarker = bio_data['biomarker']
                threshold_ng_ml = bio_data['value_ng_ml']
                
                st.subheader("🔍 Classification Results")
                
                result = classify_infection(biomarker, threshold_ng_ml, stats_dict, verbose=False)
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.container(border=True):
                        st.markdown(f"### 🧪 {biomarker}")
                        st.markdown(f"**Lab reading:** {bio_data['value']} {bio_data['unit']} ({threshold_ng_ml:.4f} ng/mL)")
                        st.markdown(f"**Classification Method:** {result['Classification_Method']}")
                        
                        if result['Total_Matches'] == 0:
                            st.warning("⚠️ No matching infection found.")
                        else:
                            st.success(f"✅ Found {result['Total_Matches']} matching infection(s)")
                            st.markdown("#### Infection Matches:")
                            for i, match in enumerate(result['Matches'], 1):
                                st.markdown(f"{i}. **{match['Infection']}** — {match['Confidence']:.2f}% confidence")
                        st.markdown("---")
                        st.markdown("#### Visualization")
                        fig = plot_classification_ranges(biomarker, threshold_ng_ml, stats_dict, result)
                        st.pyplot(fig, clear_figure=True)
                
                with col2:
                    st.write("")
            
            else:
                st.subheader("🔍 Individual Biomarker Results")
                individual_results = {}
                biomarker_list = list(st.session_state.biomarker_thresholds.keys())
                
                for idx in range(0, len(biomarker_list), 2):
                    cols = st.columns(2)
                    
                    for col_idx, col in enumerate(cols):
                        biomarker_idx = idx + col_idx
                        if biomarker_idx < len(biomarker_list):
                            bio_key = biomarker_list[biomarker_idx]
                            bio_data = st.session_state.biomarker_thresholds[bio_key]
                            biomarker = bio_data['biomarker']
                            threshold_ng_ml = bio_data['value_ng_ml']
                            
                            with col:
                                with st.container(border=True):
                                    st.markdown(f"### {biomarker}")
                                    st.markdown(f"**Lab reading:** {bio_data['value']} {bio_data['unit']} ({threshold_ng_ml:.4f} ng/mL)")
                                    result = classify_infection(biomarker, threshold_ng_ml, stats_dict, verbose=False)
                                    individual_results[biomarker] = result
                                    st.markdown(f"**Method:** {result['Classification_Method']}")
                                    
                                    if result['Total_Matches'] == 0:
                                        st.warning("No matches found")
                                    else:
                                        st.success(f"{result['Total_Matches']} match(es)")
                                        for i, match in enumerate(result['Matches'][:4], 1):
                                            st.markdown(f"{i}. **{match['Infection']}**: {match['Confidence']:.2f}%")
                                    fig = plot_classification_ranges(biomarker, threshold_ng_ml, stats_dict, result)
                                    st.pyplot(fig, clear_figure=True)
                
                st.markdown("---")
                st.subheader("🎯 Combined Multi-Biomarker Classification")
                
                combined_result = classify_infection_bayesian_geometric_smoothed(
                    st.session_state.biomarker_thresholds, 
                    stats_dict, 
                    verbose=False
                )
                
                if combined_result and combined_result['Status'] == 'Success':
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("#### 📋 Summary")
                        st.markdown(f"**Biomarkers Used:** {len(combined_result['Biomarkers_Used'])} / {combined_result['Total_Biomarkers']}")
                        st.markdown(f"**Method:** Bayesian with Smoothing")
                        st.markdown(f"**Min Probability:** {combined_result.get('Min_Probability', 'N/A')}")
                    
                    with col2:
                        st.markdown("#### Final Classification Results")
                        
                        results_data = []
                        for classification in combined_result['Classifications']:
                            rank = classification['Rank']
                            infection = classification['Infection']
                            confidence = classification['Confidence']
                            
                            if rank == 1:
                                status = "⭐ MOST LIKELY"
                            elif confidence > 10:
                                status = "✓ Possible"
                            else:
                                status = "○ Unlikely"
                            
                            results_data.append({
                                'Rank': rank,
                                'Infection': infection,
                                'Confidence (%)': f"{confidence:.2f}",
                                'Status': status
                            })
                        
                        import pandas as pd
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, hide_index=True, use_container_width=True)
                    
                    if combined_result['Classifications']:
                        top_infection = combined_result['Classifications'][0]['Infection']
                        top_confidence = combined_result['Classifications'][0]['Confidence']
                        st.success(f"### 🎯 Predicted Infection: **{top_infection}** ({top_confidence:.2f}% confidence)")
                        
                        # Show uncertainty warning if applicable
                        if len(combined_result['Classifications']) >= 2:
                            second_conf = combined_result['Classifications'][1]['Confidence']
                            if abs(top_confidence - second_conf) < 10:
                                st.warning("⚠️ Close confidence scores detected. This may indicate conflicting biomarker evidence or co-infection.")
                else:
                    st.error("❌ Could not perform multi-biomarker classification. Please check your inputs.")
    else:
        st.info("👆 Please add at least one biomarker to begin classification.")
    
    st.markdown("---")
    st.markdown("*Powered by ASU*")

# LLM-AIDED PAGE
elif st.session_state.current_page == "LLM-Aided":
    st.title("🤖 LLM-Aided Classification")
    st.markdown("---")
    
    if not st.session_state.llm_submitted:
        st.subheader("Add Biomarkers")
        render_biomarker_input("llm_")
        display_added_biomarkers("llm_")
        
        st.markdown("---")
        
        if st.session_state.biomarker_thresholds:
            if st.button("🔬 Classify", type="primary", use_container_width=True, key="llm_classify_btn"):
                st.session_state.llm_submitted = True
                
                # Create initial user message with biomarkers (with units)
                biomarker_text = "\n".join([
                    f"{data['biomarker']}: {data['value']} {data['unit']}" 
                    for data in st.session_state.biomarker_thresholds.values()
                ])
                
                st.session_state.llm_chat_history.append({
                    'role': 'user',
                    'content': biomarker_text
                })
                
                st.rerun()
        else:
            st.info("👆 Please add at least one biomarker to begin classification.")
    
    else:
        st.subheader("💬 Classification Analysis")
        
        # Display chat history
        for idx, message in enumerate(st.session_state.llm_chat_history):
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(f"```\n{message['content']}\n```")
            else:
                with st.chat_message("assistant"):
                    st.markdown(message['content'])
        
        # If last message was user, get LLM response
        if st.session_state.llm_chat_history and st.session_state.llm_chat_history[-1]['role'] == 'user':
            with st.chat_message("assistant"):
                with st.spinner("🔄 Processing your request..."):
                    response = get_llm_response(
                        st.session_state.llm_chat_history[-1]['content'],
                        st.session_state.llm_chat_history[:-1]
                    )
                    st.markdown(response)
            
            st.session_state.llm_chat_history.append({
                'role': 'assistant',
                'content': response
            })
            st.rerun()
        
        # Chat input for follow-up questions
        user_input = st.chat_input("Ask a follow-up question...")
        
        if user_input:
            st.session_state.llm_chat_history.append({
                'role': 'user',
                'content': user_input
            })
            st.rerun()
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 New Classification", use_container_width=True):
            st.session_state.llm_submitted = False
            st.session_state.llm_chat_history = []
            st.session_state.biomarker_thresholds = {}
            st.session_state.input_counter += 1
            st.rerun()
    
    st.markdown("---")
    st.markdown("*Powered by ASU*")
