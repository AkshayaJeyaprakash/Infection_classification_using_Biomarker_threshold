import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import os
import google.generativeai as genai
from datetime import datetime
import pandas as pd

# ============================================================================
# UNIT CONVERSION
# ============================================================================

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

SYMPTOMS_LIST = [
    'Fever',
    'Malaise/Fatigue',
    'Chills',
    'Nausea/Vomiting',
    'Muscle aches',
    'Cough',
    'Headache',
    'Sore throat',
    'Pain and swelling',
    'Congestion'
]

def convert_to_ng_ml(value, unit):
    return value * CONVERSION_FACTORS.get(unit, 1)

def format_biomarker_display(biomarker, value, unit):
    return f"{biomarker}: {value} {unit} ({convert_to_ng_ml(value, unit):.4f} ng/mL)"

# ============================================================================
# GEMINI API CONFIGURATION
# ============================================================================

def load_system_prompt():
    try:
        with open('prompt.pmt', 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        st.error("❌ prompt.pmt file not found!")
        return "You are a helpful medical AI assistant specializing in biomarker analysis and infection classification."

def initialize_gemini():
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

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_PROBABILITY = 0.001
TOP_N_DISEASE_PRINT = 5

SYMPTOM_TO_CODE = {
    "Bacterial": "BR",
    "Viral": "VR",
    "Viral/Bacterial": "BVR",
    "Control": "C",
    "C": "C"
}

CODE_TO_SYMPTOM = {
    "BR": "Bacterial",
    "VR": "Viral",
    "BVR": "Viral/Bacterial",
    "C": "Control"
}

ML_SYMPTOM_COLS = [
    'Fever/ \nTemp\u2009> 38°C',
    'Chills/Shaking/Shivering/Rigors',
    'Malaise/Fatigue',
    'Muscle/Body Aches/Arthralgia/Myalgia',
    'Lymphadenopathy/Swollen Lymph',
    'Sweat',
    'Cough',
    'Sore Throat/\nThroat Irritation/\nThroat Infection/\nInflamed pharynx/\nPharyngitis',
    'Nasal Congestion',
    'Diarrhea/Other GI Symptoms',
    'Vomiting',
    'Dysuria',
    'Neck Stiffness',
    'Rash/Skin Reactions',
    'Tachycardia/Circulation//Palptations',
    'Tachypnea/Tachypnoea/Shortness of Breath/Wheezing/Dyspnea/Grunting/Stridor/Difficulty Breathing',
    'Dry Throat/ Hoarse voice',
    'Dry Nose',
    ' Loss of Smell/Anosmia ',
    'Loss of Taste/Dysgeusia ',
    'Loss of Appetite',
    'Headache ',
    'Chest Pain',
    'Oxygen saturation\u2009<\u200990% in room air',
    'Elevated pulse (>90 bpm)',
    'Respiratory rate >20 breaths per minute',
    'Dizziness',
    'Nausea',
    'Sputum/Expectoration',
    'Runny Nose/Rhinorrhoea',
    'Conjunctivitis/Eye Pain',
    'Arthralgias/Joint Pain',
    'Disturbed Sleep',
    'Rhinitis',
    'Sinus',
    'Ear Pain (Earache, Otalgia)',
    'Chest Indrawing',
    'Cognitive health/Brain fog',
    'Sneezing',
    'Phlegm',
    'Weight loss',
    'Hemoptysis/ \nBlood in mucus',
    'Exudates',
    'Tonsillitis/ \nInflamed tonsils '
]

ML_SYMPTOM_LABELS = {
    'Fever/ \nTemp\u2009> 38°C': 'Fever > 38 C',
    'Chills/Shaking/Shivering/Rigors': 'Chills / Rigors',
    'Malaise/Fatigue': 'Malaise / Fatigue',
    'Muscle/Body Aches/Arthralgia/Myalgia': 'Muscle / Body aches',
    'Lymphadenopathy/Swollen Lymph': 'Swollen lymph nodes',
    'Sweat': 'Sweating',
    'Cough': 'Cough',
    'Sore Throat/\nThroat Irritation/\nThroat Infection/\nInflamed pharynx/\nPharyngitis': 'Sore throat',
    'Nasal Congestion': 'Nasal congestion',
    'Diarrhea/Other GI Symptoms': 'Diarrhea / GI symptoms',
    'Vomiting': 'Vomiting',
    'Dysuria': 'Dysuria',
    'Neck Stiffness': 'Neck stiffness',
    'Rash/Skin Reactions': 'Rash / Skin reaction',
    'Tachycardia/Circulation//Palptations': 'Tachycardia / Palpitations',
    'Tachypnea/Tachypnoea/Shortness of Breath/Wheezing/Dyspnea/Grunting/Stridor/Difficulty Breathing': 'Shortness of breath',
    'Dry Throat/ Hoarse voice': 'Dry throat / Hoarse voice',
    'Dry Nose': 'Dry nose',
    ' Loss of Smell/Anosmia ': 'Loss of smell',
    'Loss of Taste/Dysgeusia ': 'Loss of taste',
    'Loss of Appetite': 'Loss of appetite',
    'Headache ': 'Headache',
    'Chest Pain': 'Chest pain',
    'Oxygen saturation\u2009<\u200990% in room air': 'Oxygen saturation < 90%',
    'Elevated pulse (>90 bpm)': 'Elevated pulse > 90 bpm',
    'Respiratory rate >20 breaths per minute': 'Respiratory rate > 20',
    'Dizziness': 'Dizziness',
    'Nausea': 'Nausea',
    'Sputum/Expectoration': 'Sputum / Expectoration',
    'Runny Nose/Rhinorrhoea': 'Runny nose',
    'Conjunctivitis/Eye Pain': 'Conjunctivitis / Eye pain',
    'Arthralgias/Joint Pain': 'Joint pain',
    'Disturbed Sleep': 'Disturbed sleep',
    'Rhinitis': 'Rhinitis',
    'Sinus': 'Sinus symptoms',
    'Ear Pain (Earache, Otalgia)': 'Ear pain',
    'Chest Indrawing': 'Chest indrawing',
    'Cognitive health/Brain fog': 'Brain fog',
    'Sneezing': 'Sneezing',
    'Phlegm': 'Phlegm',
    'Weight loss': 'Weight loss',
    'Hemoptysis/ \nBlood in mucus': 'Blood in mucus',
    'Exudates': 'Exudates',
    'Tonsillitis/ \nInflamed tonsils ': 'Tonsillitis'
}

ML_SYMPTOM_GROUPS = [
    ("General / Systemic", [
        'Fever/ \nTemp\u2009> 38°C',
        'Chills/Shaking/Shivering/Rigors',
        'Malaise/Fatigue',
        'Muscle/Body Aches/Arthralgia/Myalgia',
        'Lymphadenopathy/Swollen Lymph',
        'Sweat',
        'Loss of Appetite',
        'Weight loss'
    ]),
    ("Respiratory / ENT", [
        'Cough',
        'Sore Throat/\nThroat Irritation/\nThroat Infection/\nInflamed pharynx/\nPharyngitis',
        'Nasal Congestion',
        'Tachypnea/Tachypnoea/Shortness of Breath/Wheezing/Dyspnea/Grunting/Stridor/Difficulty Breathing',
        'Dry Throat/ Hoarse voice',
        'Dry Nose',
        'Sputum/Expectoration',
        'Runny Nose/Rhinorrhoea',
        'Rhinitis',
        'Sinus',
        'Ear Pain (Earache, Otalgia)',
        'Chest Indrawing',
        'Sneezing',
        'Phlegm',
        'Hemoptysis/ \nBlood in mucus',
        'Exudates',
        'Tonsillitis/ \nInflamed tonsils '
    ]),
    ("GI / Urinary", [
        'Diarrhea/Other GI Symptoms',
        'Vomiting',
        'Dysuria',
        'Nausea'
    ]),
    ("Neurologic / Sensory", [
        'Neck Stiffness',
        ' Loss of Smell/Anosmia ',
        'Loss of Taste/Dysgeusia ',
        'Headache ',
        'Dizziness',
        'Disturbed Sleep',
        'Cognitive health/Brain fog'
    ]),
    ("Skin / Pain / Other", [
        'Rash/Skin Reactions',
        'Chest Pain',
        'Conjunctivitis/Eye Pain',
        'Arthralgias/Joint Pain'
    ]),
    ("Vitals / Severity", [
        'Tachycardia/Circulation//Palptations',
        'Oxygen saturation\u2009<\u200990% in room air',
        'Elevated pulse (>90 bpm)',
        'Respiratory rate >20 breaths per minute'
    ])
]

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

    for biomarker in successful_biomarkers:
        result = temp_results[biomarker]
        individual_results[biomarker] = result
        biomarker_probs = {}
        for match in result['Matches']:
            biomarker_probs[match['Infection']] = match['Confidence'] / 100.0
        
        for infection in all_infections:
            if infection in biomarker_probs:
                infection_probs[infection] *= biomarker_probs[infection]
            elif infection in matched_infections:
                infection_probs[infection] *= MIN_PROBABILITY
            else:
                infection_probs[infection] *= 0.0

    n_biomarkers = len(successful_biomarkers)

    if n_biomarkers > 0:
        for infection in infection_probs:
            if infection_probs[infection] > 0:
                infection_probs[infection] = infection_probs[infection] ** (1.0 / n_biomarkers)
    
    total_prob = sum(infection_probs.values())

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

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for i, row in stats_df.iterrows():
        inf = row['Infection']
        color = infection_colors[inf]
        alpha = 0.9 if inf in matched else 0.4
        lw = 5 if inf in matched else 3
        low, high = row['Mean - Std'], row['Mean + Std']

        ax.plot([low, high], [i, i], color=color, linewidth=lw, alpha=alpha)
        ax.scatter([row['Mean']], [i], color=color, edgecolors='black', s=200, alpha=alpha)
        
        label_text = f"{inf} (n={int(row['Count'])})"
        ax.text(low - (high-low)*0.05, i, label_text, ha='right', va='center', fontsize=9)

    ax.axvline(threshold_value, linestyle='--', linewidth=2, color='red', label=f'Lab reading: {threshold_value}')
    ax.set_title("Mean ± Std Range", fontsize=11, fontweight='bold')
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
    try:
        model = initialize_gemini()
        messages = []
        for msg in chat_history:
            role = 'model' if msg['role'] == 'assistant' else 'user'
            messages.append({'role': role, 'parts': [msg['content']]})
        chat = model.start_chat(history=messages)
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
page_name_aliases = {
    "Statistical": "Statistical Approach",
    "LLM-Aided": "LLM-Based Approach",
    "ML-Aided (RF)": "Machine Learning Approach"
}
st.session_state.current_page = page_name_aliases.get(
    st.session_state.current_page,
    st.session_state.current_page
)
if 'biomarker_thresholds' not in st.session_state:
    st.session_state.biomarker_thresholds = {}
if 'input_counter' not in st.session_state:
    st.session_state.input_counter = 0
if 'llm_chat_history' not in st.session_state:
    st.session_state.llm_chat_history = []
if 'llm_submitted' not in st.session_state:
    st.session_state.llm_submitted = False
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = []
if 'other_symptoms' not in st.session_state:
    st.session_state.other_symptoms = ""


if 'rf_biomarker_thresholds' not in st.session_state:
    st.session_state.rf_biomarker_thresholds = {}
if 'rf_input_counter' not in st.session_state:
    st.session_state.rf_input_counter = 0
if 'rf_submitted' not in st.session_state:
    st.session_state.rf_submitted = False
if 'rf_individual_results' not in st.session_state:
    st.session_state.rf_individual_results = {}
if 'rf_combined_result' not in st.session_state:
    st.session_state.rf_combined_result = None
if 'rf_pipeline_result' not in st.session_state:
    st.session_state.rf_pipeline_result = None
if 'rf_symptoms' not in st.session_state:
    st.session_state.rf_symptoms = {col: 0 for col in ML_SYMPTOM_COLS}

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
        st.session_state.selected_symptoms = []
        st.session_state.other_symptoms = ""

        st.session_state.rf_biomarker_thresholds = {}
        st.session_state.rf_submitted = False
        st.session_state.rf_individual_results = {}
        st.session_state.rf_combined_result = None
        st.session_state.rf_pipeline_result = None
        st.session_state.rf_symptoms = {col: 0 for col in ML_SYMPTOM_COLS}
        st.session_state.rf_input_counter = 0
        
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📑 Navigation")
    
    if st.button("🏠 Home", use_container_width=True, 
                 type="primary" if st.session_state.current_page == "Home" else "secondary"):
        st.session_state.current_page = "Home"
        st.rerun()
    
    if st.button("📊 Statistical Approach", use_container_width=True,
                 type="primary" if st.session_state.current_page == "Statistical Approach" else "secondary"):
        st.session_state.current_page = "Statistical Approach"
        st.rerun()
    
    if st.button("🌲 Machine Learning Approach", use_container_width=True,
                 type="primary" if st.session_state.current_page == "Machine Learning Approach" else "secondary"):
        st.session_state.current_page = "Machine Learning Approach"
        st.rerun()

    if st.button("🤖 LLM-Based Approach", use_container_width=True,
                 type="primary" if st.session_state.current_page == "LLM-Based Approach" else "secondary"):
        st.session_state.current_page = "LLM-Based Approach"
        st.rerun()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def render_biomarker_input(page_prefix=""):
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
            
            st.caption(f"→ {bio_data['value_ng_ml']:.4f} ng/mL")
        
        st.write(f"**Total Biomarkers:** {len(st.session_state.biomarker_thresholds)}")

@st.cache_resource
def load_ml_artifact(name):
    with open(os.path.join(MODEL_DIR, name), "rb") as f:
        return pickle.load(f)


def geometric_mean_combine(source_prob_dicts, all_classes):
    n = len(source_prob_dicts)
    if n == 0:
        return {}

    matched_classes = set()
    for source in source_prob_dicts:
        for cls, probability in source.items():
            if probability > 0:
                matched_classes.add(cls)

    combined = {}
    for cls in all_classes:
        if cls not in matched_classes:
            combined[cls] = 0.0
            continue

        product = 1.0
        for source in source_prob_dicts:
            probability = source.get(cls, 0.0)
            product *= probability if probability > 0 else MIN_PROBABILITY
        combined[cls] = product ** (1.0 / n)

    total = sum(combined.values())
    if total == 0:
        return {cls: 0.0 for cls in all_classes}

    return {cls: value / total for cls, value in combined.items()}


def rank_classifications(normalized_probs):
    sorted_items = sorted(normalized_probs.items(), key=lambda x: x[1], reverse=True)
    classifications = []

    for rank, (cls, confidence) in enumerate(sorted_items, start=1):
        if confidence <= 0:
            break
        classifications.append({
            "Rank": rank,
            "Class": cls,
            "Confidence": confidence * 100
        })

    return classifications


def status_label(rank, confidence):
    if rank == 1:
        return "MOST LIKELY"
    if confidence > 10:
        return "Possible"
    return "Unlikely"


def infection_type_label(infection_type):
    readable = CODE_TO_SYMPTOM.get(infection_type)
    if readable:
        return f"{infection_type} ({readable})"
    return infection_type or "N/A"


def rf1_single(biomarker, threshold, le_bio, le_unit, le_target, clf):
    if biomarker not in le_bio.classes_:
        return None

    bio_enc = le_bio.transform([biomarker])[0]
    unit_enc = 0
    x = np.array([[bio_enc, unit_enc, threshold]])
    proba = clf.predict_proba(x)[0]

    return {cls: float(probability) for cls, probability in zip(le_target.classes_, proba)}


def predict_infection_from_biomarkers(biomarker_thresholds):
    clf = load_ml_artifact("rf1_infection_type.pkl")
    le_bio = load_ml_artifact("rf1_le_biomarker.pkl")
    le_unit = load_ml_artifact("rf1_le_units.pkl")
    le_target = load_ml_artifact("rf1_le_infection_type.pkl")
    all_classes = list(le_target.classes_)

    source_dicts = []
    individual_results = {}
    successful = []
    skipped = []

    for biomarker, threshold in biomarker_thresholds.items():
        prob_dict = rf1_single(biomarker, threshold, le_bio, le_unit, le_target, clf)
        if prob_dict is None:
            skipped.append(biomarker)
            continue

        source_dicts.append(prob_dict)
        individual_results[biomarker] = prob_dict
        successful.append(biomarker)

    if not source_dicts:
        return {
            "status": "No Classification",
            "predicted_infection_type": None,
            "confidence": 0.0,
            "classifications": [],
            "biomarkers_used": successful,
            "individual_results": individual_results,
            "skipped_biomarkers": skipped
        }

    normalized = geometric_mean_combine(source_dicts, all_classes)
    ranked = rank_classifications(normalized)
    top = ranked[0] if ranked else None

    return {
        "status": "Success",
        "predicted_infection_type": top["Class"] if top else None,
        "confidence": top["Confidence"] if top else 0.0,
        "classifications": ranked,
        "biomarkers_used": successful,
        "individual_results": individual_results,
        "skipped_biomarkers": skipped
    }


def predict_infection_from_symptoms(symptoms):
    clf = load_ml_artifact("rf3_infection_type_symptoms.pkl")
    le_target = load_ml_artifact("rf3_le_infection_type.pkl")

    x = np.array([[symptoms.get(col, 0) for col in ML_SYMPTOM_COLS]])
    proba = clf.predict_proba(x)[0]
    prob_dict = {
        SYMPTOM_TO_CODE.get(cls, cls): float(probability)
        for cls, probability in zip(le_target.classes_, proba)
    }
    ranked = rank_classifications(prob_dict)
    top = ranked[0] if ranked else None

    return {
        "status": "Success" if top else "No Classification",
        "predicted_infection_type": top["Class"] if top else None,
        "confidence": top["Confidence"] if top else 0.0,
        "classifications": ranked,
        "prob_dict": prob_dict
    }


def predict_infection_combined(biomarker_thresholds, symptoms):
    rf1 = predict_infection_from_biomarkers(biomarker_thresholds)
    rf3 = predict_infection_from_symptoms(symptoms)

    all_classes = sorted(
        set(c["Class"] for c in rf1["classifications"]) |
        set(c["Class"] for c in rf3["classifications"])
    )

    sources = []
    if rf1["status"] == "Success":
        sources.append({c["Class"]: c["Confidence"] / 100 for c in rf1["classifications"]})
    if rf3["status"] == "Success":
        sources.append({c["Class"]: c["Confidence"] / 100 for c in rf3["classifications"]})

    if not sources:
        return {
            "status": "No Classification",
            "predicted_infection_type": None,
            "confidence": 0.0,
            "classifications": [],
            "rf1_result": rf1,
            "rf3_result": rf3
        }

    normalized = geometric_mean_combine(sources, all_classes)
    ranked = rank_classifications(normalized)
    top = ranked[0] if ranked else None

    return {
        "status": "Success",
        "predicted_infection_type": top["Class"] if top else None,
        "confidence": top["Confidence"] if top else 0.0,
        "classifications": ranked,
        "rf1_result": rf1,
        "rf3_result": rf3
    }


def rf2_single(biomarker, threshold, infection_type, le_bio, le_unit, le_inf, le_target, clf):
    if biomarker not in le_bio.classes_ or infection_type not in le_inf.classes_:
        return None

    bio_enc = le_bio.transform([biomarker])[0]
    unit_enc = 0
    inf_enc = le_inf.transform([infection_type])[0]
    x = np.array([[bio_enc, unit_enc, threshold, inf_enc]])
    proba = clf.predict_proba(x)[0]

    return {cls: float(probability) for cls, probability in zip(le_target.classes_, proba)}


def predict_disease_from_biomarkers(biomarker_thresholds, infection_type):
    clf = load_ml_artifact("rf2_disease_name.pkl")
    le_bio = load_ml_artifact("rf2_le_biomarker.pkl")
    le_unit = load_ml_artifact("rf2_le_units.pkl")
    le_inf = load_ml_artifact("rf2_le_infection_type.pkl")
    le_target = load_ml_artifact("rf2_le_disease_name.pkl")
    all_classes = list(le_target.classes_)

    source_dicts = []
    individual_results = {}
    successful = []
    skipped = []

    for biomarker, threshold in biomarker_thresholds.items():
        prob_dict = rf2_single(
            biomarker, threshold, infection_type, le_bio, le_unit, le_inf, le_target, clf
        )
        if prob_dict is None:
            skipped.append(biomarker)
            continue

        source_dicts.append(prob_dict)
        individual_results[biomarker] = prob_dict
        successful.append(biomarker)

    if not source_dicts:
        return {
            "status": "No Classification",
            "predicted_disease": None,
            "confidence": 0.0,
            "classifications": [],
            "biomarkers_used": successful,
            "individual_results": individual_results,
            "skipped_biomarkers": skipped
        }

    normalized = geometric_mean_combine(source_dicts, all_classes)
    ranked = rank_classifications(normalized)
    top = ranked[0] if ranked else None

    return {
        "status": "Success",
        "predicted_disease": top["Class"] if top else None,
        "confidence": top["Confidence"] if top else 0.0,
        "classifications": ranked,
        "biomarkers_used": successful,
        "individual_results": individual_results,
        "skipped_biomarkers": skipped
    }


def predict_disease_from_symptoms(symptoms, infection_type):
    clf = load_ml_artifact("rf4_disease_name_symptoms.pkl")
    le_inf = load_ml_artifact("rf4_le_infection_type.pkl")
    le_target = load_ml_artifact("rf4_le_disease_name.pkl")
    infection_type_rf4 = CODE_TO_SYMPTOM.get(infection_type, infection_type)

    if infection_type_rf4 not in le_inf.classes_:
        return {
            "status": "No Classification",
            "predicted_disease": None,
            "confidence": 0.0,
            "classifications": []
        }

    inf_enc = le_inf.transform([infection_type_rf4])[0]
    symptom_vec = [symptoms.get(col, 0) for col in ML_SYMPTOM_COLS]
    x = np.array([symptom_vec + [inf_enc]])
    proba = clf.predict_proba(x)[0]
    prob_dict = {cls: float(probability) for cls, probability in zip(le_target.classes_, proba)}
    ranked = rank_classifications(prob_dict)
    top = ranked[0] if ranked else None

    return {
        "status": "Success" if top else "No Classification",
        "predicted_disease": top["Class"] if top else None,
        "confidence": top["Confidence"] if top else 0.0,
        "classifications": ranked,
        "prob_dict": prob_dict
    }


def predict_disease_combined(biomarker_thresholds, symptoms, infection_type):
    rf2 = predict_disease_from_biomarkers(biomarker_thresholds, infection_type)
    rf4 = predict_disease_from_symptoms(symptoms, infection_type)

    all_classes = sorted(
        set(c["Class"] for c in rf2["classifications"]) |
        set(c["Class"] for c in rf4["classifications"])
    )

    sources = []
    if rf2["status"] == "Success":
        sources.append({c["Class"]: c["Confidence"] / 100 for c in rf2["classifications"]})
    if rf4["status"] == "Success":
        sources.append({c["Class"]: c["Confidence"] / 100 for c in rf4["classifications"]})

    if not sources:
        return {
            "status": "No Classification",
            "predicted_disease": None,
            "confidence": 0.0,
            "classifications": [],
            "rf2_result": rf2,
            "rf4_result": rf4
        }

    normalized = geometric_mean_combine(sources, all_classes)
    ranked = rank_classifications(normalized)
    top = ranked[0] if ranked else None

    return {
        "status": "Success",
        "predicted_disease": top["Class"] if top else None,
        "confidence": top["Confidence"] if top else 0.0,
        "classifications": ranked,
        "rf2_result": rf2,
        "rf4_result": rf4
    }


def run_full_ml_pipeline(biomarker_thresholds, symptoms):
    stage1 = predict_infection_combined(biomarker_thresholds, symptoms)
    infection_type = stage1.get("predicted_infection_type")

    if infection_type is None:
        return {
            "infection_type_result": stage1,
            "disease_result": None,
            "final_infection_type": None,
            "final_disease": None,
            "infection_confidence": 0.0,
            "disease_confidence": 0.0
        }

    stage2 = predict_disease_combined(biomarker_thresholds, symptoms, infection_type)

    return {
        "infection_type_result": stage1,
        "disease_result": stage2,
        "final_infection_type": infection_type,
        "final_disease": stage2.get("predicted_disease"),
        "infection_confidence": stage1["confidence"],
        "disease_confidence": stage2.get("confidence", 0.0)
    }


def classifications_to_dataframe(classifications, class_label, top_n=None):
    shown = classifications if top_n is None else classifications[:top_n]
    return pd.DataFrame([
        {
            "Rank": item["Rank"],
            class_label: infection_type_label(item["Class"]) if class_label == "Infection Type" else item["Class"],
            "Confidence (%)": f"{item['Confidence']:.2f}",
            "Status": status_label(item["Rank"], item["Confidence"])
        }
        for item in shown
    ])


def render_classification_table(classifications, class_label, top_n=None):
    if not classifications:
        st.warning("No classification available.")
        return

    table_df = classifications_to_dataframe(classifications, class_label, top_n=top_n)
    st.dataframe(table_df, hide_index=True, use_container_width=True)

    if len(classifications) >= 2:
        top_conf = classifications[0]["Confidence"]
        second_conf = classifications[1]["Confidence"]
        if abs(top_conf - second_conf) < 10:
            st.warning("⚠️ Close confidence scores detected. This may indicate conflicting evidence or co-infection.")


def render_individual_biomarker_probabilities(individual_results, class_label, top_n=3):
    if not individual_results:
        st.info("No individual biomarker results available.")
        return

    for biomarker, probabilities in individual_results.items():
        ranked = rank_classifications(probabilities)
        st.markdown(f"**{biomarker}**")
        render_classification_table(ranked, class_label, top_n=top_n)


def active_ml_symptom_labels(symptoms):
    return [
        ML_SYMPTOM_LABELS.get(col, col.replace("\n", " ").strip())
        for col, value in symptoms.items()
        if value
    ]


def render_ml_symptom_input(page_prefix=""):
    current_symptoms = st.session_state.get("rf_symptoms", {col: 0 for col in ML_SYMPTOM_COLS})
    selected_symptoms = {}

    for group_idx, (group_name, symptom_cols) in enumerate(ML_SYMPTOM_GROUPS):
        with st.expander(f"{group_name} ({len(symptom_cols)})", expanded=(group_idx == 0)):
            for row_start in range(0, len(symptom_cols), 4):
                cols = st.columns(4)
                for col_idx, symptom_col in enumerate(symptom_cols[row_start:row_start + 4]):
                    label = ML_SYMPTOM_LABELS.get(symptom_col, symptom_col.replace("\n", " ").strip())
                    symptom_idx = ML_SYMPTOM_COLS.index(symptom_col)
                    with cols[col_idx]:
                        selected_symptoms[symptom_col] = 1 if st.checkbox(
                            label,
                            value=bool(current_symptoms.get(symptom_col, 0)),
                            key=f"{page_prefix}symptom_{st.session_state.rf_input_counter}_{symptom_idx}"
                        ) else 0

    for symptom_col in ML_SYMPTOM_COLS:
        selected_symptoms.setdefault(symptom_col, 0)

    if selected_symptoms != current_symptoms:
        st.session_state.rf_submitted = False

    st.session_state.rf_symptoms = selected_symptoms
    active_count = sum(selected_symptoms.values())
    st.caption(f"Selected symptoms: {active_count}")


def render_rf_biomarker_input(page_prefix=""):
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

    with col1:
        available_biomarkers = [b for b in biomarkers if b not in st.session_state.rf_biomarker_thresholds]
        selected_biomarker = st.selectbox(
            "Select Biomarker",
            [""] + available_biomarkers,
            key=f"{page_prefix}rf_biomarker_select_{st.session_state.rf_input_counter}"
        )

    with col2:
        threshold_input = st.number_input(
            "Lab reading value",
            min_value=0.0,
            value=None,
            step=0.01,
            format="%.4f",
            placeholder="Enter value...",
            key=f"{page_prefix}rf_threshold_input_{st.session_state.rf_input_counter}"
        )

    with col3:
        selected_unit = st.selectbox(
            "Unit",
            UNITS_LIST,
            key=f"{page_prefix}rf_unit_select_{st.session_state.rf_input_counter}"
        )

    with col4:
        st.write("")
        st.write("")
        add_button = st.button("➕ Add", use_container_width=True, key=f"{page_prefix}rf_add_btn")

    if add_button and selected_biomarker and threshold_input is not None:
        if selected_biomarker not in st.session_state.rf_biomarker_thresholds:
            st.session_state.rf_biomarker_thresholds[selected_biomarker] = {
                'biomarker': selected_biomarker,
                'value': threshold_input,
                'unit': selected_unit,
                'value_ng_ml': convert_to_ng_ml(threshold_input, selected_unit)
            }
            st.session_state.rf_input_counter += 1
            st.session_state.rf_submitted = False
            st.rerun()
        else:
            st.warning(f"{selected_biomarker} is already added!")
    elif add_button and threshold_input is None:
        st.warning("Please enter a value!")


def display_added_rf_biomarkers(page_prefix=""):
    if st.session_state.rf_biomarker_thresholds:
        st.markdown("### Currently Added Biomarkers:")

        for bio_key in list(st.session_state.rf_biomarker_thresholds.keys()):
            bio_data = st.session_state.rf_biomarker_thresholds[bio_key]
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

            with col1:
                st.write(f"**{bio_data['biomarker']}**")

            with col2:
                new_value = st.number_input(
                    "Value",
                    value=bio_data['value'],
                    min_value=0.0,
                    step=0.01,
                    format="%.4f",
                    key=f"{page_prefix}rf_edit_val_{bio_key}",
                    label_visibility="collapsed"
                )

            with col3:
                new_unit = st.selectbox(
                    "Unit",
                    UNITS_LIST,
                    index=UNITS_LIST.index(bio_data['unit']),
                    key=f"{page_prefix}rf_edit_unit_{bio_key}",
                    label_visibility="collapsed"
                )

            if new_value != bio_data['value'] or new_unit != bio_data['unit']:
                st.session_state.rf_biomarker_thresholds[bio_key] = {
                    'biomarker': bio_data['biomarker'],
                    'value': new_value,
                    'unit': new_unit,
                    'value_ng_ml': convert_to_ng_ml(new_value, new_unit)
                }
                st.session_state.rf_submitted = False

            with col4:
                if st.button("🗑️", key=f"{page_prefix}rf_remove_{bio_key}", use_container_width=True):
                    del st.session_state.rf_biomarker_thresholds[bio_key]
                    st.session_state.rf_submitted = False
                    st.rerun()

            st.caption(f"→ {bio_data['value_ng_ml']:.4f} ng/mL")

        st.write(f"**Total Biomarkers:** {len(st.session_state.rf_biomarker_thresholds)}")

# ============================================================================
# PAGE ROUTING
# ============================================================================

if st.session_state.current_page == "Home":
    st.title("🧬 Biomarker Infection Classifier")
    st.markdown("---")
    
    st.markdown("## Welcome!")
    st.markdown("This tool helps classify infection types based on biomarker laboratory readings.")
    
    st.markdown("### Choose an approach based on how you want to analyze the case.")
    
    method_rows = [
        {
            "title": "📊 Statistical Approach",
            "button": "Open Statistical Approach →",
            "page": "Statistical Approach",
            "description": "Compares biomarker readings against statistical ranges from the reference data. Best for transparent, range-based classification that shows which infections match each biomarker and how multiple biomarkers combine."
        },
        {
            "title": "🌲 Machine Learning Approach",
            "button": "Open Machine Learning Approach →",
            "page": "Machine Learning Approach",
            "description": "Runs the staged Random Forest pipeline using both biomarkers and selected symptoms. It first predicts the infection type, then uses that infection type to rank the most likely common disease names."
        },
        {
            "title": "🤖 LLM-Based Approach",
            "button": "Open LLM-Based Approach →",
            "page": "LLM-Based Approach",
            "description": "Uses biomarker readings and symptoms to generate a clinical-style explanation and interactive follow-up. Best for narrative interpretation, reasoning, and asking additional questions after entering the case details."
        }
    ]

    for method in method_rows:
        title_col, button_col = st.columns([3, 1])
        with title_col:
            st.markdown(f"#### {method['title']}")
        with button_col:
            if st.button(method["button"], use_container_width=True, type="primary", key=f"home_{method['page']}"):
                st.session_state.current_page = method["page"]
                st.rerun()
        st.markdown(method["description"])
        st.markdown("---")
    
    st.markdown("---")
    st.markdown("*Powered by ASU*")

elif st.session_state.current_page == "Statistical Approach":
    st.title("📊 Statistical Approach")
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

elif st.session_state.current_page == "LLM-Based Approach":
    st.title("🤖 LLM-Based Approach")
    st.markdown("---")
    
    if not st.session_state.llm_submitted:
        st.subheader("Add Biomarkers")
        render_biomarker_input("llm_")
        display_added_biomarkers("llm_")
        
        st.markdown("---")
        st.subheader("Select Symptoms")
        
        selected_symptoms = st.multiselect(
            "Choose symptoms from the list:",
            SYMPTOMS_LIST,
            default=st.session_state.selected_symptoms,
            key="symptoms_multiselect"
        )
        st.session_state.selected_symptoms = selected_symptoms
        
        other_symptoms_input = st.text_area(
            "Other symptoms (comma-separated):",
            value=st.session_state.other_symptoms,
            placeholder="e.g., Diarrhea, Rash, Difficulty breathing",
            key="other_symptoms_input"
        )
        st.session_state.other_symptoms = other_symptoms_input
        
        st.markdown("---")
        
        if st.session_state.biomarker_thresholds:
            if st.button("🔬 Classify", type="primary", use_container_width=True, key="llm_classify_btn"):
                st.session_state.llm_submitted = True
                
                biomarker_text = "\n".join([
                    f"{data['biomarker']}: {data['value']} {data['unit']}" 
                    for data in st.session_state.biomarker_thresholds.values()
                ])
                
                all_symptoms = list(st.session_state.selected_symptoms)
                if st.session_state.other_symptoms.strip():
                    other_symp_list = [s.strip() for s in st.session_state.other_symptoms.split(',') if s.strip()]
                    all_symptoms.extend(other_symp_list)
                
                symptoms_text = ", ".join(all_symptoms) if all_symptoms else "None reported"
                
                user_message = f"<thresholds>\n{biomarker_text}\n</thresholds>\n\nSymptoms: {symptoms_text}"
                
                st.session_state.llm_chat_history.append({
                    'role': 'user',
                    'content': user_message
                })
                
                st.rerun()
        else:
            st.info("👆 Please add at least one biomarker to begin classification.")
    
    else:
        st.subheader("💬 Classification Analysis")
        
        for idx, message in enumerate(st.session_state.llm_chat_history):
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(f"```\n{message['content']}\n```")
            else:
                with st.chat_message("assistant"):
                    st.markdown(message['content'])
        
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
        
        user_input = st.chat_input("Ask a follow-up question...")
        
        if user_input:
            st.session_state.llm_chat_history.append({
                'role': 'user',
                'content': user_input
            })
            st.rerun()
        
        st.markdown("---")
        if st.button("🔄 New Classification", use_container_width=True):
            st.session_state.llm_submitted = False
            st.session_state.llm_chat_history = []
            st.session_state.biomarker_thresholds = {}
            st.session_state.selected_symptoms = []
            st.session_state.other_symptoms = ""
            st.session_state.input_counter += 1
            st.rerun()

elif st.session_state.current_page == "Machine Learning Approach":
    st.title("🌲 Machine Learning Approach")
    st.markdown("---")

    st.subheader("Add Biomarkers")
    render_rf_biomarker_input("rf_")
    display_added_rf_biomarkers("rf_")

    st.markdown("---")
    st.subheader("Select Symptoms")
    render_ml_symptom_input("rf_")

    st.markdown("---")
    col_btn1, col_btn2 = st.columns([2, 1])

    with col_btn1:
        classify_button = st.button(
            "🔬 Run Machine Learning Pipeline",
            type="primary",
            use_container_width=True,
            key="rf_classify_btn",
            disabled=not st.session_state.rf_biomarker_thresholds
        )

    with col_btn2:
        if st.button("🔄 New Classification", use_container_width=True, key="rf_new_btn"):
            st.session_state.rf_biomarker_thresholds = {}
            st.session_state.rf_submitted = False
            st.session_state.rf_individual_results = {}
            st.session_state.rf_combined_result = None
            st.session_state.rf_pipeline_result = None
            st.session_state.rf_symptoms = {col: 0 for col in ML_SYMPTOM_COLS}
            st.session_state.rf_input_counter += 1
            st.rerun()

    if not st.session_state.rf_biomarker_thresholds:
        st.info("👆 Please add at least one biomarker to begin classification.")

    if classify_button:
        ml_biomarker_thresholds = {
            bio_data["biomarker"]: bio_data["value_ng_ml"]
            for bio_data in st.session_state.rf_biomarker_thresholds.values()
        }

        try:
            st.session_state.rf_pipeline_result = run_full_ml_pipeline(
                ml_biomarker_thresholds,
                st.session_state.rf_symptoms
            )
            st.session_state.rf_submitted = True
            st.rerun()
        except FileNotFoundError as e:
            st.session_state.rf_submitted = False
            st.session_state.rf_pipeline_result = None
            st.error(f"❌ Missing model file: {os.path.basename(str(e).strip())}")
        except Exception as e:
            st.session_state.rf_submitted = False
            st.session_state.rf_pipeline_result = None
            st.error(f"❌ Could not run the machine learning pipeline: {e}")

    if st.session_state.rf_submitted and st.session_state.rf_pipeline_result:
        pipeline_result = st.session_state.rf_pipeline_result
        infection_result = pipeline_result["infection_type_result"]
        rf1_result = infection_result.get("rf1_result", {})
        rf3_result = infection_result.get("rf3_result", {})
        symptoms_selected = active_ml_symptom_labels(st.session_state.rf_symptoms)

        st.markdown("---")

        with st.expander("Infection Type from Biomarkers", expanded=False):
            if rf1_result.get("status") == "Success":
                st.success(
                    f"Prediction: **{infection_type_label(rf1_result['predicted_infection_type'])}** "
                    f"({rf1_result['confidence']:.2f}% confidence)"
                )
            else:
                st.warning("No biomarker-based infection type classification available.")

            skipped = rf1_result.get("skipped_biomarkers", [])
            if skipped:
                st.warning(f"Skipped biomarkers: {', '.join(skipped)}")

            st.markdown("#### Ranked Infection Types")
            render_classification_table(rf1_result.get("classifications", []), "Infection Type")
            st.markdown("#### Per-Biomarker Results")
            render_individual_biomarker_probabilities(
                rf1_result.get("individual_results", {}),
                "Infection Type",
                top_n=3
            )

        with st.expander("Infection Type from Symptoms", expanded=False):
            if symptoms_selected:
                st.markdown("**Active symptoms:** " + ", ".join(symptoms_selected))
            else:
                st.info("No symptoms selected.")

            if rf3_result.get("status") == "Success":
                st.success(
                    f"Prediction: **{infection_type_label(rf3_result['predicted_infection_type'])}** "
                    f"({rf3_result['confidence']:.2f}% confidence)"
                )
            else:
                st.warning("No symptom-based infection type classification available.")

            render_classification_table(rf3_result.get("classifications", []), "Infection Type")

        st.subheader("Combined Prediction for Infection Type")
        with st.container(border=True):
            if infection_result.get("status") == "Success":
                st.success(
                    f"### {infection_type_label(infection_result['predicted_infection_type'])} "
                    f"({infection_result['confidence']:.2f}% confidence)"
                )
                render_classification_table(infection_result.get("classifications", []), "Infection Type")
            else:
                st.error("Could not produce a combined infection type prediction.")

        disease_result = pipeline_result.get("disease_result")

        if disease_result:
            rf2_result = disease_result.get("rf2_result", {})
            rf4_result = disease_result.get("rf4_result", {})

            with st.expander("Common Disease from Biomarkers + Infection Type", expanded=False):
                if rf2_result.get("status") == "Success":
                    st.success(
                        f"Prediction: **{rf2_result['predicted_disease']}** "
                        f"({rf2_result['confidence']:.2f}% confidence)"
                    )
                else:
                    st.warning("No biomarker-based common disease classification available.")

                skipped = rf2_result.get("skipped_biomarkers", [])
                if skipped:
                    st.warning(f"Skipped biomarkers: {', '.join(skipped)}")

                st.markdown("#### Top Common Disease Names")
                render_classification_table(
                    rf2_result.get("classifications", []),
                    "Common Disease Name",
                    top_n=TOP_N_DISEASE_PRINT
                )
                st.markdown("#### Per-Biomarker Results")
                render_individual_biomarker_probabilities(
                    rf2_result.get("individual_results", {}),
                    "Common Disease Name",
                    top_n=TOP_N_DISEASE_PRINT
                )

            with st.expander("Common Disease from Symptoms + Infection Type", expanded=False):
                if symptoms_selected:
                    st.markdown("**Active symptoms:** " + ", ".join(symptoms_selected))
                else:
                    st.info("No symptoms selected.")

                if rf4_result.get("status") == "Success":
                    st.success(
                        f"Prediction: **{rf4_result['predicted_disease']}** "
                        f"({rf4_result['confidence']:.2f}% confidence)"
                    )
                else:
                    st.warning("No symptom-based common disease classification available.")

                render_classification_table(
                    rf4_result.get("classifications", []),
                    "Common Disease Name",
                    top_n=TOP_N_DISEASE_PRINT
                )

            st.subheader("Combined Prediction for Common Disease Name")
            with st.container(border=True):
                if disease_result.get("status") == "Success":
                    st.success(
                        f"### {disease_result['predicted_disease']} "
                        f"({disease_result['confidence']:.2f}% confidence)"
                    )
                    render_classification_table(
                        disease_result.get("classifications", []),
                        "Common Disease Name",
                        top_n=TOP_N_DISEASE_PRINT
                    )
                else:
                    st.error("Could not produce a combined common disease prediction.")
        else:
            st.error("Stage 1 produced no infection type, so common disease prediction was not run.")

    st.markdown("---")
    st.markdown("*Powered by ASU*")
