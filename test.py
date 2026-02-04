import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np

@st.cache_resource
def load_statistics():
    with open("biomarker_statistics.pkl", "rb") as f:
        return pickle.load(f)

stats_dict = load_statistics()
biomarkers = sorted(stats_dict.keys())


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


def classify_infection_bayesian_geometric(biomarker_thresholds, stats_dict, verbose=True):
    first_biomarker = list(biomarker_thresholds.keys())[0]
    if first_biomarker not in stats_dict:
        if verbose:
            st.error(f"Error: No statistics found for biomarker: {first_biomarker}")
        return None

    all_infections = stats_dict[first_biomarker]['Infection'].unique()
    infection_probs = {infection: 1.0 for infection in all_infections}

    successful_biomarkers = []
    individual_results = {}

    for biomarker, threshold_value in biomarker_thresholds.items():
        if biomarker not in stats_dict:
            if verbose:
                st.warning(f"No statistics for {biomarker}, skipping...")
            continue
        result = classify_infection(biomarker, threshold_value, stats_dict, verbose=False)
        if result is None or result['Total_Matches'] == 0:
            if verbose:
                st.warning(f"No matches found for {biomarker}={threshold_value}, skipping...")
            continue

        individual_results[biomarker] = result
        successful_biomarkers.append(biomarker)

        biomarker_probs = {}
        for match in result['Matches']:
            biomarker_probs[match['Infection']] = match['Confidence'] / 100.0

        for infection in all_infections:
            if infection in biomarker_probs:
                infection_probs[infection] *= biomarker_probs[infection]
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
            'Method': 'Bayesian Probability (Geometric Mean)',
            'Biomarkers_Used': successful_biomarkers,
            'Total_Biomarkers': len(biomarker_thresholds),
            'Classifications': [],
            'Status': 'No Classification'
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
            'Method': 'Bayesian Probability (Geometric Mean)',
            'Biomarkers_Used': successful_biomarkers,
            'Total_Biomarkers': len(biomarker_thresholds),
            'Individual_Results': individual_results,
            'Classifications': classifications,
            'Status': 'Success',
            'Note': f'Geometric mean applied (N={n_biomarkers})'
        }

    return result


def plot_classification_ranges(biomarker, threshold_value, stats_dict, classification_result):
    stats_df = stats_dict[biomarker]
    infections = stats_df['Infection'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(infections)))
    infection_colors = dict(zip(infections, colors))
    matched = [m['Infection'] for m in classification_result['Matches']]

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
            ax.text(low - (high-low)*0.05, i, inf, ha='right', va='center', fontsize=9)

        ax.axvline(threshold_value, linestyle='--', linewidth=2, color='red', label=f'Threshold: {threshold_value}')
        ax.set_title(f"{mode} Range", fontsize=11, fontweight='bold')
        ax.set_xlabel('Threshold (ng/mL)', fontsize=10)
        ax.set_yticks([])
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"{biomarker} Classification", fontsize=13, fontweight='bold')
    return fig


# Initialize session state
if 'biomarker_thresholds' not in st.session_state:
    st.session_state.biomarker_thresholds = {}

# Streamlit UI
st.set_page_config(page_title="Biomarker Infection Classifier", layout="wide")
st.title("🧬 Biomarker-based Infection Classification")

st.markdown("---")

# Input Section
st.subheader("📊 Add Biomarkers")

col1, col2, col3 = st.columns([3, 2, 1])

with col1:
    # Filter out already added biomarkers
    available_biomarkers = [b for b in biomarkers if b not in st.session_state.biomarker_thresholds]
    selected_biomarker = st.selectbox("Select Biomarker", [""] + available_biomarkers, key="biomarker_select")

with col2:
    threshold_input = st.number_input("Threshold (ng/mL)", min_value=0.0, step=0.01, format="%.4f", key="threshold_input")

with col3:
    st.write("")  # Spacer
    st.write("")  # Spacer
    add_button = st.button("➕ Add", use_container_width=True)

# Add biomarker to session state
if add_button and selected_biomarker:
    if selected_biomarker not in st.session_state.biomarker_thresholds:
        st.session_state.biomarker_thresholds[selected_biomarker] = threshold_input
        st.rerun()
    else:
        st.warning(f"{selected_biomarker} is already added!")

# Display currently added biomarkers
if st.session_state.biomarker_thresholds:
    st.markdown("### Currently Added Biomarkers:")
    
    for biomarker in list(st.session_state.biomarker_thresholds.keys()):
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.write(f"**{biomarker}**")
        
        with col2:
            # Editable threshold value
            new_value = st.number_input(
                f"Value", 
                value=st.session_state.biomarker_thresholds[biomarker],
                min_value=0.0,
                step=0.01,
                format="%.4f",
                key=f"edit_{biomarker}",
                label_visibility="collapsed"
            )
            # Update if changed
            if new_value != st.session_state.biomarker_thresholds[biomarker]:
                st.session_state.biomarker_thresholds[biomarker] = new_value
        
        with col3:
            if st.button("🗑️ Remove", key=f"remove_{biomarker}", use_container_width=True):
                del st.session_state.biomarker_thresholds[biomarker]
                st.rerun()
    
    st.write(f"**Total Biomarkers:** {len(st.session_state.biomarker_thresholds)}")

st.markdown("---")

# Classification Button
if st.session_state.biomarker_thresholds:
    if st.button("🔬 Classify Infection", type="primary", use_container_width=True):
        
        num_biomarkers = len(st.session_state.biomarker_thresholds)
        
        # Single biomarker - use old logic
        if num_biomarkers == 1:
            biomarker = list(st.session_state.biomarker_thresholds.keys())[0]
            threshold_value = st.session_state.biomarker_thresholds[biomarker]
            
            st.subheader("🔍 Classification Results")
            
            result = classify_infection(biomarker, threshold_value, stats_dict, verbose=False)
            
            st.markdown(f"**Biomarker:** {biomarker}")
            st.markdown(f"**Threshold:** {threshold_value} ng/mL")
            st.markdown(f"**Classification Method:** {result['Classification_Method']}")
            
            if result['Total_Matches'] == 0:
                st.warning("⚠️ No matching infection found.")
            else:
                st.success(f"✅ Found {result['Total_Matches']} matching infection(s)")
                
                # Display matches
                st.markdown("#### Infection Matches:")
                for i, match in enumerate(result['Matches'], 1):
                    st.markdown(f"{i}. **{match['Infection']}** — {match['Confidence']:.2f}% confidence")
            
            # Plot
            st.markdown("---")
            st.markdown("#### Visualization")
            fig = plot_classification_ranges(biomarker, threshold_value, stats_dict, result)
            st.pyplot(fig, clear_figure=True)
        
        # Multiple biomarkers - use new multi-biomarker logic
        else:
            st.subheader("🔍 Individual Biomarker Results")
            
            # Store individual results
            individual_results = {}
            
            # Create columns for 2-per-row layout
            biomarker_list = list(st.session_state.biomarker_thresholds.keys())
            
            for idx in range(0, len(biomarker_list), 2):
                cols = st.columns(2)
                
                for col_idx, col in enumerate(cols):
                    biomarker_idx = idx + col_idx
                    if biomarker_idx < len(biomarker_list):
                        biomarker = biomarker_list[biomarker_idx]
                        threshold_value = st.session_state.biomarker_thresholds[biomarker]
                        
                        with col:
                            # Create a styled container
                            with st.container(border=True):
                                st.markdown(f"### 🧪 {biomarker}")
                                st.markdown(f"**Threshold:** {threshold_value} ng/mL")
                                
                                # Classify
                                result = classify_infection(biomarker, threshold_value, stats_dict, verbose=False)
                                individual_results[biomarker] = result
                                
                                # Display results
                                st.markdown(f"**Method:** {result['Classification_Method']}")
                                
                                if result['Total_Matches'] == 0:
                                    st.warning("No matches found")
                                else:
                                    st.success(f"{result['Total_Matches']} match(es)")
                                    for i, match in enumerate(result['Matches'][:3], 1):  # Top 3
                                        st.markdown(f"{i}. **{match['Infection']}**: {match['Confidence']:.2f}%")
                                
                                # Plot
                                fig = plot_classification_ranges(biomarker, threshold_value, stats_dict, result)
                                st.pyplot(fig, clear_figure=True)
            
            # Combined Multi-Biomarker Classification
            st.markdown("---")
            st.subheader("🎯 Combined Multi-Biomarker Classification")
            st.markdown("*Using Bayesian Probability with Geometric Mean*")
            
            # Run multi-biomarker classification
            combined_result = classify_infection_bayesian_geometric(
                st.session_state.biomarker_thresholds, 
                stats_dict, 
                verbose=False
            )
            
            if combined_result and combined_result['Status'] == 'Success':
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("#### 📋 Summary")
                    st.markdown(f"**Method:** {combined_result['Method']}")
                    st.markdown(f"**Biomarkers Used:** {len(combined_result['Biomarkers_Used'])} / {combined_result['Total_Biomarkers']}")
                    st.markdown(f"**Note:** {combined_result.get('Note', 'N/A')}")
                
                with col2:
                    st.markdown("#### 🏆 Final Classification Results")
                    
                    # Create a results table
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
                
                # Highlight top prediction
                if combined_result['Classifications']:
                    top_infection = combined_result['Classifications'][0]['Infection']
                    top_confidence = combined_result['Classifications'][0]['Confidence']
                    st.success(f"### 🎉 Predicted Infection: **{top_infection}** with {top_confidence:.2f}% confidence")
            else:
                st.error("❌ Could not perform multi-biomarker classification. Please check your inputs.")
else:
    st.info("👆 Please add at least one biomarker to begin classification.")

# Footer
st.markdown("---")
st.markdown("*Powered by Bayesian Probability & Statistical Range Analysis*")
