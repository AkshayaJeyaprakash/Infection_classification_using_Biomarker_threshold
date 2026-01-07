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


def plot_classification_ranges(biomarker, threshold_value, stats_dict, classification_result):
    stats_df = stats_dict[biomarker]
    infections = stats_df['Infection'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(infections)))
    infection_colors = dict(zip(infections, colors))
    matched = [m['Infection'] for m in classification_result['Matches']]

    fig, axes = plt.subplots(2, 1, figsize=(16, 12), constrained_layout=True)

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
            ax.text(low - (high-low)*0.05, i, inf, ha='right', va='center')

        ax.axvline(threshold_value, linestyle='--', linewidth=3)
        ax.set_title(f"{mode} Range Classification")
        ax.set_yticks([])
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    fig.suptitle(f"Infection Classification for {biomarker}", fontsize=16)
    return fig


# Streamlit UI
st.set_page_config(page_title="Biomarker Infection Classifier", layout="wide")
st.title("🧬 Biomarker-based Infection Classification")

biomarker = st.selectbox("Select Biomarker", biomarkers)
threshold_value = st.number_input("Unified Threshold (ng/mL)", step=0.01, format="%.4f")

if st.button("Classify Infection"):
    result = classify_infection(biomarker, threshold_value, stats_dict)

    st.markdown(f"**Classification Range Used: {result['Classification_Method']}**")

    st.subheader("🔍 Results")
    if result['Total_Matches'] == 0:
        st.warning("No matching infection found.")
    else:
        for m in result['Matches']:
            st.write(f"**{m['Infection']}** — {m['Confidence']:.2f}% confidence")

    fig = plot_classification_ranges(biomarker, threshold_value, stats_dict, result)
    st.pyplot(fig, clear_figure=True)

