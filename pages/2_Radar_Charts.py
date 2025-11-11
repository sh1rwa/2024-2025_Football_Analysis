import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

final_df = pd.read_csv("data/final_df.csv")

st.title("🕸️ Radar Charts by Cluster")

# Define columns used in radar
# Adjusted to match your columns exactly
per90_cols = ['Goals/90', 'Assists/90', '(Goals+Assists)/90', 'Non-Penalty Goals/90',
              '(Non-Penalty Goals + Assists)/90', 'Expected Goals/90',
              'Expected Assists/90', '(Expected Goals + Expected Assists)/90',
              'Non-Penalty Expected Goals/90',
              '(Non-Penalty Expected Goals + Expected Assists)/90',
              'Progressive Carries', 'Progressive Passes', 'Progressive Passes Received']

custom_names = {
    'Attacker': ['Elite Goalscorer', 'Reliable Goal-Volume Forward', 'Elite Playmaker', 'Defensive/Utility Attacker'],
    'Defender': ['Anchors & Pure CBs', 'Build-up Defenders', 'High-Progression Wing-Back', 'Defensive/Balanced Full-Back', 'Aggressive Ball-Carrying Defender'],
    'Midfielder': ['Anchor-Engine Hybrid', 'Midfield Controllers', 'Elite Advanced Playmakers', 'Utility/Secondary Goal Threat']
}

# Position filter
positions = final_df['Group'].unique().tolist()
selected_position = st.selectbox("Select Position Group", positions)

# Get cluster summaries
cluster_summary = final_df.groupby(['Group', 'Cluster'])[per90_cols].mean().round(2)

# Generate radar plots for selected group
group_clusters = cluster_summary.loc[selected_position]

for cluster in group_clusters.index:
    stats = group_clusters.loc[cluster]
    labels = stats.index.tolist()          # radar labels
    values = stats.values.tolist()         # radar values

    # Close the loop
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # Create radar chart
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # Correct label placement
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)

    # Optional: remove y-axis labels for clarity
    ax.set_yticklabels([])

    # Title with cluster info
    
    cluster_name = custom_names[selected_position][cluster]
    ax.set_title(f'{selected_position} - {cluster_name}', fontsize=13, fontweight='bold')

    # Render in Streamlit
    st.pyplot(fig)