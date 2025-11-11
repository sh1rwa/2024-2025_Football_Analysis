import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from adjustText import adjust_text
import unicodedata

# ---------------------------
# Configuration
# ---------------------------
st.set_page_config(page_title="PCA Cluster & Similarity", layout="wide")
st.title("📈 PCA Cluster Visualizer & Player Similarity")

# ---------------------------
# Load Data
# ---------------------------

norm_df = pd.read_csv("data/final_df.csv")       # normalized data (PCA, clusters)
raw_df = pd.read_csv("data/final_df_raw.csv")    # raw stats data (no PCA)

# Recreate PCA1 and PCA2 if missing
if not {'PCA1', 'PCA2'}.issubset(norm_df.columns): 

    per90_cols_numeric = [
        'Goals/90', 'Assists/90', '(Goals+Assists)/90', 'Non-Penalty Goals/90',
        '(Non-Penalty Goals + Assists)/90', 'Expected Goals/90', 'Expected Assists/90',
        '(Expected Goals + Expected Assists)/90', 'Non-Penalty Expected Goals/90',
        '(Non-Penalty Expected Goals + Expected Assists)/90', 'Progressive Carries',
        'Progressive Passes', 'Progressive Passes Received'
    ]
    
    # Defensive coding: ensure only numeric columns used
    X = norm_df[per90_cols_numeric].select_dtypes(include=[np.number]).fillna(0).values
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X)
    norm_df['PCA1'] = pca_coords[:, 0]
    norm_df['PCA2'] = pca_coords[:, 1]

# ---------------------------
# Helper Function
# ---------------------------
def remove_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

# Add stripped player names for both datasets
for df in [norm_df, raw_df]:
    df['Player_stripped'] = df['Player'].apply(lambda x: remove_accents(x).lower())

# ---------------------------
# Configuration for clusters
# ---------------------------
custom_names = {
    'Attacker': ['Elite Goalscorer', 'Reliable Goal-Volume Forward', 'Elite Playmaker', 'Defensive/Utility Attacker'],
    'Defender': ['Anchors & Pure CBs', 'Build-up Defenders', 'High-Progression Wing-Back',
                 'Defensive/Balanced Full-Back', 'Aggressive Ball-Carrying Defender'],
    'Midfielder': ['Anchor-Engine Hybrid', 'Midfield Controllers', 'Elite Advanced Playmakers',
                   'Utility/Secondary Goal Threat']
}

# ---------------------------
# Sidebar filters
# ---------------------------
player_search_input = st.sidebar.text_input("Search Player (optional)")
positions = norm_df['Group'].unique().tolist()
selected_group = st.sidebar.selectbox("Select Position (used if no search)", positions)

group_df = norm_df[norm_df['Group'] == selected_group].copy()
player_row = pd.DataFrame()
player_row_raw = pd.DataFrame()
similar_players_df = pd.DataFrame()
similar_players_raw = pd.DataFrame()

# ---------------------------
# Player search + suggestions
# ---------------------------
if player_search_input:
    search_str = remove_accents(player_search_input).lower()
    suggestions = norm_df[norm_df['Player_stripped'].str.contains(search_str)]['Player'].tolist()

    if suggestions:
        player_search = st.sidebar.selectbox("Select Player", suggestions)
        player_row = norm_df[norm_df['Player'] == player_search]
        player_row_raw = raw_df[raw_df['Player'] == player_search]

        # Update PCA visualization group
        selected_group = player_row['Group'].values[0]
        group_df = norm_df[norm_df['Group'] == selected_group].copy()

        # Compute top 5 most similar players (based on normalized stats)
        per90_cols_numeric = [
            'Goals/90', 'Assists/90', '(Goals+Assists)/90', 'Non-Penalty Goals/90',
            '(Non-Penalty Goals + Assists)/90', 'Expected Goals/90', 'Expected Assists/90',
            '(Expected Goals + Expected Assists)/90', 'Non-Penalty Expected Goals/90',
            '(Non-Penalty Expected Goals + Expected Assists)/90', 'Progressive Carries',
            'Progressive Passes', 'Progressive Passes Received'
        ]

        X = norm_df[per90_cols_numeric].values
        similarity_matrix = cosine_similarity(X)

        player_idx = norm_df.index[norm_df['Player'] == player_search][0]
        sim_scores = similarity_matrix[player_idx]

        similar_idxs = sim_scores.argsort()[::-1]
        similar_idxs = [i for i in similar_idxs if i != player_idx][:5]


        similar_players = norm_df.iloc[similar_idxs]['Player'].tolist()
        similar_players_df = norm_df[norm_df['Player'].isin(similar_players)]
        similar_players_raw = raw_df[raw_df['Player'].isin(similar_players)]

    else:
        st.sidebar.warning("No players match your input. Showing selected position PCA.")

# ---------------------------
# PCA Visualization
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 7))
all_texts = []

player_selected = not player_row.empty

if not group_df.empty and 'Cluster' in group_df.columns:
    centroids = group_df.groupby('Cluster')[['PCA1', 'PCA2']].mean()

    for cluster_id, cluster_data in group_df.groupby('Cluster'):
        cluster_label = custom_names[selected_group][cluster_id]
        ax.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=cluster_label, alpha=0.6)

        centroid = centroids.loc[cluster_id].values.reshape(1, -1)
        ax.scatter(*centroid.T, color='black', marker='X', s=80)

        if not player_selected:
            # --- Closest 3 to centroid ---
            distances = np.linalg.norm(cluster_data[['PCA1','PCA2']].values - centroid, axis=1)
            closest_idx = distances.argsort()[:3]
            for idx in closest_idx:
                row = cluster_data.iloc[idx]
                txt = ax.text(row['PCA1'], row['PCA2'], row['Player'], fontsize=9, color='black', weight='bold')
                all_texts.append(txt)  # leader line will be added via adjust_text

            # --- Farthest 3 from centroid ---
            farthest_idx = distances.argsort()[-3:]
            for idx in farthest_idx:
                row = cluster_data.iloc[idx]
                txt = ax.text(row['PCA1'], row['PCA2'], row['Player'], fontsize=8, color='black', style='italic')
                all_texts.append(txt)

# After all text objects added
adjust_text(all_texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))


# --- Highlight searched player ---
if not player_row.empty:
    cluster_name = custom_names[selected_group][player_row['Cluster'].values[0]]
    ax.scatter(player_row['PCA1'], player_row['PCA2'], color='red', s=150, marker='*',
               label=f"{player_row['Player'].values[0]} ({cluster_name})")
    for _, row in player_row.iterrows():
        txt = ax.text(row['PCA1'], row['PCA2'], row['Player'], fontsize=10, color='blue', weight='bold')
        all_texts.append(txt)

# --- Highlight top 5 similar players ---
if not similar_players_df.empty:
    ax.scatter(similar_players_df['PCA1'], similar_players_df['PCA2'], color='black', label='Top 5 Similar')
    for _, row in similar_players_df.iterrows():
        txt = ax.text(row['PCA1'], row['PCA2'], row['Player'], fontsize=9, color='black') 
        all_texts.append(txt)

adjust_text(all_texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
ax.set_title(f'{selected_group} Clusters + Similarity', fontsize=14)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.legend()
st.pyplot(fig)

# ---------------------------
# Display Stats
# ---------------------------
def reorder_columns(df):
    all_cols = df.columns.tolist()
    front_cols = ['Player', 'Club', 'League']
    end_cols = ['Group']
    middle_cols = [c for c in all_cols if c not in front_cols + end_cols]
    return df[front_cols + middle_cols + end_cols]

def add_cluster_name_column(df):
    df = df.copy()
    df['Cluster Name'] = df.apply(
        lambda row: custom_names.get(row['Group'], ['Unknown'])[row['Cluster']]
        if (row['Group'] in custom_names and row['Cluster'] < len(custom_names[row['Group']])) 
        else "Unknown", 
        axis=1
    )
    return df

if not player_row_raw.empty: 
    cluster_name = custom_names[selected_group][player_row['Cluster'].values[0]] 
    st.subheader(f"📊 {player_row['Player'].values[0]} Stats — {cluster_name} ({selected_group})") 
    st.dataframe(reorder_columns(player_row_raw).drop(columns=['Player_stripped'])) 
    
if not similar_players_raw.empty: 
    st.subheader(f"🧩 Top 5 Most Similar Players to {player_row['Player'].values[0]}") 
    st.dataframe(reorder_columns(similar_players_raw).drop(columns=['Player_stripped']).reset_index(drop=True))

