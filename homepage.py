import streamlit as st

st.set_page_config(
    page_title="Player Analysis Dashboard",
    page_icon="⚽",
    layout="wide"
)

st.title("Player Analysis Dashboard")

st.markdown("""
Welcome to the interactive player analysis tool by Shirwa Abdullahi! \n
This tool was created to analyse the 2024-2025 football season in Europe's top five leagues: Premier League (England), Ligue 1 (France), LaLiga (Spain), Serie A (Italy), and Bundesliga (Germany). \n
Use the sidebar to navigate between:
- **PCA Visualiser & Player Similarity**: Explore clusters and compare player stats, as well as find the five most similar players to a chosen player.
- **Radar Charts**: View cluster performance profiles by position.
        \n

Some example uses for this tool:
- Quickly identifying players that have similar styles or profiles
- Scouting replacements for the 2025/2026 season.
- Comparing players across leagues
- Categorising players into certain archetypes
- Discover players who are underrated
- Tactical planning for the season ahead
- Presenting data in a visual format in order to explain complex player data to fans, coaches, analysts etc.

And many more. Enjoy!
            

**How to use the PCA Visualiser and Player Similarity Tool:**
- Search for a player's name using the search bar provided in the tab (by first name, last name, or both). 
- This will bring up a 'Select Player' dropdown, where players with matching names (or parts of it) can be selected, regardless of position. 
- Selected players will be highlighted with a **red star**. The five most similar players to that player will be highlighted in red text with a black circle corresponding to their data point on the PCA plot.
- A table below will appear showing the player's stats for that season (all per-90 stats).
- Below that, a table showing the five most similar players to that selected player will appear, alongside all their stats.
- Note: Before searching for a player, the PCA plot will show the three players closest to the centre and the three players furthest away per cluster. When a player is found, these players will disappear and only show the selected player and the five similar players. When the name is cleared from the search bar, the plot will return to its original state.   
                """)

