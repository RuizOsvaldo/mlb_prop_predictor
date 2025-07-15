import streamlit as st
from datetime import datetime
import statsapi
from pybaseball import batting_stats, pitching_stats, playerid_lookup
import pandas as pd

def get_schedule(date):
    return statsapi.schedule(date=date)

def get_boxscore(game_id):
    return statsapi.get('game_boxscore', {'gamePk': game_id})

def extract_starters(players):
    hitters = []
    pitcher = None
    for p in players.values():
        person = p.get('person', {})
        position = p.get('position', {}).get('abbreviation', 'NA')
        pid = person.get('id')
        name = person.get('fullName')
        bo = p.get('battingOrder')
        if bo and int(bo) > 0:
            hitters.append({
                'id': pid,
                'name': name,
                'position': position,
                'batting_order': int(bo) // 100
            })
        elif position == 'P':
            pitcher = {
                'id': pid,
                'name': name,
                'position': 'P'
            }
    hitters.sort(key=lambda x: x['batting_order'])
    return hitters, pitcher

def find_mlbam_id(name):
    parts = name.split()
    if len(parts) < 2:
        return None
    first_name = parts[0]
    last_name = parts[-1]
    try:
        lookup = playerid_lookup(last_name, first_name)
        if not lookup.empty:
            return lookup.iloc[0]['key_mlbam']
    except:
        return None
    return None

def get_player_stats(name, bat_df, pit_df, is_pitcher=False):
    mlbam_id = find_mlbam_id(name)
    if is_pitcher:
        stats_row = pit_df[pit_df['IDfg'] == mlbam_id] if mlbam_id else pd.DataFrame()
        if not stats_row.empty:
            stats = stats_row.iloc[0]
            return {
                'IP': stats['IP'],
                'ERA': stats['ERA'],
                'FIP': stats['FIP'],
                'WHIP': stats['WHIP'],
                'K/9': stats['K/9'],
                'BB/9': stats['BB/9'],
                'WAR': stats['WAR']
            }
    else:
        stats_row = bat_df[bat_df['IDfg'] == mlbam_id] if mlbam_id else pd.DataFrame()
        if not stats_row.empty:
            stats = stats_row.iloc[0]
            return {
                'PA': stats['PA'],
                'AVG': stats['AVG'],
                'OBP': stats['OBP'],
                'SLG': stats['SLG'],
                'OPS': stats['OPS'],
                'wOBA': stats['wOBA'],
                'wRC+': stats['wRC+']
            }
    return None

st.title("MLB Starting Lineups & Sabermetrics")

date = st.date_input("Select date", value=datetime.now())
year = date.year

bat_df = batting_stats(year)
pit_df = pitching_stats(year)

schedule = get_schedule(date.strftime('%Y-%m-%d'))
if not schedule:
    st.info("No games scheduled for this date.")
else:
    for game in schedule:
        game_id = game['game_id']
        boxscore = get_boxscore(game_id)
        teams = boxscore['teams']
        for side in ['away', 'home']:
            team_name = teams[side]['team']['name']
            players = teams[side]['players']
            hitters, pitcher = extract_starters(players)
            st.subheader(f"{team_name} Lineup")
            lineup_data = []
            for h in hitters:
                stats = get_player_stats(h['name'], bat_df, pit_df, is_pitcher=False)
                if stats:
                    lineup_data.append({
                        "Order": h['batting_order'],
                        "Name": h['name'],
                        "Position": h['position'],
                        **stats
                    })
                else:
                    lineup_data.append({
                        "Order": h['batting_order'],
                        "Name": h['name'],
                        "Position": h['position'],
                        "PA": "N/A", "AVG": "N/A", "OBP": "N/A", "SLG": "N/A", "OPS": "N/A", "wOBA": "N/A", "wRC+": "N/A"
                    })
            st.dataframe(pd.DataFrame(lineup_data))
            if pitcher:
                stats = get_player_stats(pitcher['name'], bat_df, pit_df, is_pitcher=True)
                if stats:
                    st.write(f"**SP:** {pitcher['name']} ({pitcher['position']})")
                    st.write(stats)
                else:
                    st.write(f"**SP:** {pitcher['name']} ({pitcher['position']}) - No stats found")