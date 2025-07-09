import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import statsapi
from datetime import datetime
from pybaseball import statcast, playerid_lookup, cache
import pandas as pd

cache.enable()

def get_today_schedule():
    today = datetime.now().strftime('%Y-%m-%d')
    return statsapi.schedule(date=today)

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
                'role': 'hitter',
                'batting_order': int(bo) // 100
            })
        elif position == 'P':
            pitcher = {
                'id': pid,
                'name': name,
                'position': 'P',
                'role': 'pitcher'
            }
    hitters.sort(key=lambda x: x['batting_order'])
    return hitters + ([pitcher] if pitcher else [])

def get_player_handedness(pid):
    try:
        data = statsapi.get('person', {'personId': pid})
        return data.get('batSide', {}).get('code', 'NA'), data.get('pitchHand', {}).get('code', 'NA')
    except:
        return 'NA', 'NA'

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

def aggregate_and_print_all_stats(statcast_df, player_col, mlbam_id):
    df_player = statcast_df[statcast_df[player_col] == mlbam_id]
    if df_player.empty:
        print("   ⚠️ No Statcast data found.")
        return

    numeric_cols = df_player.select_dtypes(include=['number']).columns
    print("   Available aggregated stats:")
    for col in numeric_cols:
        val = df_player[col].mean()
        if pd.isna(val):
            val_str = 'N/A'
        else:
            val_str = round(val, 3)
        print(f"     - {col}: {val_str}")

def process_lineup(team, players, selected, statcast_df):
    if selected != 'all' and selected not in team.lower():
        return
    print(f"\n🔷 {team} Starters:")
    starters = extract_starters(players)
    if not starters:
        print("   ⚠️ No starters found.")
        return
    for p in starters:
        bat, thr = get_player_handedness(p['id'])
        mlbam_id = find_mlbam_id(p['name'])
        if mlbam_id is None:
            print(f"\n👤 {p['name']} - MLBAM ID not found, skipping Statcast stats.")
            continue
        if p['role'] == 'hitter':
            dh = " (DH)" if p['position'] == 'DH' else ""
            print(f"\n👤 {p['batting_order']}. {p['name']}{dh} | Pos: {p['position']} | Bats: {bat} | ID: {p['id']}")
            aggregate_and_print_all_stats(statcast_df, 'batter', mlbam_id)
        else:
            print(f"\n👤 {p['name']} | Pos: {p['position']} | Throws: {thr} | ID: {p['id']}")
            aggregate_and_print_all_stats(statcast_df, 'pitcher', mlbam_id)

def main():
    selected = input("Enter team name or 'all' for all teams: ").strip().lower()
    schedule = get_today_schedule()
    if not schedule:
        print("⚠️ No MLB games today.")
        return

    season = datetime.now().year
    print("📊 Loading full Statcast data for this season... (can take ~30-60s)")
    statcast_df = statcast(f'{season}-03-01', datetime.now().strftime('%Y-%m-%d'))

    for g in schedule:
        box = get_boxscore(g['game_id'])
        home_team = box['teams']['home']['team']['name']
        away_team = box['teams']['away']['team']['name']
        home_players = box['teams']['home']['players']
        away_players = box['teams']['away']['players']

        process_lineup(away_team, away_players, selected, statcast_df)
        process_lineup(home_team, home_players, selected, statcast_df)

        print("\n" + "="*80)

if __name__ == '__main__':
    main()
