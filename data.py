from datetime import datetime, timedelta
import pandas as pd
import requests
from pybaseball import statcast_batter, statcast_pitcher
from tqdm import tqdm

def get_today_games():
    today = datetime.now().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    games = []

    for date_info in data['dates']:
        for game in date_info['games']:
            games.append({
                'game_pk': game['gamePk'],
                'home_team': game['teams']['home']['team']['name'],
                'away_team': game['teams']['away']['team']['name'],
                'home_pitcher_id': game['teams']['home'].get('probablePitcher', {}).get('id'),
                'away_pitcher_id': game['teams']['away'].get('probablePitcher', {}).get('id'),
            })
    return games

def get_starting_lineup(game_pk):
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    response = requests.get(url)
    response.raise_for_status()
    box = response.json()

    players = []
    for side in ['home', 'away']:
        team_info = box['teams'][side]
        pitcher_list = team_info.get('pitchers', [])
        pitcher_id = pitcher_list[0] if pitcher_list else None

        for player_id, pdata in team_info['players'].items():
            details = pdata.get('person', {})
            position = pdata.get('position', {}).get('code', '')
            batting_order = pdata.get('battingOrder', '')

            # Only include players with a valid batting order (i.e., starters)
            if batting_order and position != 'P':
                players.append({
                    'player_id': details['id'],
                    'team_side': side,
                    'pitcher_id': pitcher_id
                })

    return players



def get_player_info(player_id):
    if not player_id:
        return {'player_name': None, 'batting_hand': None, 'pitching_hand': None}
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
    r = requests.get(url)
    r.raise_for_status()
    person = r.json()['people'][0]
    return {
        'player_name': person.get('fullName'),
        'batting_hand': person.get('batSide', {}).get('code'),
        'pitching_hand': person.get('pitchHand', {}).get('code')
    }


# def get_statcast_stats_batter(player_id, days=60):
#     end = datetime.today()
#     start = end - timedelta(days=days)
#     try:
#         df = statcast_batter(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), player_id)
#         batted_balls = df[df['launch_speed'].notnull()]
#         hard_hit_pct = (batted_balls['launch_speed'] >= 95).mean() * 100
#         if df.empty:
#             print(f"⚠️ No Statcast batter data found for player_id {player_id}")
#             return {}
#         return {
#             'avg_exit_velocity': round(batted_balls['launch_speed'].mean(), 2),
#             'avg_launch_angle': round(df['launch_angle'].mean(), 2),
#             'hard_hit_pct': round(hard_hit_pct, 1),
#             #'xSLG': df['expected_slg'].mean() if 'expected_slg' in df.columns else None,
#             'barrel_pct': round((batted_balls['launch_speed'] >= 98).mean() * 100, 1)
#         }
#     except Exception as e:
#         print(f"❌ Error getting batter statcast for {player_id}: {e}")
#         return {}
    
def get_statcast_stats_batter(player_id, days=60):
    end = datetime.today()
    start = end - timedelta(days=days)
    try:
        df = statcast_batter(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), player_id)
        if df.empty:
            return {}

        # ✅ Filter only valid batted ball events
        df = df[df['launch_speed'].notnull()]
        df = df[df['launch_angle'].notnull()]
        df = df[df['events'].notnull()]
        df = df[df['launch_speed'].between(60, 120)]  # clip outliers

        if len(df) < 15:  # not enough data to be meaningful
            return {}

        return {
            'avg_exit_velocity': round(df['launch_speed'].mean(), 2),
            'avg_launch_angle': round(df['launch_angle'].mean(), 2),
            'hard_hit_pct': round((df['launch_speed'] >= 95).mean() * 100, 1),
            'barrel_pct': round((df['launch_speed'] >= 98).mean() * 100, 1),  # crude proxy
            #'xSLG': round(df['estimated_slg'].dropna().mean(), 3) if 'estimated_slg' in df.columns else None
        }
    except Exception as e:
        print(f"Error fetching batter {player_id}: {e}")
        return {}



# def get_statcast_stats_pitcher(player_id, days=60):
#     end = datetime.today()
#     start = end - timedelta(days=days)
#     try:
#         df = statcast_pitcher(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), player_id)
#         if df.empty:
#             return {}
#         return {
#             'pitcher_ev': df['launch_speed'].mean(),
#             'pitcher_la': df['launch_angle'].mean(),
#             'pitcher_hard_hit_pct': (df['launch_speed'] >= 95).mean(),
#             'pitcher_barrel_pct': (df['launch_speed'] >= 98).mean()
#         }
#     except Exception as e:
#         return {}
    
def get_statcast_stats_pitcher(player_id, days=60):
    from datetime import datetime, timedelta
    from pybaseball import statcast_pitcher

    end = datetime.today()
    start = end - timedelta(days=days)
    try:
        df = statcast_pitcher(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), player_id)
        if df.empty:
            return {}

        # ✅ Filter only balls put in play
        df = df[df['launch_speed'].notnull()]
        df = df[df['launch_angle'].notnull()]
        df = df[df['events'].notnull()]
        df = df[df['launch_speed'].between(60, 120)]  # trim outliers

        if len(df) < 15:
            return {}

        return {
            'pitcher_ev': round(df['launch_speed'].mean(), 2),
            'pitcher_la': round(df['launch_angle'].mean(), 2),
            'pitcher_hard_hit_pct': round((df['launch_speed'] >= 95).mean() * 100, 1),
            'pitcher_barrel_pct': round((df['launch_speed'] >= 98).mean() * 100, 1)
        }
    except Exception as e:
        print(f"Error fetching pitcher {player_id}: {e}")
        return {}


def load_today_lineup_data():
    games = get_today_games()
    all_rows = []
    pitcher_cache = {}

    for game in games:
        lineup = get_starting_lineup(game['game_pk'])
        if not lineup:
            print(f"⚠️ No confirmed lineup for {game['home_team']} vs {game['away_team']}")
            continue
        else:
            print(f"✅ Found {len(lineup)} starting players for {game['home_team']} vs {game['away_team']}")

        for player in tqdm(lineup, desc=f"🔄 Processing {game['home_team']} vs {game['away_team']}", leave=False):
            try:
                batter_id = player['player_id']
                pitcher_id = game['away_pitcher_id'] if player['team_side'] == 'home' else game['home_pitcher_id']


                batter_info = get_player_info(batter_id)
                pitcher_info = get_player_info(pitcher_id)

                batter_stats = get_statcast_stats_batter(batter_id)

                if pitcher_id in pitcher_cache:
                    pitcher_stats = pitcher_cache[pitcher_id]
                else:
                    pitcher_stats = get_statcast_stats_pitcher(pitcher_id)
                    pitcher_cache[pitcher_id] = pitcher_stats

                team = game['home_team'] if player['team_side'] == 'home' else game['away_team']

                row = {
                    'player_id': batter_id,
                    'player_name': batter_info['player_name'],
                    'team': team,
                    'batting_hand': batter_info['batting_hand'],
                    'opp_pitcher': pitcher_info['player_name'],
                    'pitching_hand': pitcher_info['pitching_hand'],
                    **batter_stats,
                    **pitcher_stats,
                    'park_factor': 1.0,         # Placeholder
                    'weather_factor': 1.0       # Placeholder
                }
                all_rows.append(row)
            except Exception as e:
                print(f"⚠️ Error processing player {player['player_id']}: {e}")

    df = pd.DataFrame(all_rows)
    print(f"✅ Loaded {len(df)} players in confirmed lineups.")
    return pd.DataFrame(all_rows)

    