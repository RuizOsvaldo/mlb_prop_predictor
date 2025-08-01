"""
Realistic MLB Data Collector - FIXED VERSION

Fixes:
1. Uses realistic player names instead of "NYY Player 13"
2. Creates realistic hitting statistics that produce reasonable probabilities
3. Better distribution of player performance levels
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')

class RealisticMLBDataCollector:
    def __init__(self):
        """Initialize realistic data collector with actual-style names and stats"""
        self.current_year = datetime.now().year
        
        # Realistic player first and last names
        self.first_names = [
            'Aaron', 'Alex', 'Anthony', 'Brandon', 'Carlos', 'Chris', 'Daniel', 'David', 'Eddie', 'Francisco',
            'George', 'Hunter', 'Isaiah', 'Jacob', 'Jose', 'Juan', 'Kevin', 'Luis', 'Marcus', 'Michael',
            'Nick', 'Oscar', 'Pablo', 'Rafael', 'Roberto', 'Salvador', 'Tommy', 'Victor', 'William', 'Xavier',
            'Yusuke', 'Zach', 'Andrew', 'Brian', 'Christian', 'Derek', 'Emanuel', 'Felix', 'Gabriel', 'Hector',
            'Ivan', 'Javier', 'Kyle', 'Lorenzo', 'Manuel', 'Nathan', 'Oliver', 'Patrick', 'Quinton', 'Ricardo'
        ]
        
        self.last_names = [
            'Rodriguez', 'Martinez', 'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
            'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee',
            'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker',
            'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores', 'Green',
            'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell', 'Carter', 'Roberts', 'Gomez'
        ]
        
        # Team mappings
        self.teams = ['NYY', 'BOS', 'LAD', 'SF', 'HOU', 'SEA', 'ATL', 'PHI', 'STL', 'CHC']
        
        # Ballpark factors
        self.ballpark_factors = {
            'Fenway Park': {'hr_factor': 1.03, 'hit_factor': 1.02},
            'Yankee Stadium': {'hr_factor': 1.10, 'hit_factor': 1.01},
            'Oracle Park': {'hr_factor': 0.92, 'hit_factor': 0.98},
            'T-Mobile Park': {'hr_factor': 0.95, 'hit_factor': 1.00},
            'Truist Park': {'hr_factor': 1.05, 'hit_factor': 1.01}
        }
        
        print("Realistic MLB Data Collector initialized - real names and realistic stats!")
    
    def generate_realistic_player_name(self, used_names):
        """Generate a realistic player name that hasn't been used"""
        max_attempts = 100
        for _ in range(max_attempts):
            first = np.random.choice(self.first_names)
            last = np.random.choice(self.last_names)
            full_name = f"{first} {last}"
            if full_name not in used_names:
                used_names.add(full_name)
                return full_name
        
        # Fallback if all combinations used
        return f"{np.random.choice(self.first_names)} {np.random.choice(self.last_names)} Jr."
    
    def collect_all_data_enhanced(self):
        """
        Master function that returns realistic data with proper names and stats
        """
        print("=" * 60)
        print("REALISTIC DATA COLLECTION - REAL NAMES & REALISTIC STATS")
        print("=" * 60)
        
        # Create comprehensive hitting data with realistic names and stats
        hitting_data = self.create_realistic_hitting_data()
        
        # Create pitcher data
        pitcher_data = self.create_realistic_pitcher_data()
        
        # Create today's games with lineups
        daily_games = self.create_todays_games_with_lineups()
        
        # Enhance hitting data with game-specific information
        enhanced_hitting_data = self.add_game_context_to_players(hitting_data, daily_games)
        
        print(f"✅ REALISTIC COLLECTION COMPLETE!")
        print(f"   Games: {len(daily_games)}")
        print(f"   Players: {len(enhanced_hitting_data)}")
        print(f"   Pitchers: {len(pitcher_data)}")
        print(f"   Features: {len(enhanced_hitting_data.columns)}")
        
        return enhanced_hitting_data, pitcher_data, daily_games
    
    def create_realistic_hitting_data(self):
        """Create realistic hitting statistics that produce reasonable probabilities"""
        print("Creating realistic hitting data with real names...")
        
        players = []
        used_names = set()
        
        # Create different tiers of players with realistic stat distributions
        player_tiers = [
            # Elite hitters (5 players) - these should have high probabilities
            {'tier': 'elite', 'avg_range': (0.290, 0.330), 'iso_range': (0.220, 0.300), 'obp_range': (0.350, 0.420), 'count': 5},
            # Above average hitters (15 players) - decent probabilities
            {'tier': 'above_avg', 'avg_range': (0.260, 0.290), 'iso_range': (0.160, 0.220), 'obp_range': (0.320, 0.360), 'count': 15},
            # Average hitters (20 players) - around league average probabilities
            {'tier': 'average', 'avg_range': (0.240, 0.270), 'iso_range': (0.140, 0.180), 'obp_range': (0.300, 0.340), 'count': 20},
            # Below average hitters (15 players) - lower probabilities
            {'tier': 'below_avg', 'avg_range': (0.210, 0.250), 'iso_range': (0.110, 0.160), 'obp_range': (0.280, 0.320), 'count': 15},
            # Struggling hitters (5 players) - low probabilities
            {'tier': 'struggling', 'avg_range': (0.180, 0.220), 'iso_range': (0.080, 0.130), 'obp_range': (0.250, 0.290), 'count': 5}
        ]
        
        for tier_info in player_tiers:
            for i in range(tier_info['count']):
                # Generate realistic name
                player_name = self.generate_realistic_player_name(used_names)
                
                # Generate realistic base stats for this tier
                avg = np.random.uniform(*tier_info['avg_range'])
                iso = np.random.uniform(*tier_info['iso_range'])
                obp = np.random.uniform(*tier_info['obp_range'])
                
                # Calculate derived stats
                slg = avg + iso
                ops = obp + slg
                
                # Generate counting stats based on performance level
                pa = np.random.randint(350, 650)
                ab = int(pa * np.random.uniform(0.82, 0.92))  # Account for walks, HBP
                h = int(ab * avg)
                bb = int((obp * pa - h - 5) if (obp * pa - h - 5) > 0 else pa * 0.05)  # 5 HBP assumed
                
                # HR based on ISO and tier
                if tier_info['tier'] == 'elite':
                    hr = np.random.randint(25, 50)
                elif tier_info['tier'] == 'above_avg':
                    hr = np.random.randint(15, 30)
                elif tier_info['tier'] == 'average':
                    hr = np.random.randint(8, 20)
                elif tier_info['tier'] == 'below_avg':
                    hr = np.random.randint(3, 12)
                else:  # struggling
                    hr = np.random.randint(0, 8)
                
                # Strikeouts based on tier
                if tier_info['tier'] == 'elite':
                    so = int(pa * np.random.uniform(0.15, 0.22))  # Elite hitters strike out less
                elif tier_info['tier'] == 'above_avg':
                    so = int(pa * np.random.uniform(0.18, 0.25))
                elif tier_info['tier'] == 'average':
                    so = int(pa * np.random.uniform(0.20, 0.28))
                elif tier_info['tier'] == 'below_avg':
                    so = int(pa * np.random.uniform(0.22, 0.30))
                else:  # struggling
                    so = int(pa * np.random.uniform(0.25, 0.35))
                
                # Calculate BABIP (more realistic)
                singles = h - (int(h * 0.20) + int(h * 0.02) + hr)  # Subtract 2B, 3B, HR
                balls_in_play = ab - so - hr
                babip = singles / balls_in_play if balls_in_play > 0 else 0.300
                babip = max(0.200, min(0.400, babip))  # Realistic BABIP range
                
                # Calculate wOBA (more realistic)
                doubles = int(h * 0.18)
                triples = int(h * 0.015)
                hbp = 5
                sf = 3
                
                woba_numerator = (0.69 * bb + 0.72 * hbp + 0.89 * singles + 
                                1.27 * doubles + 1.62 * triples + 2.10 * hr)
                woba_denominator = ab + bb + sf + hbp
                woba = woba_numerator / woba_denominator if woba_denominator > 0 else 0.315
                
                # Team assignment
                team = np.random.choice(self.teams)
                
                # Advanced metrics based on performance tier
                if tier_info['tier'] == 'elite':
                    contact_pct = np.random.uniform(80, 90)
                    whiff_pct = np.random.uniform(10, 18)
                    barrel_pct = np.random.uniform(12, 20)
                    hard_hit_pct = np.random.uniform(45, 60)
                elif tier_info['tier'] == 'above_avg':
                    contact_pct = np.random.uniform(75, 82)
                    whiff_pct = np.random.uniform(18, 25)
                    barrel_pct = np.random.uniform(8, 14)
                    hard_hit_pct = np.random.uniform(38, 48)
                elif tier_info['tier'] == 'average':
                    contact_pct = np.random.uniform(70, 78)
                    whiff_pct = np.random.uniform(22, 28)
                    barrel_pct = np.random.uniform(6, 10)
                    hard_hit_pct = np.random.uniform(32, 42)
                elif tier_info['tier'] == 'below_avg':
                    contact_pct = np.random.uniform(65, 73)
                    whiff_pct = np.random.uniform(25, 32)
                    barrel_pct = np.random.uniform(4, 8)
                    hard_hit_pct = np.random.uniform(28, 38)
                else:  # struggling
                    contact_pct = np.random.uniform(60, 68)
                    whiff_pct = np.random.uniform(30, 40)
                    barrel_pct = np.random.uniform(2, 6)
                    hard_hit_pct = np.random.uniform(25, 35)
                
                player = {
                    'Name': player_name,
                    'Team': team,
                    'Tm': team,
                    'Tier': tier_info['tier'],  # For debugging
                    'BA': round(avg, 3),
                    'AVG': round(avg, 3),
                    'OBP': round(obp, 3),
                    'SLG': round(slg, 3),
                    'OPS': round(ops, 3),
                    'ISO': round(iso, 3),
                    'wOBA': round(woba, 3),
                    'BABIP': round(babip, 3),
                    'PA': pa,
                    'AB': ab,
                    'H': h,
                    'HR': hr,
                    'BB': bb,
                    'SO': so,
                    '2B': doubles,
                    '3B': triples,
                    'RBI': hr + np.random.randint(5, 25),
                    'SB': np.random.randint(0, 15),
                    'HBP': hbp,
                    'SF': sf,
                    'GDP': np.random.randint(3, 12),
                    # Advanced metrics
                    'Contact%': round(contact_pct, 1),
                    'Whiff%': round(whiff_pct, 1),
                    'Zone%': round(47 + np.random.uniform(-4, 4), 1),
                    'Chase%': round(29 + np.random.uniform(-6, 6), 1),
                    'Barrel%': round(barrel_pct, 1),
                    'HardHit%': round(hard_hit_pct, 1),
                    'avg_exit_velocity': round(88.5 + np.random.uniform(-3, 3), 1),
                    'avg_launch_angle': round(12 + np.random.uniform(-4, 6), 1),
                }
                
                # Calculate rates
                player['HR_Rate'] = hr / pa
                player['BB_Rate'] = bb / pa
                player['K_Rate'] = so / pa
                
                players.append(player)
        
        hitting_df = pd.DataFrame(players)
        
        # Show distribution
        tier_counts = hitting_df['Tier'].value_counts()
        print(f"✅ Created realistic hitting data for {len(hitting_df)} players:")
        for tier, count in tier_counts.items():
            avg_avg = hitting_df[hitting_df['Tier'] == tier]['AVG'].mean()
            print(f"   {tier}: {count} players (avg: {avg_avg:.3f})")
        
        return hitting_df
    
    def create_realistic_pitcher_data(self):
        """Create realistic pitcher data with real names"""
        print("Creating realistic pitcher data...")
        
        pitchers = []
        used_names = set()
        
        for i in range(30):
            pitcher_name = self.generate_realistic_player_name(used_names)
            
            # Generate realistic pitcher stats
            era = np.random.uniform(2.80, 5.20)
            whip = np.random.uniform(1.10, 1.50)
            k9 = np.random.uniform(7.0, 11.5)
            bb9 = np.random.uniform(2.2, 4.8)
            hr9 = np.random.uniform(0.9, 1.8)
            
            pitcher = {
                'Name': pitcher_name,
                'Team': np.random.choice(self.teams),
                'ERA': round(era, 2),
                'WHIP': round(whip, 3),
                'K9': round(k9, 1),
                'BB9': round(bb9, 1),
                'HR9': round(hr9, 1),
                'IP': round(np.random.uniform(120, 200), 1),
                'throws': np.random.choice(['R', 'L'], p=[0.75, 0.25])
            }
            pitchers.append(pitcher)
        
        pitcher_df = pd.DataFrame(pitchers)
        print(f"✅ Created realistic pitcher data for {len(pitcher_df)} pitchers")
        return pitcher_df
    
    def create_todays_games_with_lineups(self):
        """Create today's games with realistic lineups"""
        print("Creating today's games with realistic lineups...")
        
        today = date.today()
        
        games = [
            {
                'game_date': today,
                'away_team': 'NYY',
                'home_team': 'BOS', 
                'ballpark': 'Fenway Park',
                'game_time': '19:10',
                'game_id': f'realistic_{today}_1'
            },
            {
                'game_date': today,
                'away_team': 'LAD',
                'home_team': 'SF',
                'ballpark': 'Oracle Park', 
                'game_time': '21:45',
                'game_id': f'realistic_{today}_2'
            },
            {
                'game_date': today,
                'away_team': 'HOU',
                'home_team': 'SEA',
                'ballpark': 'T-Mobile Park',
                'game_time': '22:10', 
                'game_id': f'realistic_{today}_3'
            }
        ]
        
        # Add lineups and additional game info
        enhanced_games = []
        used_pitcher_names = set()
        
        for game in games:
            # Create lineups
            away_lineup = self.create_realistic_lineup(game['away_team'])
            home_lineup = self.create_realistic_lineup(game['home_team'])
            
            # Add weather
            weather = self.create_weather(game['ballpark'])
            
            # Add pitchers with realistic names
            away_pitcher_name = self.generate_realistic_player_name(used_pitcher_names)
            home_pitcher_name = self.generate_realistic_player_name(used_pitcher_names)
            
            away_pitcher = {'name': away_pitcher_name, 'throws': np.random.choice(['R', 'L'])}
            home_pitcher = {'name': home_pitcher_name, 'throws': np.random.choice(['R', 'L'])}
            
            enhanced_game = game.copy()
            enhanced_game.update({
                'away_lineup': away_lineup,
                'home_lineup': home_lineup,
                'weather': weather,
                'away_pitcher': away_pitcher,
                'home_pitcher': home_pitcher
            })
            
            enhanced_games.append(enhanced_game)
        
        games_df = pd.DataFrame(enhanced_games)
        print(f"✅ Created {len(games_df)} games with realistic lineups and pitchers")
        return games_df
    
    def create_realistic_lineup(self, team):
        """Create a 9-player lineup for a team with realistic player names"""
        positions = ['CF', '2B', '1B', 'LF', '3B', 'RF', 'C', 'SS', 'P']
        used_names = set()
        
        lineup = []
        for i in range(9):
            player_name = self.generate_realistic_player_name(used_names)
            lineup.append({
                'batting_order': i + 1,
                'player_name': player_name,
                'position': positions[i],
                'stats': {}
            })
        
        return lineup
    
    def create_weather(self, ballpark):
        """Create realistic weather for a ballpark"""
        month = datetime.now().month
        
        if 'Boston' in ballpark or 'Fenway' in ballpark:
            temp_base = 55 if month in [4, 10] else 72 if month in [5, 9] else 78
        elif 'San Francisco' in ballpark or 'Oracle' in ballpark:
            temp_base = 62  # Cool SF weather
        elif 'Seattle' in ballpark or 'T-Mobile' in ballpark:
            temp_base = 65  # Mild Seattle weather
        else:
            temp_base = 75
        
        weather = {
            'temperature': temp_base + np.random.randint(-6, 8),
            'wind_speed': np.random.randint(4, 14),
            'wind_direction': np.random.choice(['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']),
            'humidity': np.random.randint(45, 75),
            'conditions': np.random.choice(['Clear', 'Partly Cloudy', 'Overcast'], p=[0.6, 0.3, 0.1])
        }
        
        return weather
    
    def add_game_context_to_players(self, hitting_data, daily_games):
        """Add game-specific context to player data"""
        print("Adding game context to players...")
        
        enhanced_players = []
        
        for _, game in daily_games.iterrows():
            # Process away team lineup
            away_team = game['away_team']
            away_players = hitting_data[hitting_data['Team'] == away_team].copy()
            
            # Take best 9 players for lineup (sorted by OPS)
            if len(away_players) >= 9:
                away_lineup_players = away_players.nlargest(9, 'OPS')
            else:
                away_lineup_players = away_players
                while len(away_lineup_players) < 9:
                    # Add average players if not enough team players
                    avg_players = hitting_data[hitting_data['Tier'] == 'average'].sample(1)
                    avg_player = avg_players.iloc[0].copy()
                    avg_player['Team'] = away_team
                    avg_player['Name'] = f"{away_team} {avg_player['Name']}"
                    away_lineup_players = pd.concat([away_lineup_players, pd.DataFrame([avg_player])], ignore_index=True)
            
            # Add game context to away players
            for i, (_, player) in enumerate(away_lineup_players.iterrows()):
                enhanced_player = player.copy()
                enhanced_player = self.add_enhanced_features(enhanced_player, game, 'away', i + 1)
                enhanced_players.append(enhanced_player)
            
            # Process home team lineup
            home_team = game['home_team']
            home_players = hitting_data[hitting_data['Team'] == home_team].copy()
            
            if len(home_players) >= 9:
                home_lineup_players = home_players.nlargest(9, 'OPS')
            else:
                home_lineup_players = home_players
                while len(home_lineup_players) < 9:
                    avg_players = hitting_data[hitting_data['Tier'] == 'average'].sample(1)
                    avg_player = avg_players.iloc[0].copy()
                    avg_player['Team'] = home_team
                    avg_player['Name'] = f"{home_team} {avg_player['Name']}"
                    home_lineup_players = pd.concat([home_lineup_players, pd.DataFrame([avg_player])], ignore_index=True)
            
            # Add game context to home players  
            for i, (_, player) in enumerate(home_lineup_players.iterrows()):
                enhanced_player = player.copy()
                enhanced_player = self.add_enhanced_features(enhanced_player, game, 'home', i + 1)
                enhanced_players.append(enhanced_player)
        
        enhanced_df = pd.DataFrame(enhanced_players)
        print(f"✅ Enhanced {len(enhanced_df)} players with game context")
        
        # Show sample names
        print("Sample player names:")
        for name in enhanced_df['Name'].head(10):
            print(f"   {name}")
        
        return enhanced_df
    
    def add_enhanced_features(self, player, game, home_away, batting_order):
        """Add enhanced game-specific features to a player"""
        # Add game context
        player['batting_order'] = batting_order
        player['position'] = self.get_position_by_order(batting_order)
        player['home_away'] = home_away
        player['ballpark'] = game['ballpark']
        player['game_id'] = game['game_id']
        
        # Add ballpark factors
        ballpark_info = self.ballpark_factors.get(game['ballpark'], {'hr_factor': 1.0, 'hit_factor': 1.0})
        player['ballpark_hr_factor'] = ballpark_info['hr_factor']
        player['ballpark_hit_factor'] = ballpark_info['hit_factor']
        
        # Add opposing pitcher info
        opp_pitcher = game.get(f'{"home" if home_away == "away" else "away"}_pitcher', {})
        player['opp_pitcher_name'] = opp_pitcher.get('name', 'TBD')
        player['opp_pitcher_throws'] = opp_pitcher.get('throws', 'R')
        
        # Add weather data
        weather = game.get('weather', {})
        player['game_temperature'] = weather.get('temperature', 72)
        player['game_wind_speed'] = weather.get('wind_speed', 8)
        player['game_wind_direction'] = weather.get('wind_direction', 'W')
        player['game_humidity'] = weather.get('humidity', 60)
        
        # Add weather adjustment factors (conservative)
        temp = weather.get('temperature', 72)
        wind_speed = weather.get('wind_speed', 8)
        
        # Temperature adjustment (much smaller effect)
        player['weather_temp_adj'] = 1.0 + ((temp - 72) * 0.001)  # Reduced from 0.002
        
        # Wind adjustment (smaller effect)
        wind_direction = weather.get('wind_direction', 'W')
        if wind_direction in ['S', 'SW', 'SE'] and wind_speed > 12:
            player['weather_wind_adj'] = 1.0 + (wind_speed * 0.002)  # Reduced from 0.003
        elif wind_direction in ['N', 'NW', 'NE'] and wind_speed > 12:
            player['weather_wind_adj'] = 1.0 - (wind_speed * 0.001)  # Reduced from 0.002
        else:
            player['weather_wind_adj'] = 1.0
        
        # Humidity adjustment (minimal effect)
        humidity = weather.get('humidity', 60)
        player['weather_humidity_adj'] = 1.0 - ((humidity - 60) * 0.0005)  # Reduced from 0.001
        
        return player
    
    def get_position_by_order(self, batting_order):
        """Get typical position by batting order"""
        position_map = {
            1: 'CF', 2: '2B', 3: '1B', 4: 'LF', 5: '3B',
            6: 'RF', 7: 'C', 8: 'SS', 9: 'P'
        }
        return position_map.get(batting_order, 'OF')

# Enhanced classes for compatibility (same as before)
class PitcherAnalyzer:
    def __init__(self):
        pass
    
    def analyze_pitcher_matchup(self, batter_stats, pitcher_stats):
        return {
            'advantage': 'neutral',
            'confidence': 0.6,
            'key_factors': ['Realistic matchup analysis']
        }

class WeatherImpactCalculator:
    def __init__(self):
        pass
    
    def calculate_weather_impact(self, weather_data, ballpark):
        temp = weather_data.get('temperature', 72)
        wind_speed = weather_data.get('wind_speed', 8)
        
        # Much smaller weather effects for realistic probabilities
        hr_factor = 1.0 + ((temp - 72) * 0.002)  # Reduced impact
        hit_factor = 1.0 + ((temp - 72) * 0.001)  # Even smaller for hits
        
        impact_summary = []
        if temp > 85:
            impact_summary.append(f"Hot weather ({temp}°F) slightly helps hitting")
        elif temp < 55:
            impact_summary.append(f"Cold weather ({temp}°F) slightly reduces hitting")
        
        if wind_speed > 15:
            impact_summary.append(f"Strong wind ({wind_speed} mph) affects ball flight")
        
        return {
            'hr_factor': hr_factor,
            'hit_factor': hit_factor,
            'total_impact': (hr_factor + hit_factor) / 2,
            'impact_summary': impact_summary if impact_summary else ["Neutral weather conditions"]
        }

# Aliases for compatibility
EnhancedMLBDataCollector = RealisticMLBDataCollector
DebugMLBDataCollector = RealisticMLBDataCollector

if __name__ == "__main__":
    collector = RealisticMLBDataCollector()
    
    print("Testing Realistic MLB Data Collector...")
    
    # Test data collection
    hitting_data, pitching_data, daily_games = collector.collect_all_data_enhanced()
    
    print(f"\n✅ SUCCESS! Realistic collection working.")
    print(f"Players: {len(hitting_data)}")
    print(f"Games: {len(daily_games)}")
    
    # Show sample realistic data
    if len(hitting_data) > 0:
        print(f"\nSample realistic players:")
        for i, player in hitting_data.head(10).iterrows():
            print(f"   {player['Name']} ({player['Team']}) - AVG: {player['AVG']:.3f}, Tier: {player['Tier']}")
    
    print(f"\n✅ Realistic data collector ready!")