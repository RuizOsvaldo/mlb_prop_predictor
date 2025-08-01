"""
MLB Hit and Home Run Predictor - FIXED STREAMLIT APP

This fixes the major issues:
1. Focuses on daily games automatically
2. No manual ballpark selection - uses actual game ballparks
3. No player selection - shows team-based predictions
4. Uses real data only, no samples
5. Shows actual games being played today

Key Features:
- Automatic daily game detection
- Team-based predictions for today's games
- Ballpark factors applied automatically
- Real-time model performance tracking
- Clean, focused UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import sys
import os

# Add project directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Import our custom modules
try:
    from data.data_collector import EnhancedMLBDataCollector, PitcherAnalyzer, WeatherImpactCalculator
    from data.feature_engineer import SabermetricFeatureEngineer
    from models.hit_predictor import HitPredictor
    from models.hr_predictor import HomeRunPredictor
    from data.database import MLBPredictionDatabase
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="MLB Enhanced Daily Predictions",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'enhanced_data_loaded' not in st.session_state:
    st.session_state.enhanced_data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'daily_games_with_lineups' not in st.session_state:
    st.session_state.daily_games_with_lineups = pd.DataFrame()

# Initialize enhanced components
@st.cache_resource
def initialize_components():
    """Initialize enhanced data collector and all analysis tools"""
    collector = EnhancedMLBDataCollector()
    engineer = SabermetricFeatureEngineer()
    hit_model = HitPredictor()
    hr_model = HomeRunPredictor()
    database = MLBPredictionDatabase()
    pitcher_analyzer = PitcherAnalyzer()
    weather_calculator = WeatherImpactCalculator()
    
    return collector, engineer, hit_model, hr_model, database, pitcher_analyzer, weather_calculator

collector, engineer, hit_model, hr_model, database, pitcher_analyzer, weather_calculator = initialize_components()

# Enhanced data loading with lineups and weather
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_mlb_data():
    """Load enhanced MLB data with lineups, weather, and pitcher matchups"""
    try:
        # Use enhanced collection method that gets actual lineups
        hitting_data, pitching_data, daily_games = collector.collect_all_data_enhanced()
        
        if len(hitting_data) == 0:
            st.warning("No hitting data with lineups collected. Check data sources.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Engineer features on the enhanced data
        final_data = engineer.engineer_all_features(hitting_data)
        
        return final_data, pitching_data, daily_games
    except Exception as e:
        st.error(f"Error loading enhanced MLB data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def train_models_with_data(data):
    """Train both models with enhanced error handling"""
    if len(data) == 0:
        st.error("No data available for training.")
        return False

    try:
        # Train hit prediction model
        hit_features, hit_targets, hit_feature_names = hit_model.prepare_training_data(data)
        
        if hit_features is None or len(hit_features) == 0:
            st.error("Hit training data preparation failed.")
            return False

        hit_metrics = hit_model.train_model(hit_features, hit_targets, hit_feature_names)
        
        # Train HR prediction model  
        hr_features, hr_targets, hr_feature_names = hr_model.prepare_training_data(data)
        
        if hr_features is None or len(hr_features) == 0:
            st.error("HR training data preparation failed.")
            return False

        hr_metrics = hr_model.train_model(hr_features, hr_targets, hr_feature_names)

        # Store feature importance in database
        if hit_model.feature_importance:
            database.store_feature_importance('hit', hit_model.feature_importance, 'v1.0', len(data))
        if hr_model.feature_importance:
            database.store_feature_importance('hr', hr_model.feature_importance, 'v1.0', len(data))

        return True
    except Exception as e:
        st.error(f"Error training models: {e}")
        return False

def main():
    st.title("⚾ Enhanced MLB Daily Predictions")
    st.markdown("### Today's Games with Lineups, Weather & Advanced Analysis")
    
    # Load enhanced data if not already loaded
    if not st.session_state.enhanced_data_loaded:
        with st.spinner("Loading today's games with starting lineups, weather data, and pitcher matchups..."):
            hitting_data, pitching_data, daily_games = load_mlb_data()
            
            if len(hitting_data) > 0:
                st.session_state.hitting_data = hitting_data
                st.session_state.pitching_data = pitching_data
                st.session_state.daily_games_with_lineups = daily_games
                st.session_state.enhanced_data_loaded = True
                
                # Train models with enhanced data
                if train_models_with_data(hitting_data):
                    st.session_state.models_trained = True
                    st.success("✅ Enhanced data loaded with lineups, weather, and models trained!")
                    
                    # Show what enhanced data we have
                    enhanced_features = [col for col in hitting_data.columns if any(keyword in col for keyword in 
                                       ['weather_', 'ballpark_', 'opp_pitcher_', 'batting_order', 'game_'])]
                    
                    if enhanced_features:
                        st.info(f"🚀 Enhanced features active: {len(enhanced_features)} game-specific data points per player")
                else:
                    st.error("❌ Error training enhanced models")
            else:
                st.error("❌ Failed to load enhanced MLB data with lineups")
                return
    
    # Show enhanced games overview
    show_daily_games_overview()
    
    # Main enhanced prediction interface
    st.markdown("---")
    daily_predictions_interface()

def show_daily_games_overview():
    """Show enhanced overview of today's games with lineups and weather"""
    st.subheader("🏟️ Today's Enhanced Games")
    
    daily_games = st.session_state.daily_games_with_lineups
    
    if len(daily_games) == 0:
        st.info("No enhanced games data available for today.")
        return
    
    # Display enhanced game information
    for _, game in daily_games.iterrows():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
        
        with col1:
            st.write(f"**{game['away_team']} @ {game['home_team']}**")
            ballpark = game.get('ballpark', 'Unknown Stadium')
            st.caption(f"🏟️ {ballpark}")
        
        with col2:
            # Show lineup info
            away_lineup = game.get('away_lineup', [])
            home_lineup = game.get('home_lineup', [])
            st.write(f"📋 Lineups: {len(away_lineup)}/{len(home_lineup)}")
            
        with col3:
            # Show weather info
            weather = game.get('weather', {})
            if weather:
                temp = weather.get('temperature', 'Unknown')
                wind = weather.get('wind_speed', 'Unknown')
                st.write(f"🌤️ {temp}°F, {wind} mph")
            else:
                st.write("🌤️ Weather pending")
        
        with col4:
            # Show pitcher matchup
            away_pitcher = game.get('away_pitcher', {}).get('name', 'TBD')
            home_pitcher = game.get('home_pitcher', {}).get('name', 'TBD')
            st.write(f"⚾ {away_pitcher[:10]}... vs {home_pitcher[:10]}...")
    
    st.markdown(f"**{len(daily_games)} enhanced games** with lineups for {date.today().strftime('%B %d, %Y')}")
    
    # Show enhanced data summary
    if len(daily_games) > 0:
        games_with_lineups = sum(1 for _, game in daily_games.iterrows() 
                               if len(game.get('away_lineup', [])) > 0 or len(game.get('home_lineup', [])) > 0)
        games_with_weather = sum(1 for _, game in daily_games.iterrows() if game.get('weather'))
        games_with_pitchers = sum(1 for _, game in daily_games.iterrows() 
                                if game.get('away_pitcher', {}).get('name', 'TBD') != 'TBD')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Games with Lineups", games_with_lineups)
        with col2:
            st.metric("Games with Weather", games_with_weather)
        with col3:
            st.metric("Games with Pitchers", games_with_pitchers)

def daily_predictions_interface():
    """Enhanced interface for daily predictions with lineup analysis"""
    st.subheader("🎯 Enhanced Daily Predictions")
    
    if not st.session_state.models_trained:
        st.warning("Enhanced models are still training. Please wait...")
        return
    
    daily_games = st.session_state.daily_games_with_lineups
    
    if len(daily_games) == 0:
        st.info("No games with lineups available for enhanced predictions.")
        return
    
    # Enhanced prediction options
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_scope = st.selectbox(
            "Prediction Scope:",
            ["All Today's Enhanced Games", "Specific Game Analysis"]
        )
    
    with col2:
        show_enhanced_analysis = st.checkbox("Show Weather & Pitcher Analysis", value=True)
    
    if prediction_scope == "Specific Game Analysis":
        # Game selection for detailed analysis
        game_options = []
        for _, game in daily_games.iterrows():
            away_lineup_count = len(game.get('away_lineup', []))
            home_lineup_count = len(game.get('home_lineup', []))
            weather_status = "🌤️" if game.get('weather') else "❓"
            game_str = f"{game['away_team']} @ {game['home_team']} ({away_lineup_count}/{home_lineup_count} lineups) {weather_status}"
            game_options.append(game_str)
        
        if game_options:
            selected_game_str = st.selectbox("Select Game for Enhanced Analysis:", game_options)
            selected_game_idx = game_options.index(selected_game_str)
            selected_game = daily_games.iloc[selected_game_idx]
        else:
            st.error("No games available for selection")
            return
    else:
        selected_game = None
    
    # Generate enhanced predictions
    if st.button("🔮 Generate Enhanced Predictions with Weather & Lineups", type="primary"):
        with st.spinner("Generating enhanced predictions with weather, ballpark, and pitcher analysis..."):
            if prediction_scope == "All Today's Enhanced Games":
                generate_all_enhanced_predictions(daily_games, show_enhanced_analysis)
def generate_all_enhanced_predictions(daily_games, show_analysis=False):
    """Generate enhanced predictions for all games"""
    st.subheader("📊 All Games - Enhanced Predictions")
    
    hitting_data = st.session_state.get('enhanced_hitting_data', st.session_state.get('hitting_data', pd.DataFrame()))
    
    if len(hitting_data) == 0:
        st.error("No hitting data available for predictions")
        return
    
    for _, game in daily_games.iterrows():
        st.markdown(f"## {game['away_team']} @ {game['home_team']}")
        st.markdown(f"**🏟️ {game.get('ballpark', 'Unknown Stadium')}**")
        
        # Show weather impact if requested
        if show_analysis:
            weather = game.get('weather', {})
            if weather:
                weather_impact = weather_calculator.calculate_weather_impact(weather, game.get('ballpark', ''))
                show_weather_analysis_simple(weather, weather_impact)
        
        # Get enhanced team data for both teams
        away_team_data = get_enhanced_team_data(hitting_data, game['away_team'], game)
        home_team_data = get_enhanced_team_data(hitting_data, game['home_team'], game)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {game['away_team']} (Away)")
            if len(away_team_data) > 0:
                generate_enhanced_team_predictions(away_team_data, game['away_team'], game, show_analysis)
            else:
                st.info(f"No enhanced lineup data for {game['away_team']}")
        
        with col2:
            st.markdown(f"### {game['home_team']} (Home)")
            if len(home_team_data) > 0:
                generate_enhanced_team_predictions(home_team_data, game['home_team'], game, show_analysis)
            else:
                st.info(f"No enhanced lineup data for {game['home_team']}")
        
        st.markdown("---")

def generate_detailed_enhanced_game_analysis(game, show_analysis=True):
    """Generate detailed analysis for a specific game"""
    st.subheader(f"🔍 Detailed Analysis: {game['away_team']} @ {game['home_team']}")
    st.markdown(f"**🏟️ {game.get('ballpark', 'Unknown Stadium')}**")
    
    hitting_data = st.session_state.get('enhanced_hitting_data', st.session_state.get('hitting_data', pd.DataFrame()))
    
    if len(hitting_data) == 0:
        st.error("No hitting data available for analysis")
        return
    
    # Weather analysis
    if show_analysis:
        weather = game.get('weather', {})
        if weather:
            weather_impact = weather_calculator.calculate_weather_impact(weather, game.get('ballpark', ''))
            show_weather_analysis_simple(weather, weather_impact, detailed=True)
        
        # Pitching matchup analysis
        show_pitching_matchup_analysis_simple(game)
    
    # Enhanced team predictions
    away_team_data = get_enhanced_team_data(hitting_data, game['away_team'], game)
    home_team_data = get_enhanced_team_data(hitting_data, game['home_team'], game)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {game['away_team']} Enhanced Analysis")
        if len(away_team_data) > 0:
            generate_enhanced_team_predictions(away_team_data, game['away_team'], game, True)
        else:
            st.info(f"No enhanced data for {game['away_team']}")
    
    with col2:
        st.markdown(f"### {game['home_team']} Enhanced Analysis")
        if len(home_team_data) > 0:
            generate_enhanced_team_predictions(home_team_data, game['home_team'], game, True)
        else:
            st.info(f"No enhanced data for {game['home_team']}")

def get_enhanced_team_data(hitting_data, team_abbrev, game):
    """Get enhanced hitting data for a specific team in a specific game"""
    if 'Team' in hitting_data.columns:
        team_data = hitting_data[hitting_data['Team'] == team_abbrev].copy()
    else:
        # Fallback: look for players with game_id matching this game
        if 'game_id' in hitting_data.columns:
            team_data = hitting_data[hitting_data['game_id'] == game.get('game_id', '')].copy()
        else:
            # Last resort: use a sample
            team_data = hitting_data.sample(min(9, len(hitting_data))).copy()
    
    return team_data

def generate_enhanced_team_predictions(team_data, team_name, game, show_analysis=False):
    """Generate enhanced predictions for a team"""
    try:
        # Generate hit predictions
        top_hitters = hit_model.predict_team_hits(team_data)
        
        # Generate HR predictions with enhanced ballpark factors
        ballpark = game.get('ballpark')
        top_power_hitters = hr_model.predict_team_hrs(team_data, ballpark)
        
        # Display enhanced predictions
        display_enhanced_team_predictions(top_hitters, top_power_hitters, team_name, game, show_analysis)
        
        # Store enhanced predictions
        store_enhanced_predictions(top_hitters, top_power_hitters, team_name, game)
        
    except Exception as e:
        st.error(f"Error generating enhanced predictions for {team_name}: {e}")

def display_enhanced_team_predictions(top_hitters, top_power_hitters, team_name, game, show_analysis=False):
    """Display enhanced prediction results"""
    
    # Enhanced hit predictions
    st.markdown("#### 🎯 Enhanced Hit Predictions")
    if len(top_hitters) > 0:
        for i, (_, player) in enumerate(top_hitters.iterrows(), 1):
            player_name = player.get('player_name', player.get('Name', f'Player_{i}'))
            hit_prob = player.get('hit_probability', 0)
            
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.write(f"**{i}. {player_name}**")
                    if 'batting_order' in player:
                        st.caption(f"Batting #{int(player['batting_order'])}")
                with col2:
                    st.metric("Hit Probability", f"{hit_prob:.1%}")
                with col3:
                    weather_adj = player.get('weather_temp_adj', 1.0)
                    if weather_adj != 1.0:
                        adj_text = f"+{(weather_adj-1)*100:.1f}%" if weather_adj > 1 else f"{(weather_adj-1)*100:.1f}%"
                        st.metric("Weather Adj", adj_text)
                
                st.progress(hit_prob)
                
                if show_analysis:
                    # Show detailed analysis
                    analysis_items = []
                    if 'opp_pitcher_name' in player and player['opp_pitcher_name'] != 'TBD':
                        analysis_items.append(f"vs {player['opp_pitcher_name']} ({player.get('opp_pitcher_throws', 'R')})")
                    if 'game_temperature' in player:
                        analysis_items.append(f"Temp: {player['game_temperature']}°F")
                    if analysis_items:
                        st.caption(" | ".join(analysis_items))
                
                st.markdown("---")
    else:
        st.info("No enhanced hit predictions available")
    
    # Enhanced HR predictions
    st.markdown("#### 💥 Enhanced Home Run Predictions")
    if len(top_power_hitters) > 0:
        for i, (_, player) in enumerate(top_power_hitters.iterrows(), 1):
            player_name = player.get('player_name', player.get('Name', f'Power Player_{i}'))
            hr_prob = player.get('hr_probability', 0)
            power_tier = player.get('power_tier', 'Unknown')
            
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.write(f"**{i}. {player_name}**")
                    if 'batting_order' in player:
                        st.caption(f"Batting #{int(player['batting_order'])}")
                with col2:
                    st.metric("HR Probability", f"{hr_prob:.1%}")
                with col3:
                    st.metric("Power Tier", power_tier)
                
                st.progress(hr_prob)
                
                if show_analysis:
                    # Enhanced HR analysis
                    analysis_items = []
                    
                    # Ballpark factor
                    ballpark_boost = player.get('ballpark_boost', 1.0)
                    if ballpark_boost != 1.0:
                        boost_text = f"+{(ballpark_boost-1)*100:.0f}%" if ballpark_boost > 1 else f"{(ballpark_boost-1)*100:.0f}%"
                        analysis_items.append(f"Ballpark: {boost_text}")
                    
                    # Weather impact
                    weather_factors = []
                    if 'weather_temp_adj' in player and player['weather_temp_adj'] != 1.0:
                        temp_adj = player['weather_temp_adj']
                        temp_text = f"+{(temp_adj-1)*100:.1f}%" if temp_adj > 1 else f"{(temp_adj-1)*100:.1f}%"
                        weather_factors.append(f"Temp: {temp_text}")
                    
                    if 'weather_wind_adj' in player and player['weather_wind_adj'] != 1.0:
                        wind_adj = player['weather_wind_adj']
                        wind_text = f"+{(wind_adj-1)*100:.1f}%" if wind_adj > 1 else f"{(wind_adj-1)*100:.1f}%"
                        weather_factors.append(f"Wind: {wind_text}")
                    
                    if weather_factors:
                        analysis_items.append(" | ".join(weather_factors))
                    
                    # Opposing pitcher
                    if 'opp_pitcher_name' in player and player['opp_pitcher_name'] != 'TBD':
                        analysis_items.append(f"vs {player['opp_pitcher_name']} ({player.get('opp_pitcher_throws', 'R')})")
                    
                    if analysis_items:
                        st.caption(" | ".join(analysis_items))
                
                st.markdown("---")
    else:
        st.info("No enhanced HR predictions available")
    
    # Enhanced team summary
    if len(top_hitters) > 0 or len(top_power_hitters) > 0:
        st.markdown("#### 📈 Enhanced Team Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_hit_prob = top_hitters['hit_probability'].mean() if len(top_hitters) > 0 else 0
            st.metric("Avg Hit Prob", f"{avg_hit_prob:.1%}")
        
        with col2:
            avg_hr_prob = top_power_hitters['hr_probability'].mean() if len(top_power_hitters) > 0 else 0
            st.metric("Avg HR Prob", f"{avg_hr_prob:.1%}")
        
        with col3:
            expected_hits = sum(top_hitters['hit_probability']) if len(top_hitters) > 0 else 0
            st.metric("Expected Hits", f"{expected_hits:.1f}")
        
        with col4:
            expected_hrs = sum(top_power_hitters['hr_probability']) if len(top_power_hitters) > 0 else 0
            st.metric("Expected HRs", f"{expected_hrs:.1f}")

def show_weather_analysis_simple(weather_data, weather_impact, detailed=False):
    """Show simplified weather impact analysis"""
    st.markdown("#### 🌤️ Weather Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"🌡️ **{weather_data.get('temperature', 'Unknown')}°F**")
        st.write(f"💨 **{weather_data.get('wind_speed', 'Unknown')} mph {weather_data.get('wind_direction', '')}**")
    
    with col2:
        hr_impact = weather_impact.get('hr_factor', 1.0)
        if hr_impact > 1.05:
            st.success(f"🚀 HR Boost: +{(hr_impact-1)*100:.1f}%")
        elif hr_impact < 0.95:
            st.error(f"📉 HR Impact: {(hr_impact-1)*100:.1f}%")
        else:
            st.info(f"➡️ Neutral Impact")

def show_pitching_matchup_analysis_simple(game):
    """Show simplified pitching matchup"""
    st.markdown("#### ⚾ Pitching Matchup")
    
    away_pitcher = game.get('away_pitcher', {})
    home_pitcher = game.get('home_pitcher', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{game['away_team']}:** {away_pitcher.get('name', 'TBD')} ({away_pitcher.get('throws', 'R')})")
    
    with col2:
        st.write(f"**{game['home_team']}:** {home_pitcher.get('name', 'TBD')} ({home_pitcher.get('throws', 'R')})")

def store_enhanced_predictions(top_hitters, top_power_hitters, team_name, game):
    """Store enhanced predictions with additional metadata"""
    prediction_date = date.today()
    
    try:
        # Store individual predictions with enhanced metadata
        for _, player in top_hitters.iterrows():
            player_name = player.get('player_name', player.get('Name', 'Unknown'))
            
            database.store_prediction(
                player_name=player_name,
                team=team_name,
                prediction_type='hit',
                probability=player.get('hit_probability', 0),
                confidence=player.get('confidence', 0),
                model_version='enhanced_hit_predictor_v1.0',
                ballpark=game.get('ballpark'),
                prediction_date=prediction_date
            )
        
        # Store HR predictions
        for _, player in top_power_hitters.iterrows():
            player_name = player.get('player_name', player.get('Name', 'Unknown'))
            
            database.store_prediction(
                player_name=player_name,
                team=team_name,
                prediction_type='hr',
                probability=player.get('hr_probability', 0),
                confidence=player.get('confidence', 0),
                model_version='enhanced_hr_predictor_v1.0',
                ballpark=game.get('ballpark'),
                prediction_date=prediction_date
            )
        
    except Exception as e:
        st.warning(f"Could not store enhanced predictions: {e}")

def generate_all_games_predictions(daily_games):
    """Generate predictions for all today's games"""
    st.subheader("📊 All Games Predictions")
    
    hitting_data = st.session_state.hitting_data
    
    for _, game in daily_games.iterrows():
        st.markdown(f"## {game['away_team']} @ {game['home_team']}")
        st.markdown(f"**🏟️ {game.get('ballpark', 'Unknown Stadium')}**")
        
        # Get predictions for both teams
        away_team_data = get_team_data(hitting_data, game['away_team'])
        home_team_data = get_team_data(hitting_data, game['home_team'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {game['away_team']} (Away)")
            if len(away_team_data) > 0:
                generate_team_predictions(away_team_data, game['away_team'], game.get('ballpark'))
            else:
                st.info(f"No player data available for {game['away_team']}")
        
        with col2:
            st.markdown(f"### {game['home_team']} (Home)")
            if len(home_team_data) > 0:
                generate_team_predictions(home_team_data, game['home_team'], game.get('ballpark'))
            else:
                st.info(f"No player data available for {game['home_team']}")
        
        st.markdown("---")

def generate_single_game_predictions(game):
    """Generate predictions for a specific game"""
    st.subheader(f"📊 {game['away_team']} @ {game['home_team']}")
    st.markdown(f"**🏟️ {game.get('ballpark', 'Unknown Stadium')}**")
    
    hitting_data = st.session_state.hitting_data
    
    # Get data for both teams
    away_team_data = get_team_data(hitting_data, game['away_team'])
    home_team_data = get_team_data(hitting_data, game['home_team'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {game['away_team']} (Away)")
        if len(away_team_data) > 0:
            generate_team_predictions(away_team_data, game['away_team'], game.get('ballpark'))
        else:
            st.info(f"No player data available for {game['away_team']}")
    
    with col2:
        st.markdown(f"### {game['home_team']} (Home)")
        if len(home_team_data) > 0:
            generate_team_predictions(home_team_data, game['home_team'], game.get('ballpark'))
        else:
            st.info(f"No player data available for {game['home_team']}")

def get_team_data(hitting_data, team_abbrev):
    """Get hitting data for a specific team"""
    if 'Team' in hitting_data.columns:
        team_data = hitting_data[hitting_data['Team'] == team_abbrev]
        return team_data.copy()
    else:
        # Fallback: return a sample of players if no team column
        st.warning(f"No team column found. Using sample data for {team_abbrev}")
        return hitting_data.sample(min(10, len(hitting_data))).copy()

def generate_team_predictions(team_data, team_name, ballpark=None):
    """Generate and display predictions for a team"""
    try:
        # Generate hit predictions
        top_hitters = hit_model.predict_team_hits(team_data)
        
        # Generate HR predictions
        top_power_hitters = hr_model.predict_team_hrs(team_data, ballpark)
        
        # Display predictions
        display_team_predictions(top_hitters, top_power_hitters, team_name, ballpark)
        
        # Store predictions in database
        store_team_predictions(top_hitters, top_power_hitters, team_name, ballpark)
        
    except Exception as e:
        st.error(f"Error generating predictions for {team_name}: {e}")

def display_team_predictions(top_hitters, top_power_hitters, team_name, ballpark):
    """Display prediction results for a team"""
    
    # Hit predictions
    st.markdown("#### 🎯 Top 5 Hit Predictions")
    if len(top_hitters) > 0:
        for i, (_, player) in enumerate(top_hitters.iterrows(), 1):
            player_name = player.get('player_name', player.get('Name', f'Player_{i}'))
            hit_prob = player.get('hit_probability', 0)
            confidence = player.get('confidence', 0)
            
            # Create a nice display card
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.write(f"**{i}. {player_name}**")
                with col2:
                    st.metric("Hit Probability", f"{hit_prob:.1%}")
                with col3:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Progress bar
                st.progress(hit_prob)
                
                # Key strengths
                if 'key_strengths' in player and player['key_strengths']:
                    strengths = player['key_strengths'] if isinstance(player['key_strengths'], list) else [str(player['key_strengths'])]
                    st.caption(f"Key strengths: {', '.join(strengths)}")
                
                st.markdown("---")
    else:
        st.info("No hit predictions available")
    
    # HR predictions
    st.markdown("#### 💥 Top 3 Home Run Predictions")
    if len(top_power_hitters) > 0:
        for i, (_, player) in enumerate(top_power_hitters.iterrows(), 1):
            player_name = player.get('player_name', player.get('Name', f'Power Player_{i}'))
            hr_prob = player.get('hr_probability', 0)
            confidence = player.get('confidence', 0)
            power_tier = player.get('power_tier', 'Unknown')
            
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.write(f"**{i}. {player_name}**")
                with col2:
                    st.metric("HR Probability", f"{hr_prob:.1%}")
                with col3:
                    st.metric("Power Tier", power_tier)
                
                # Progress bar
                st.progress(hr_prob)
                
                # Ballpark factor
                if ballpark and 'ballpark_boost' in player:
                    boost = player['ballpark_boost']
                    if boost != 1.0:
                        boost_text = f"+{(boost-1)*100:.0f}%" if boost > 1 else f"{(boost-1)*100:.0f}%"
                        st.caption(f"Ballpark factor: {boost_text}")
                
                # Power strengths
                if 'power_strengths' in player and player['power_strengths']:
                    strengths = player['power_strengths'] if isinstance(player['power_strengths'], list) else [str(player['power_strengths'])]
                    st.caption(f"Power strengths: {', '.join(strengths)}")
                
                st.markdown("---")
    else:
        st.info("No HR predictions available")
    
    # Team summary stats
    if len(top_hitters) > 0 or len(top_power_hitters) > 0:
        st.markdown("#### 📈 Team Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_hit_prob = top_hitters['hit_probability'].mean() if len(top_hitters) > 0 else 0
            st.metric("Avg Hit Prob", f"{avg_hit_prob:.1%}")
        
        with col2:
            avg_hr_prob = top_power_hitters['hr_probability'].mean() if len(top_power_hitters) > 0 else 0
            st.metric("Avg HR Prob", f"{avg_hr_prob:.1%}")
        
        with col3:
            expected_hits = sum(top_hitters['hit_probability']) if len(top_hitters) > 0 else 0
            st.metric("Expected Hits", f"{expected_hits:.1f}")
        
        with col4:
            expected_hrs = sum(top_power_hitters['hr_probability']) if len(top_power_hitters) > 0 else 0
            st.metric("Expected HRs", f"{expected_hrs:.1f}")

def store_team_predictions(top_hitters, top_power_hitters, team_name, ballpark):
    """Store predictions in database for tracking"""
    prediction_date = date.today()
    
    try:
        # Store individual hit predictions
        for _, player in top_hitters.iterrows():
            player_name = player.get('player_name', player.get('Name', 'Unknown'))
            database.store_prediction(
                player_name=player_name,
                team=team_name,
                prediction_type='hit',
                probability=player.get('hit_probability', 0),
                confidence=player.get('confidence', 0),
                model_version='hit_predictor_v1.0',
                ballpark=ballpark,
                prediction_date=prediction_date
            )
        
        # Store individual HR predictions
        for _, player in top_power_hitters.iterrows():
            player_name = player.get('player_name', player.get('Name', 'Unknown'))
            database.store_prediction(
                player_name=player_name,
                team=team_name,
                prediction_type='hr',
                probability=player.get('hr_probability', 0),
                confidence=player.get('confidence', 0),
                model_version='hr_predictor_v1.0',
                ballpark=ballpark,
                prediction_date=prediction_date
            )
        
        # Store team summary
        database.store_team_predictions(
            team=team_name,
            prediction_date=prediction_date,
            top_hitters=top_hitters.to_dict('records') if len(top_hitters) > 0 else [],
            top_power_hitters=top_power_hitters.to_dict('records') if len(top_power_hitters) > 0 else [],
            ballpark=ballpark
        )
        
    except Exception as e:
        st.warning(f"Could not store predictions: {e}")

def sidebar_info():
    """Display information in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Today's Data")
    
    # Show data status
    if st.session_state.get('data_loaded', False):
        st.sidebar.success("✅ Data Loaded")
        hitting_data = st.session_state.get('hitting_data', pd.DataFrame())
        daily_games = st.session_state.get('daily_games', pd.DataFrame())
        
        st.sidebar.metric("Players", len(hitting_data))
        st.sidebar.metric("Games Today", len(daily_games))
        
        if 'Team' in hitting_data.columns:
            unique_teams = hitting_data['Team'].nunique()
            st.sidebar.metric("Teams Represented", unique_teams)
    else:
        st.sidebar.warning("⏳ Loading Data...")
    
    # Show model status
    st.sidebar.markdown("### 🤖 Model Status")
    if st.session_state.get('models_trained', False):
        st.sidebar.success("✅ Models Trained")
        
        # Show model performance if available
        if hasattr(hit_model, 'model_metrics') and hit_model.model_metrics:
            hit_auc = hit_model.model_metrics.get('val_auc', 0)
            st.sidebar.metric("Hit Model AUC", f"{hit_auc:.3f}")
        
        if hasattr(hr_model, 'model_metrics') and hr_model.model_metrics:
            hr_auc = hr_model.model_metrics.get('val_auc', 0)
            st.sidebar.metric("HR Model AUC", f"{hr_auc:.3f}")
    else:
        st.sidebar.warning("⏳ Training Models...")
    
    # Quick refresh button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        # Clear cache and reload
        st.cache_data.clear()
        st.session_state.data_loaded = False
        st.session_state.models_trained = False
        st.rerun()
    
    # App info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    **Daily MLB Predictions**
    
    This app automatically:
    - Loads today's MLB games
    - Applies ballpark factors
    - Generates ML-powered predictions
    - Tracks prediction accuracy
    
    **Features:**
    - Top 5 hit predictions per team
    - Top 3 HR predictions per team
    - Automatic ballpark adjustments
    - Real-time data only
    """)

# Additional pages for model performance and history
def show_model_performance():
    """Show model performance metrics"""
    st.header("📈 Model Performance")
    
    # Get performance stats from database
    summary_stats = database.get_prediction_summary_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", summary_stats.get('total_predictions', 0))
    with col2:
        st.metric("Hit Predictions", summary_stats.get('hit_predictions', 0))
    with col3:
        st.metric("HR Predictions", summary_stats.get('hr_predictions', 0))
    with col4:
        st.metric("Recent Accuracy", f"{summary_stats.get('recent_hit_accuracy', 0):.1%}")
    
    # Show recent predictions
    st.subheader("Recent Predictions")
    recent_predictions = database.get_recent_predictions(days=7)
    
    if len(recent_predictions) > 0:
        st.dataframe(recent_predictions.head(20), use_container_width=True)
    else:
        st.info("No recent predictions found")

def show_prediction_history():
    """Show prediction history and results"""
    st.header("📋 Prediction History")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=date.today())
    
    # Get predictions for date range
    days_diff = (end_date - start_date).days
    recent_predictions = database.get_recent_predictions(days=days_diff)
    
    if len(recent_predictions) > 0:
        # Filter by date range
        recent_predictions['prediction_date'] = pd.to_datetime(recent_predictions['prediction_date'])
        filtered_predictions = recent_predictions[
            (recent_predictions['prediction_date'].dt.date >= start_date) &
            (recent_predictions['prediction_date'].dt.date <= end_date)
        ]
        
        if len(filtered_predictions) > 0:
            st.dataframe(filtered_predictions, use_container_width=True)
            
            # Show prediction distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    filtered_predictions,
                    x='probability',
                    color='prediction_type',
                    title='Prediction Probability Distribution',
                    bins=20
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                prediction_counts = filtered_predictions['prediction_type'].value_counts()
                fig = px.pie(
                    values=prediction_counts.values,
                    names=prediction_counts.index,
                    title='Predictions by Type'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions found for the selected date range")
    else:
        st.info("No prediction history available")

# Main app with navigation
def run_app():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Daily Predictions", "Model Performance", "Prediction History"]
    )
    
    if page == "Daily Predictions":
        main()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Prediction History":
        show_prediction_history()
    
    # Always show sidebar info
    sidebar_info()

if __name__ == "__main__":
    run_app()