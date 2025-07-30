"""
MLB Hit and Home Run Predictor - Streamlit Interface

This is the main application interface that allows users to:
1. Select teams, players, or view all games
2. Get top 5 hit predictions and top 3 HR predictions
3. View model performance and accuracy metrics
4. Track prediction success over time

Features:
- Interactive team/player selection
- Real-time predictions using trained def train_models(data):
- Performance dashboard with accuracy metrics
- Sabermetric explanations for predictions
- Database integration for tracking results
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
    from data.data_collector import MLBDataCollector
    from data.feature_engineer import SabermetricFeatureEngineer
    from models.hit_predictor import HitPredictor
    from models.hr_predictor import HomeRunPredictor
    from data.database import MLBPredictionDatabase
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="MLB Hit & HR Predictor",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize data collector, feature engineer, models, and database"""
    collector = MLBDataCollector()
    engineer = SabermetricFeatureEngineer()
    hit_model = HitPredictor()
    hr_model = HomeRunPredictor()
    database = MLBPredictionDatabase()
    
    return collector, engineer, hit_model, hr_model, database

collector, engineer, hit_model, hr_model, database = initialize_components()

# Load and prepare data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_mlb_data():
    """Load and prepare MLB data for predictions"""
    try:
        # Collect current season data
        hitting_data, pitching_data = collector.collect_all_data()
        
        # Engineer features
        final_data = engineer.engineer_all_features(hitting_data)
        
        return final_data, pitching_data
    except Exception as e:
        st.error(f"Error loading MLB data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def train_models(data):
    """Train both hit and HR prediction models with robust checks"""
    if len(data) == 0:
        st.error("No data available for training.")
        print("[DEBUG] No data available for training.")
        return False

    try:
        # Prepare hit prediction data
        hit_features, hit_targets, hit_feature_names = hit_model.prepare_training_data(data)
        print(f"[DEBUG] Hit features shape: {getattr(hit_features, 'shape', None)}")
        print(f"[DEBUG] Hit targets length: {len(hit_targets) if hit_targets is not None else None}")
        if hit_features is None or hit_targets is None or len(hit_features) == 0 or len(hit_targets) == 0:
            st.error("Hit training data is empty or invalid.")
            print("[DEBUG] Hit training data is empty or invalid.")
            return False

        # Train hit prediction model
        hit_metrics = hit_model.train_model(hit_features, hit_targets, hit_feature_names)
        if not hasattr(hit_model, 'model') or hit_model.model is None:
            st.error("Hit model was not fit. Check training data and model code.")
            print("[DEBUG] Hit model was not fit after training.")
            return False
        print(f"[DEBUG] Hit model trained. Metrics: {hit_metrics}")

        # Prepare HR prediction data
        hr_features, hr_targets, hr_feature_names = hr_model.prepare_training_data(data)
        print(f"[DEBUG] HR features shape: {getattr(hr_features, 'shape', None)}")
        print(f"[DEBUG] HR targets length: {len(hr_targets) if hr_targets is not None else None}")
        if hr_features is None or hr_targets is None or len(hr_features) == 0 or len(hr_targets) == 0:
            st.error("HR training data is empty or invalid.")
            print("[DEBUG] HR training data is empty or invalid.")
            return False

        # Train HR prediction model  
        hr_metrics = hr_model.train_model(hr_features, hr_targets, hr_feature_names)
        if not hasattr(hr_model, 'model') or hr_model.model is None:
            st.error("HR model was not fit. Check training data and model code.")
            print("[DEBUG] HR model was not fit after training.")
            return False
        print(f"[DEBUG] HR model trained. Metrics: {hr_metrics}")

        # Store feature importance in database
        if hit_model.feature_importance:
            database.store_feature_importance('hit', hit_model.feature_importance, 'v1.0', len(data))
        if hr_model.feature_importance:
            database.store_feature_importance('hr', hr_model.feature_importance, 'v1.0', len(data))

        return True
    except Exception as e:
        st.error(f"Error training models: {e}")
        print(f"[DEBUG] Exception in train_models: {e}")
        return False

# Main application
def main():
    st.title("Sabermertric Predictor")
    st.markdown("### Advanced Sabermetric Predictions Using Machine Learning")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Daily Predictions", "Model Performance", "Sabermetric Insights", "Prediction History"]
    )
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading MLB data and training models..."):
            hitting_data, pitching_data = load_mlb_data()
            
            if len(hitting_data) > 0:
                st.session_state.hitting_data = hitting_data
                st.session_state.pitching_data = pitching_data
                st.session_state.data_loaded = True
                
                # Train models
                if train_models(hitting_data):
                    st.session_state.models_trained = True
                    st.success("✅ Data loaded and models trained successfully!")
                else:
                    st.error("❌ Error training models")
            else:
                st.error("❌ Failed to load MLB data")
                return
    
    # Route to different pages
    if page == "Daily Predictions":
        daily_predictions_page()
    elif page == "Model Performance":
        model_performance_page()
    elif page == "Sabermetric Insights":
        sabermetric_insights_page()
    elif page == "Prediction History":
        prediction_history_page()

def daily_predictions_page():
    """Main prediction interface"""
    st.header("🎯 Daily Predictions")
    
    if not st.session_state.models_trained:
        st.warning("Models are still training. Please wait...")
        return
    
    # Standardize team names and drop NaNs
    if 'Team' in st.session_state.hitting_data.columns:
        st.session_state.hitting_data['Team'] = st.session_state.hitting_data['Team'].astype(str).str.strip().str.upper()
        teams = sorted(st.session_state.hitting_data['Team'].dropna().unique())
    else:
        teams = ['No Teams Found']

    # Selection options
    col1, col2, col3 = st.columns(3)

    with col1:
        prediction_mode = st.selectbox(
            "Prediction Mode:",
            ["All Games", "Specific Team", "Specific Player"]
        )

    with col2:
        if prediction_mode == "Specific Team":
            selected_team = st.selectbox("Select Team:", teams)
        else:
            selected_team = None

    with col3:
        ballpark = st.selectbox(
            "Ballpark (optional):",
            ["", "Yankee Stadium", "Fenway Park", "Coors Field", "Petco Park", "Great American Ball Park"]
        )

    # Debugging output for teams and data shape
    # st.write("Teams available:", teams)
    st.write("Data shape:", st.session_state.hitting_data.shape)
    
    # Make predictions based on selection
    if st.button("🔮 Generate Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            if prediction_mode == "All Games":
                teams = st.session_state.hitting_data['Team'].unique() if 'Team' in st.session_state.hitting_data.columns else []
                for team in teams:
                    team_data = st.session_state.hitting_data[st.session_state.hitting_data['Team'] == team]
                    try:
                        top_hitters = hit_model.predict_team_hits(team_data)
                        top_power_hitters = hr_model.predict_team_hrs(team_data, ballpark if ballpark else None)
                        st.markdown(f"## {team}")
                        display_predictions(top_hitters, top_power_hitters, ballpark)
                        store_daily_predictions(top_hitters, top_power_hitters, team, ballpark)
                    except Exception as e:
                        st.error(f"Error generating predictions for {team}: {e}")
            elif prediction_mode == "Specific Team":
                data_subset = st.session_state.hitting_data[
                    st.session_state.hitting_data['Team'] == selected_team
                ] if 'Team' in st.session_state.hitting_data.columns else st.session_state.hitting_data.head(20)
                try:
                    top_hitters = hit_model.predict_team_hits(data_subset)
                    top_power_hitters = hr_model.predict_team_hrs(data_subset, ballpark if ballpark else None)
                    display_predictions(top_hitters, top_power_hitters, ballpark)
                    store_daily_predictions(top_hitters, top_power_hitters, selected_team, ballpark)
                except Exception as e:
                    st.error(f"Error generating predictions: {e}")
            else:  # Specific player - would need player selection UI
                data_subset = st.session_state.hitting_data.head(10)
                try:
                    top_hitters = hit_model.predict_team_hits(data_subset)
                    top_power_hitters = hr_model.predict_team_hrs(data_subset, ballpark if ballpark else None)
                    display_predictions(top_hitters, top_power_hitters, ballpark)
                    store_daily_predictions(top_hitters, top_power_hitters, selected_team, ballpark)
                except Exception as e:
                    st.error(f"Error generating predictions: {e}")

def display_predictions(top_hitters, top_power_hitters, ballpark):
    """Display prediction results in a nice format"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Top 5 Hit Predictions")
        if len(top_hitters) > 0:
            for i, (_, player) in enumerate(top_hitters.iterrows(), 1):
                with st.container():
                    player_name = player['player_name'] if 'player_name' in player else player.get('Name', f'Player_{i}')
                    st.markdown(f"""
                    **#{i}. {player_name}**
                    - Hit Probability: **{player['hit_probability']:.1%}**
                    - Confidence: {player['confidence']:.1%}
                    - Key Strengths: {', '.join(player['key_strengths'])}
                    """)
                    st.progress(player['hit_probability'])
                    st.markdown("---")
        else:
            st.info("No hit predictions available")

    with col2:
        st.subheader("💥 Top 3 Home Run Predictions")
        if len(top_power_hitters) > 0:
            for i, (_, player) in enumerate(top_power_hitters.iterrows(), 1):
                with st.container():
                    player_name = player['player_name'] if 'player_name' in player else player.get('Name', f'Player_{i}')
                    st.markdown(f"""
                    **#{i}. {player_name}**
                    - HR Probability: **{player['hr_probability']:.1%}**
                    - Confidence: {player['confidence']:.1%}
                    - Power Tier: {player['power_tier']}
                    - Strengths: {', '.join(player['power_strengths'])}
                    """)
                    if ballpark and 'ballpark_boost' in player:
                        st.markdown(f"- Ballpark Factor: {player['ballpark_boost']}")
                    st.progress(player['hr_probability'])
                    st.markdown("---")
        else:
            st.info("No HR predictions available")
    
    # Summary statistics
    st.subheader("📊 Prediction Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_hit_prob = top_hitters['hit_probability'].mean() if len(top_hitters) > 0 else 0
        st.metric("Avg Hit Probability", f"{avg_hit_prob:.1%}")
    
    with col2:
        avg_hr_prob = top_power_hitters['hr_probability'].mean() if len(top_power_hitters) > 0 else 0
        st.metric("Avg HR Probability", f"{avg_hr_prob:.1%}")
    
    with col3:
        total_expected_hits = sum(top_hitters['hit_probability']) if len(top_hitters) > 0 else 0
        st.metric("Expected Hits", f"{total_expected_hits:.1f}")
    
    with col4:
        total_expected_hrs = sum(top_power_hitters['hr_probability']) if len(top_power_hitters) > 0 else 0
        st.metric("Expected HRs", f"{total_expected_hrs:.1f}")

def store_daily_predictions(top_hitters, top_power_hitters, team, ballpark):
    """Store predictions in database for tracking"""
    prediction_date = date.today()
    
    try:
        # Store individual hit predictions
        for _, player in top_hitters.iterrows():
            database.store_prediction(
                player_name=player['player_name'],
                team=team or "Unknown",
                prediction_type='hit',
                probability=player['hit_probability'],
                confidence=player['confidence'],
                model_version='hit_predictor_v1.0',
                ballpark=ballpark,
                prediction_date=prediction_date
            )
        
        # Store individual HR predictions
        for _, player in top_power_hitters.iterrows():
            database.store_prediction(
                player_name=player['player_name'],
                team=team or "Unknown", 
                prediction_type='hr',
                probability=player['hr_probability'],
                confidence=player['confidence'],
                model_version='hr_predictor_v1.0',
                ballpark=ballpark,
                prediction_date=prediction_date
            )
        
        # Store team summary
        database.store_team_predictions(
            team=team or "Mixed",
            prediction_date=prediction_date,
            top_hitters=top_hitters,
            top_power_hitters=top_power_hitters,
            ballpark=ballpark
        )
        
        st.success("✅ Predictions stored for tracking!")
        
    except Exception as e:
        st.warning(f"Could not store predictions: {e}")

def model_performance_page():
    """Display model performance metrics and analytics"""
    st.header("📈 Model Performance Dashboard")
    
    # Get performance statistics
    summary_stats = database.get_prediction_summary_stats()
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", summary_stats.get('total_predictions', 0))
    with col2:
        st.metric("Hit Predictions", summary_stats.get('hit_predictions', 0))
    with col3:
        st.metric("HR Predictions", summary_stats.get('hr_predictions', 0))
    with col4:
        st.metric("Games Tracked", summary_stats.get('total_outcomes', 0))
    
    st.markdown("---")
    
    # Recent accuracy metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Hit Prediction Accuracy")
        hit_accuracy = database.calculate_prediction_accuracy(prediction_type='hit')
        
        if 'error' not in hit_accuracy:
            st.metric("Accuracy (30 days)", f"{hit_accuracy['accuracy']:.1%}")
            st.metric("Precision", f"{hit_accuracy['precision']:.1%}")
            st.metric("Recall", f"{hit_accuracy['recall']:.1%}")
            
            if hit_accuracy['calibration_score']:
                st.metric("Calibration Score", f"{hit_accuracy['calibration_score']:.3f}")
                st.caption("Lower is better - measures how well probabilities match actual rates")
        else:
            st.info("Not enough data for hit accuracy analysis")
    
    with col2:
        st.subheader("💥 HR Prediction Accuracy")
        hr_accuracy = database.calculate_prediction_accuracy(prediction_type='hr')
        
        if 'error' not in hr_accuracy:
            st.metric("Accuracy (30 days)", f"{hr_accuracy['accuracy']:.1%}")
            st.metric("Precision", f"{hr_accuracy['precision']:.1%}")
            st.metric("Recall", f"{hr_accuracy['recall']:.1%}")
            
            if hr_accuracy['calibration_score']:
                st.metric("Calibration Score", f"{hr_accuracy['calibration_score']:.3f}")
        else:
            st.info("Not enough data for HR accuracy analysis")
    
    # Performance trends
    st.subheader("📊 Performance Trends")
    
    hit_history = database.get_model_performance_history('hit', days=90)
    hr_history = database.get_model_performance_history('hr', days=90)
    
    if len(hit_history) > 0 or len(hr_history) > 0:
        fig = go.Figure()
        
        if len(hit_history) > 0:
            fig.add_trace(go.Scatter(
                x=hit_history['date_range_start'],
                y=hit_history['accuracy'],
                mode='lines+markers',
                name='Hit Accuracy',
                line=dict(color='blue')
            ))
        
        if len(hr_history) > 0:
            fig.add_trace(go.Scatter(
                x=hr_history['date_range_start'],
                y=hr_history['accuracy'],
                mode='lines+markers',
                name='HR Accuracy',
                line=dict(color='red')
            ))
        
        fig.update_layout(
            title="Model Accuracy Over Time",
            xaxis_title="Date",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical performance data available yet")
    
    # Feature importance analysis
    st.subheader("🔍 Feature Importance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if hit_model.feature_importance:
            hit_importance_df = pd.DataFrame(
                list(hit_model.feature_importance.items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                hit_importance_df.tail(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Hit Prediction - Top 10 Features'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if hr_model.feature_importance:
            hr_importance_df = pd.DataFrame(
                list(hr_model.feature_importance.items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                hr_importance_df.tail(10),
                x='Importance', 
                y='Feature',
                orientation='h',
                title='HR Prediction - Top 10 Features'
            )
            st.plotly_chart(fig, use_container_width=True)

def sabermetric_insights_page():
    """Explain the sabermetric features and model logic"""
    st.header("🧠 Sabermetric Insights & Model Explanation")
    
    # Model explanations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Hit Prediction Model")
        
        hit_explanation = hit_model.get_model_explanation()
        
        st.markdown("**Key Features:**")
        for feature, description in hit_explanation['key_features'].items():
            st.markdown(f"- **{feature}**: {description}")
        
        st.markdown("**Model Logic:**")
        for logic_point in hit_explanation['model_logic']:
            st.markdown(f"- {logic_point}")
        
        if hit_explanation['model_metrics']:
            st.markdown("**Current Performance:**")
            st.markdown(f"- Validation Accuracy: {hit_explanation['model_metrics']['val_accuracy']:.1%}")
            st.markdown(f"- Validation AUC: {hit_explanation['model_metrics']['val_auc']:.3f}")
    
    with col2:
        st.subheader("💥 HR Prediction Model")
        
        hr_explanation = hr_model.get_model_explanation()
        
        st.markdown("**Key Features:**")
        for feature, description in hr_explanation['key_features'].items():
            st.markdown(f"- **{feature}**: {description}")
        
        st.markdown("**HR Physics:**")
        for metric, value in hr_explanation['hr_physics'].items():
            st.markdown(f"- **{metric.replace('_', ' ').title()}**: {value}")
        
        st.markdown("**Model Logic:**")
        for logic_point in hr_explanation['model_logic']:
            st.markdown(f"- {logic_point}")
    
    st.markdown("---")
    
    # Sabermetric glossary
    st.subheader("📚 Sabermetric Glossary")
    
    glossary = {
        "Contact%": "Percentage of swings that result in contact with the ball",
        "Whiff%": "Percentage of swings that miss the ball entirely", 
        "Chase%": "Percentage of swings at pitches outside the strike zone",
        "Zone%": "Percentage of pitches seen in the strike zone",
        "Exit Velocity": "Speed of the ball off the bat, measured in mph",
        "Launch Angle": "Angle of the ball's trajectory off the bat",
        "Barrel%": "Percentage of balls hit with optimal exit velocity and launch angle",
        "Hard Hit%": "Percentage of balls hit with 95+ mph exit velocity",
        "ISO (Isolated Power)": "Slugging percentage minus batting average, measures extra-base power",
        "wOBA": "Weighted on-base average - weights different outcomes by their run value",
        "xwOBA": "Expected wOBA based on quality of contact, not outcome",
        "BABIP": "Batting average on balls in play - measures luck vs skill"
    }
    
    for term, definition in glossary.items():
        with st.expander(f"**{term}**"):
            st.write(definition)
    
    # Feature engineering insights
    st.subheader("⚙️ Feature Engineering Insights")
    
    st.markdown("""
    **Hit Prediction Features:**
    - **Contact Rate Plus**: Player's contact rate relative to league average (100 = average)
    - **Contact Skill**: Inverse of whiff rate - how good at making contact
    - **Plate Discipline**: Combination of zone% and chase% - pitch selection ability
    - **BABIP Sustainability**: Whether player's BABIP is based on skill or luck
    
    **HR Prediction Features:**
    - **Power Score**: Composite metric combining ISO, barrel rate, and launch conditions
    - **HR Launch Score**: Exit velocity and launch angle optimized for home runs
    - **Power Efficiency**: How much power per hard-hit ball (ISO / Hard Hit%)
    - **Barrel Power Score**: Barrel rate weighted by exit velocity
    """)

def prediction_history_page():
    """Show historical predictions and their outcomes"""
    st.header("📋 Prediction History & Results")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=date.today())
    
    # Get recent predictions
    recent_predictions = database.get_recent_predictions(days=(end_date - start_date).days)
    
    if len(recent_predictions) > 0:
        st.subheader("🔮 Recent Predictions")
        
        # Filter by date range
        recent_predictions['prediction_date'] = pd.to_datetime(recent_predictions['prediction_date'])
        filtered_predictions = recent_predictions[
            (recent_predictions['prediction_date'].dt.date >= start_date) &
            (recent_predictions['prediction_date'].dt.date <= end_date)
        ]
        
        # Display predictions table
        st.dataframe(
            filtered_predictions.sort_values(['prediction_date', 'probability'], ascending=[False, False]),
            use_container_width=True
        )
        
        # Prediction distribution
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
    
    # Top performing predictions
    st.subheader("🏆 Top Performing Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Hit Predictions**")
        top_hit_predictions = database.get_top_performing_predictions('hit', limit=5)
        if len(top_hit_predictions) > 0:
            st.dataframe(top_hit_predictions, use_container_width=True)
        else:
            st.info("No verified hit predictions yet")
    
    with col2:
        st.markdown("**HR Predictions**")
        top_hr_predictions = database.get_top_performing_predictions('hr', limit=5)
        if len(top_hr_predictions) > 0:
            st.dataframe(top_hr_predictions, use_container_width=True)
        else:
            st.info("No verified HR predictions yet")

# Sidebar information
def sidebar_info():
    """Display information in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About This App")
    st.sidebar.markdown("""
    This app uses advanced sabermetrics and machine learning to predict:
    - **Hit Probability**: Likelihood of getting a hit
    - **Home Run Probability**: Likelihood of hitting a HR
    
    **Key Features:**
    - XGBoost models optimized for baseball data
    - 15+ sabermetric features per prediction
    - Real-time MLB data integration
    - Prediction accuracy tracking
    - Ballpark factor adjustments
    """)
    
    st.sidebar.markdown("### Model Stats")
    if st.session_state.get('models_trained', False):
        st.sidebar.success("✅ Models Trained")
        if hit_model.model_metrics:
            st.sidebar.markdown(f"Hit Model AUC: {hit_model.model_metrics.get('val_auc', 0):.3f}")
        if hr_model.model_metrics:
            st.sidebar.markdown(f"HR Model AUC: {hr_model.model_metrics.get('val_auc', 0):.3f}")
    else:
        st.sidebar.warning("⏳ Training Models...")

if __name__ == "__main__":
    sidebar_info()
    main()