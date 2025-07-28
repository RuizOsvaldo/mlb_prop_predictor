"""
Home Run Prediction Model

This model predicts the probability of a batter hitting a home run in their next at-bat
using advanced sabermetric features focused on power and launch conditions.

Model Architecture:
- Uses XGBoost optimized for the rare event nature of HRs (class imbalance)
- Features focus on power metrics, launch conditions, and barrel rate
- Incorporates ballpark factors and weather when available

Key Features Used:
1. Barrel% - Rate of optimal launch conditions (most important for HRs)
2. ISO (Isolated Power) - Raw power ability 
3. Exit Velocity + Launch Angle - Physics of home runs
4. Hard Hit% - Consistency of solid contact
5. Power Score - Composite power metric
6. Ballpark factors - How HR-friendly is the venue
7. Recent power trends - Hot power streaks

HR Physics:
- Optimal exit velocity: 95+ mph
- Optimal launch angle: 25-35 degrees  
- Barrels (combining both) have ~50% HR probability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HomeRunPredictor:
    def __init__(self):
        """Initialize the home run prediction model"""
        self.model = None
        self.feature_importance = None
        self.feature_names = None
        self.model_metrics = {}
        
        # XGBoost parameters optimized for HR prediction (rare event)
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',              # AUC better for imbalanced data
            'max_depth': 8,                    # Deeper trees for complex power patterns
            'learning_rate': 0.05,             # Lower learning rate for stability
            'n_estimators': 300,               # More trees for rare event learning
            'subsample': 0.7,                  # More aggressive sampling
            'colsample_bytree': 0.8,
            'scale_pos_weight': 10,            # Handle class imbalance (HRs are rare)
            'min_child_weight': 3,             # Prevent overfitting on rare events
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Ballpark HR factors (simplified - would be more comprehensive in real implementation)
        self.ballpark_factors = {
            'Coors Field': 1.15,
            'Great American Ball Park': 1.12,
            'Yankee Stadium': 1.10,
            'Citizens Bank Park': 1.08,
            'Progressive Field': 1.05,
            'Fenway Park': 1.03,
            'Wrigley Field': 1.02,
            'Busch Stadium': 1.00,  # Neutral
            'Kauffman Stadium': 0.95,
            'Marlins Park': 0.92,
            'Tropicana Field': 0.90,
            'Petco Park': 0.88,
            'Comerica Park': 0.85
        }
    
    def prepare_training_data(self, player_stats_df):
        """
        Prepare training data for HR prediction
        
        Creates training labels based on HR rate and power metrics
        """
        print("Preparing training data for HR prediction...")
        
        training_data = player_stats_df.copy()
        
        # Create target variable: HR probability
        if 'HR' in training_data.columns and 'PA' in training_data.columns:
            # HR rate per plate appearance
            training_data['hr_rate'] = training_data['HR'] / training_data['PA']
        elif 'HR' in training_data.columns and 'AB' in training_data.columns:
            # HR rate per at-bat
            training_data['hr_rate'] = training_data['HR'] / training_data['AB']
        else:
            # Estimate from ISO and SLG
            if 'ISO' in training_data.columns:
                training_data['hr_rate'] = training_data['ISO'] * 0.15  # Rough approximation
        
        # Create binary target (above/below average HR rate)
        league_hr_rate = 0.034  # ~3.4% of PA result in HRs (2024 MLB)
        training_data['hr_target'] = (training_data['hr_rate'] > league_hr_rate).astype(int)
        
        # Select features for HR prediction
        hr_prediction_features = [
            'ISO', 'SLG', 'Barrel%', 'HardHit%', 'avg_exit_velocity', 'avg_launch_angle',
            'ISO_Plus', 'Barrel_Plus', 'HR_Launch_Score', 'Power_Efficiency',
            'Power_Score', 'Barrel_Power_Score', 'Power_Barrel_Interaction'
        ]
        
        # Filter to available features
        available_features = [f for f in hr_prediction_features if f in training_data.columns]
        
        if len(available_features) < 3:
            print("Warning: Limited power features available for training")
            # Fall back to basic power features
            basic_power_features = ['ISO', 'SLG', 'OPS']
            available_features.extend([f for f in basic_power_features if f in training_data.columns])
        
        # Handle missing values with power-appropriate defaults
        feature_data = training_data[available_features].copy()
        
        # For power metrics, use conservative defaults (below average)
        for col in feature_data.columns:
            if 'Plus' in col or 'Score' in col:
                feature_data[col] = feature_data[col].fillna(90)  # Below average (100 = average)
            else:
                feature_data[col] = feature_data[col].fillna(feature_data[col].median())
        
        print(f"Using {len(available_features)} features for HR prediction:")
        print(available_features)
        
        return feature_data, training_data['hr_target'], available_features
    
    def train_model(self, X, y, feature_names):
        """
        Train the XGBoost HR prediction model with class imbalance handling
        """
        print("Training home run prediction model...")
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight_dict = dict(zip(np.unique(y), class_weights))
        
        # Update scale_pos_weight based on actual class distribution
        hr_ratio = sum(y) / len(y)
        self.xgb_params['scale_pos_weight'] = (1 - hr_ratio) / hr_ratio
        
        print(f"HR rate in training data: {hr_ratio:.3f}")
        print(f"Using scale_pos_weight: {self.xgb_params['scale_pos_weight']:.2f}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize and train XGBoost model
        self.model = xgb.XGBClassifier(**self.xgb_params)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )
        
        # Store feature names and importance
        self.feature_names = feature_names
        self.feature_importance = dict(zip(
            feature_names, 
            self.model.feature_importances_
        ))
        
        # Calculate model metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        train_proba = self.model.predict_proba(X_train)[:, 1]
        val_proba = self.model.predict_proba(X_val)[:, 1]
        
        self.model_metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'train_auc': roc_auc_score(y_train, train_proba),
            'val_auc': roc_auc_score(y_val, val_proba),
            'train_precision': precision_score(y_train, train_pred),
            'val_precision': precision_score(y_val, val_pred),
            'train_recall': recall_score(y_train, train_pred),
            'val_recall': recall_score(y_val, val_pred),
            'hr_rate_in_training': hr_ratio
        }
        
        print("HR prediction model training complete!")
        print(f"Validation AUC: {self.model_metrics['val_auc']:.3f}")
        print(f"Validation Precision: {self.model_metrics['val_precision']:.3f}")
        print(f"Validation Recall: {self.model_metrics['val_recall']:.3f}")
        
        return self.model_metrics
    
    def predict_hr_probability(self, player_features, ballpark=None):
        """
        Predict HR probability for a player
        
        Args:
            player_features: Dict or Series with player's current stats
            ballpark: Optional ballpark name for park factor adjustment
            
        Returns:
            Dict with HR probability and analysis
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features in correct order
        feature_values = []
        for feature in self.feature_names:
            if feature in player_features:
                feature_values.append(player_features[feature])
            else:
                # Use below-average defaults for missing power features
                if 'Plus' in feature or 'Score' in feature:
                    feature_values.append(90)  # Below average
                else:
                    feature_values.append(0.0)  # Assuming scaled features
        
        # Make base prediction
        feature_array = np.array(feature_values).reshape(1, -1)
        base_hr_probability = self.model.predict_proba(feature_array)[0, 1]
        
        # Apply ballpark factor if provided
        ballpark_factor = self.ballpark_factors.get(ballpark, 1.0)
        adjusted_hr_probability = min(base_hr_probability * ballpark_factor, 0.95)
        
        prediction_confidence = max(adjusted_hr_probability, 1 - adjusted_hr_probability)
        
        # Analyze power profile
        power_analysis = self._analyze_power_profile(player_features)
        
        # Get feature contributions
        feature_contributions = {}
        if hasattr(self.model, 'feature_importances_'):
            for i, feature in enumerate(self.feature_names):
                feature_contributions[feature] = {
                    'value': feature_values[i],
                    'importance': self.feature_importance[feature],
                    'contribution': feature_values[i] * self.feature_importance[feature]
                }
        
        return {
            'hr_probability': adjusted_hr_probability,
            'base_hr_probability': base_hr_probability,
            'ballpark_factor': ballpark_factor,
            'ballpark': ballpark,
            'confidence': prediction_confidence,
            'power_tier': power_analysis['tier'],
            'power_strengths': power_analysis['strengths'],
            'feature_contributions': feature_contributions,
            'model_version': 'hr_predictor_v1.0',
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def predict_team_hrs(self, team_roster_df, ballpark=None):
        """
        Predict HR probabilities for an entire team roster
        
        Returns top 3 players most likely to hit a home run
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        
        for idx, player in team_roster_df.iterrows():
            try:
                prediction = self.predict_hr_probability(player, ballpark)
                predictions.append({
                    'player_name': player.get('Name', f'Player_{idx}'),
                    'hr_probability': prediction['hr_probability'],
                    'confidence': prediction['confidence'],
                    'power_tier': prediction['power_tier'],
                    'power_strengths': prediction['power_strengths'],
                    'ballpark_boost': f"{((prediction['ballpark_factor'] - 1) * 100):+.1f}%" if prediction['ballpark_factor'] != 1.0 else "Neutral"
                })
            except Exception as e:
                print(f"Error predicting HR for player {idx}: {e}")
                continue
        
        # Sort by HR probability and return top 3
        predictions_df = pd.DataFrame(predictions)
        top_3_power_hitters = predictions_df.nlargest(3, 'hr_probability')
        
        return top_3_power_hitters
    
    def _analyze_power_profile(self, player_features):
        """
        Analyze a player's power profile and categorize their strengths
        """
        # Get key power metrics
        iso = player_features.get('ISO', 0.150)
        barrel_pct = player_features.get('Barrel%', 8.0)
        exit_velo = player_features.get('avg_exit_velocity', 88.0)
        hard_hit_pct = player_features.get('HardHit%', 35.0)
        
        # Determine power tier
        if iso > 0.250 and barrel_pct > 12:
            tier = "Elite Power (Top 10%)"
        elif iso > 0.200 and barrel_pct > 10:
            tier = "Above Average Power (Top 25%)"
        elif iso > 0.150 and barrel_pct > 8:
            tier = "Average Power"
        elif iso > 0.100:
            tier = "Below Average Power"
        else:
            tier = "Limited Power"
        
        # Identify strengths
        strengths = []
        if barrel_pct > 12:
            strengths.append("Excellent barrel rate")
        elif barrel_pct > 10:
            strengths.append("Good barrel rate")
        
        if exit_velo > 92:
            strengths.append("Elite exit velocity")
        elif exit_velo > 90:
            strengths.append("Above average exit velocity")
        
        if hard_hit_pct > 45:
            strengths.append("Consistent hard contact")
        elif hard_hit_pct > 40:
            strengths.append("Good hard contact rate")
        
        if iso > 0.250:
            strengths.append("Elite isolated power")
        elif iso > 0.200:
            strengths.append("Strong isolated power")
        
        if not strengths:
            strengths = ["Developing power skills"]
        
        return {
            'tier': tier,
            'strengths': strengths[:3]  # Top 3 strengths
        }
    
    def get_model_explanation(self):
        """
        Provide detailed explanation of how the HR prediction model works
        """
        explanation = {
            'model_type': 'XGBoost Classifier (Optimized for Class Imbalance)',
            'target': 'Probability of hitting a home run (binary: above/below league average HR rate)',
            'key_features': {
                'Barrel%': 'Rate of optimal launch conditions - strongest HR predictor',
                'ISO': 'Isolated power - measures raw power ability',
                'avg_exit_velocity': 'Exit velocity - physics component of HRs',
                'avg_launch_angle': 'Launch angle - optimal range is 25-35 degrees',
                'Power_Score': 'Composite power metric combining multiple factors',
                'HardHit%': 'Hard contact consistency'
            },
            'hr_physics': {
                'optimal_exit_velocity': '95+ mph',
                'optimal_launch_angle': '25-35 degrees',
                'barrel_hr_rate': '~50% of barrels become HRs',
                'league_hr_rate': '3.4% of plate appearances'
            },
            'model_logic': [
                '1. Barrel rate is the strongest predictor (combines exit velo + launch angle)',
                '2. ISO measures raw power independent of contact ability',
                '3. Exit velocity must exceed ~95 mph for HR probability',
                '4. Launch angle optimization separates good hitters from HR hitters',
                '5. Ballpark factors provide significant adjustments',
                '6. Class imbalance handling prevents model from ignoring rare HR events'
            ],
            'feature_importance': dict(sorted(
                self.feature_importance.items() if self.feature_importance else {},
                key=lambda x: x[1],
                reverse=True
            )),
            'model_metrics': self.model_metrics,
            'ballpark_factors': self.ballpark_factors
        }
        
        return explanation
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'model_metrics': self.model_metrics,
                'model_params': self.xgb_params,
                'ballpark_factors': self.ballpark_factors
            }
            joblib.dump(model_data, filepath)
            print(f"HR prediction model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.model_metrics = model_data['model_metrics']
        self.ballpark_factors = model_data.get('ballpark_factors', self.ballpark_factors)
        print(f"HR prediction model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    predictor = HomeRunPredictor()
    
    print("=== HOME RUN PREDICTION MODEL OVERVIEW ===")
    print("This model predicts HR probability using:")
    print("1. Barrel% - Optimal launch conditions (most important)")
    print("2. ISO - Raw power ability independent of contact")
    print("3. Exit Velocity - Must be 95+ mph for HR potential")
    print("4. Launch Angle - 25-35° is optimal HR range")
    print("5. Hard Hit% - Consistency of solid contact")
    print("6. Ballpark Factors - Venue-specific adjustments")
    
    print("\n=== HR PHYSICS MODEL ===")
    print("Home runs require specific physics:")
    print("- Exit Velocity: 95+ mph (higher = better)")
    print("- Launch Angle: 25-35° optimal (too low = ground out, too high = fly out)")
    print("- Barrels: Combination of both (~50% become HRs)")
    print("- League HR Rate: ~3.4% of plate appearances")
    
    print("\n=== POWER TIERS ===")
    print("Elite Power: ISO > 0.250, Barrel% > 12%")
    print("Above Average: ISO > 0.200, Barrel% > 10%") 
    print("Average Power: ISO > 0.150, Barrel% > 8%")
    print("Below Average: ISO > 0.100")
    print("Limited Power: ISO < 0.100")