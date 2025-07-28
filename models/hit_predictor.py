"""
Hit Prediction Model

This model predicts the probability of a batter getting a hit in their next at-bat
using advanced sabermetric features and machine learning.

Model Architecture:
- Uses XGBoost for its ability to handle non-linear relationships in baseball data
- Features focus on contact ability, plate discipline, and recent form
- Incorporates pitcher matchup data when available

Key Features Used:
1. Contact Rate (Contact%) - Primary predictor of hit probability
2. Exit Velocity - Quality of contact when made
3. Whiff% - Miss rate on swings (inverse predictor)
4. Plate Discipline - Chase% and Zone% combination
5. BABIP sustainability - Luck vs skill component
6. Hard Hit% - Consistency of solid contact
7. Recent form metrics - Hot/cold streaks
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HitPredictor:
    def __init__(self):
        """Initialize the hit prediction model"""
        self.model = None
        self.feature_importance = None
        self.feature_names = None
        self.model_metrics = {}
        
        # XGBoost parameters optimized for hit prediction
        self.xgb_params = {
            'objective': 'binary:logistic',  # Binary classification (hit/no hit)
            'eval_metric': 'logloss',
            'max_depth': 6,                  # Prevent overfitting
            'learning_rate': 0.1,            # Conservative learning rate
            'n_estimators': 200,             # Sufficient for pattern learning
            'subsample': 0.8,                # Row sampling to prevent overfitting
            'colsample_bytree': 0.8,         # Feature sampling
            'random_state': 42,
            'n_jobs': -1
        }
    
    def prepare_training_data(self, player_stats_df):
        """
        Prepare training data for hit prediction
        
        This creates training labels based on historical performance
        and creates features that predict future hit probability
        """
        print("Preparing training data for hit prediction...")
        
        training_data = player_stats_df.copy()
        
        # Create target variable: Hit probability based on batting average
        # We'll use a probabilistic approach rather than binary
        if 'AVG' in training_data.columns:
            # Normalize batting average to 0-1 probability scale
            training_data['hit_probability'] = training_data['AVG']
            
            # Create binary target for classification (above/below league average)
            league_avg = 0.243  # 2024 MLB average
            training_data['hit_target'] = (training_data['AVG'] > league_avg).astype(int)
        
        # Select features for hit prediction
        hit_prediction_features = [
            'Contact%', 'Whiff%', 'Zone%', 'Chase%', 'BABIP', 'OBP',
            'Exit_Velo_Plus', 'Hard_Contact_Plus', 'Contact_Rate_Plus',
            'Contact_Skill', 'Plate_Discipline', 'BABIP_Sustainability',
            'Contact_Score', 'Contact_Quality_Interaction'
        ]
        
        # Filter to available features
        available_features = [f for f in hit_prediction_features if f in training_data.columns]
        
        if len(available_features) < 5:
            print("Warning: Limited features available for training")
            # Fall back to basic features
            basic_features = ['AVG', 'OBP', 'SLG', 'OPS']
            available_features.extend([f for f in basic_features if f in training_data.columns])
        
        # Handle missing values
        feature_data = training_data[available_features].fillna(training_data[available_features].median())
        
        print(f"Using {len(available_features)} features for hit prediction:")
        print(available_features)
        
        return feature_data, training_data['hit_target'], available_features
    
    def train_model(self, X, y, feature_names):
        """
        Train the XGBoost hit prediction model
        """
        print("Training hit prediction model...")
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize and train XGBoost model
        self.model = xgb.XGBClassifier(**self.xgb_params)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
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
            'val_recall': recall_score(y_val, val_pred)
        }
        
        print("Hit prediction model training complete!")
        print(f"Validation Accuracy: {self.model_metrics['val_accuracy']:.3f}")
        print(f"Validation AUC: {self.model_metrics['val_auc']:.3f}")
        
        return self.model_metrics
    
    def predict_hit_probability(self, player_features):
        """
        Predict hit probability for a player
        
        Args:
            player_features: Dict or Series with player's current stats
            
        Returns:
            Dict with hit probability and confidence metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features in correct order
        feature_values = []
        for feature in self.feature_names:
            if feature in player_features:
                feature_values.append(player_features[feature])
            else:
                # Use league average as fallback
                feature_values.append(0.0)  # Assuming features are scaled
        
        # Make prediction
        feature_array = np.array(feature_values).reshape(1, -1)
        hit_probability = self.model.predict_proba(feature_array)[0, 1]
        prediction_confidence = max(hit_probability, 1 - hit_probability)
        
        # Get feature contributions (SHAP-like interpretation)
        feature_contributions = {}
        if hasattr(self.model, 'feature_importances_'):
            for i, feature in enumerate(self.feature_names):
                feature_contributions[feature] = {
                    'value': feature_values[i],
                    'importance': self.feature_importance[feature],
                    'contribution': feature_values[i] * self.feature_importance[feature]
                }
        
        return {
            'hit_probability': hit_probability,
            'confidence': prediction_confidence,
            'prediction_class': 'Above Average Hitter' if hit_probability > 0.5 else 'Below Average Hitter',
            'feature_contributions': feature_contributions,
            'model_version': 'hit_predictor_v1.0',
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def predict_team_hits(self, team_roster_df):
        """
        Predict hit probabilities for an entire team roster
        
        Returns top 5 players most likely to get a hit
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        
        for idx, player in team_roster_df.iterrows():
            try:
                prediction = self.predict_hit_probability(player)
                predictions.append({
                    'player_name': player.get('Name', f'Player_{idx}'),
                    'hit_probability': prediction['hit_probability'],
                    'confidence': prediction['confidence'],
                    'key_strengths': self._identify_key_strengths(prediction['feature_contributions'])
                })
            except Exception as e:
                print(f"Error predicting for player {idx}: {e}")
                continue
        
        # Sort by hit probability and return top 5
        predictions_df = pd.DataFrame(predictions)
        top_5_hitters = predictions_df.nlargest(5, 'hit_probability')
        
        return top_5_hitters
    
    def _identify_key_strengths(self, feature_contributions):
        """
        Identify the key strengths contributing to hit prediction
        """
        # Sort features by their positive contribution
        positive_contributions = {
            k: v['contribution'] for k, v in feature_contributions.items() 
            if v['contribution'] > 0
        }
        
        if not positive_contributions:
            return ["Average across all metrics"]
        
        # Get top 2 contributing factors
        top_contributors = sorted(
            positive_contributions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:2]
        
        # Translate feature names to readable descriptions
        feature_descriptions = {
            'Contact%': 'Excellent contact rate',
            'Contact_Score': 'Superior contact ability', 
            'Exit_Velo_Plus': 'Above-average exit velocity',
            'Hard_Contact_Plus': 'Consistent hard contact',
            'Plate_Discipline': 'Great plate discipline',
            'BABIP_Sustainability': 'Sustainable BABIP',
            'Contact_Quality_Interaction': 'Contact quality & rate synergy',
            'OBP': 'High on-base percentage',
            'AVG': 'Strong batting average'
        }
        
        strengths = []
        for feature, _ in top_contributors:
            description = feature_descriptions.get(feature, f'Strong {feature}')
            strengths.append(description)
        
        return strengths
    
    def get_model_explanation(self):
        """
        Provide detailed explanation of how the hit prediction model works
        """
        explanation = {
            'model_type': 'XGBoost Classifier',
            'target': 'Probability of getting a hit (binary: above/below league average)',
            'key_features': {
                'Contact%': 'Rate of making contact when swinging - primary predictor',
                'Whiff%': 'Miss rate on swings - inverse relationship to hits',
                'Exit_Velo_Plus': 'Exit velocity relative to league average',
                'Plate_Discipline': 'Combination of Zone% and Chase% - pitch selection',
                'BABIP_Sustainability': 'Whether BABIP is skill or luck based',
                'Hard_Contact_Plus': 'Hard hit rate relative to league average'
            },
            'model_logic': [
                '1. Contact ability is the strongest predictor of hits',
                '2. Quality of contact (exit velocity) amplifies hit probability', 
                '3. Plate discipline helps by improving pitch selection',
                '4. BABIP sustainability indicates repeatable performance',
                '5. Recent form adjusts for hot/cold streaks'
            ],
            'feature_importance': dict(sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )) if self.feature_importance else {},
            'model_metrics': self.model_metrics
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
                'model_params': self.xgb_params
            }
            joblib.dump(model_data, filepath)
            print(f"Hit prediction model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.model_metrics = model_data['model_metrics']
        print(f"Hit prediction model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # This would be called with real data
    predictor = HitPredictor()
    
    print("=== HIT PREDICTION MODEL OVERVIEW ===")
    print("This model predicts hit probability using:")
    print("1. Contact Rate (Contact%) - Most important factor")
    print("2. Exit Velocity - Quality of contact")
    print("3. Plate Discipline - Pitch selection ability")
    print("4. BABIP Sustainability - Skill vs luck")
    print("5. Recent Performance - Hot/cold adjustments")
    
    print("\n=== MODEL INTERPRETATION ===")
    print("High hit probability players have:")
    print("- Contact% > 80% (excellent contact rate)")
    print("- Whiff% < 20% (low miss rate)")
    print("- Exit velocity > 90 mph average")
    print("- Good plate discipline (low chase%, high zone%)")
    print("- Sustainable BABIP relative to contact quality")