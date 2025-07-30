"""
Home Run Prediction Model - FIXED VERSION

Fixed the 'hr_target' error by adding proper error handling and fallback target creation.
Similar fixes to hit_predictor.py but focused on power metrics.
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
            'eval_metric': 'auc',
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 10,
            'min_child_weight': 3,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Ballpark HR factors
        self.ballpark_factors = {
            'Coors Field': 1.15,
            'Great American Ball Park': 1.12,
            'Yankee Stadium': 1.10,
            'Citizens Bank Park': 1.08,
            'Progressive Field': 1.05,
            'Fenway Park': 1.03,
            'Wrigley Field': 1.02,
            'Busch Stadium': 1.00,
            'Kauffman Stadium': 0.95,
            'Marlins Park': 0.92,
            'Tropicana Field': 0.90,
            'Petco Park': 0.88,
            'Comerica Park': 0.85
        }
    
    def prepare_training_data(self, player_stats_df):
        """
        Prepare training data for HR prediction - FIXED VERSION
        """
        print("Preparing training data for HR prediction...")
        print(f"Input data shape: {player_stats_df.shape}")
        print(f"Available columns: {list(player_stats_df.columns)}")
        
        training_data = player_stats_df.copy()
        hr_target_created = False

        # Method 1: Use HR rate if HR and PA/AB available
        if 'HR' in training_data.columns and 'PA' in training_data.columns:
            try:
                training_data['hr_rate'] = training_data['HR'] / training_data['PA']
                league_hr_rate = 0.034
                training_data['hr_target'] = (training_data['hr_rate'] > league_hr_rate).astype(int)
                hr_target_created = True
                print(f"✅ Created hr_target using HR/PA rate (league avg: {league_hr_rate})")
            except Exception as e:
                print(f"⚠️ Failed to create target using HR/PA: {e}")

        # Method 2: Use HR rate with AB if PA not available
        if not hr_target_created and 'HR' in training_data.columns and 'AB' in training_data.columns:
            try:
                training_data['hr_rate'] = training_data['HR'] / training_data['AB']
                league_hr_rate = 0.039
                training_data['hr_target'] = (training_data['hr_rate'] > league_hr_rate).astype(int)
                hr_target_created = True
                print(f"✅ Created hr_target using HR/AB rate (league avg: {league_hr_rate})")
            except Exception as e:
                print(f"⚠️ Failed to create target using HR/AB: {e}")

        # Method 3: Use ISO (Isolated Power)
        if not hr_target_created and 'ISO' in training_data.columns:
            try:
                league_iso = 0.163
                training_data['hr_target'] = (training_data['ISO'] > league_iso).astype(int)
                hr_target_created = True
                print(f"✅ Created hr_target using ISO (league avg: {league_iso})")
            except Exception as e:
                print(f"⚠️ Failed to create target using ISO: {e}")

        # Method 4: Use SLG as power proxy
        if not hr_target_created and 'SLG' in training_data.columns:
            try:
                league_slg = 0.406
                training_data['hr_target'] = (training_data['SLG'] > league_slg).astype(int)
                hr_target_created = True
                print(f"✅ Created hr_target using SLG (league avg: {league_slg})")
            except Exception as e:
                print(f"⚠️ Failed to create target using SLG: {e}")

        # Method 5: Use OPS as last statistical resort
        if not hr_target_created and 'OPS' in training_data.columns:
            try:
                league_ops = 0.718
                training_data['hr_target'] = (training_data['OPS'] > league_ops).astype(int)
                hr_target_created = True
                print(f"✅ Created hr_target using OPS (league avg: {league_ops})")
            except Exception as e:
                print(f"⚠️ Failed to create target using OPS: {e}")

        # Method 6: Create random balanced target as absolute last resort
        if not hr_target_created:
            print("⚠️ Creating random balanced target as fallback")
            np.random.seed(42)
            training_data['hr_target'] = np.random.binomial(1, 0.3, len(training_data))
            hr_target_created = True
            print("⚠️ WARNING: Using random targets - model will not be meaningful!")

        # FEATURE SELECTION
        hr_prediction_features = [
            'ISO', 'SLG', 'Barrel%', 'HardHit%', 'avg_exit_velocity', 'avg_launch_angle',
            'ISO_Plus', 'Barrel_Plus', 'HR_Launch_Score', 'Power_Efficiency',
            'Power_Score', 'Barrel_Power_Score', 'Power_Barrel_Interaction',
            'OPS', 'HR', 'hr_rate'
        ]
        available_features = [f for f in hr_prediction_features if f in training_data.columns]
        if len(available_features) < 3:
            numeric_cols = training_data.select_dtypes(include=[np.number]).columns
            basic_features = [col for col in numeric_cols if col not in ['hr_target'] and not col.startswith('Unnamed')]
            available_features.extend(basic_features[:5])
            available_features = list(set(available_features))

        feature_data = training_data[available_features].copy()
        for col in feature_data.columns:
            if feature_data[col].dtype in ['float64', 'int64']:
                if col in ['ISO', 'SLG', 'OPS']:
                    league_averages = {
                        'ISO': 0.163, 'SLG': 0.406, 'OPS': 0.718
                    }
                    feature_data[col] = feature_data[col].fillna(league_averages.get(col, feature_data[col].median()))
                elif 'Plus' in col or 'Score' in col:
                    feature_data[col] = feature_data[col].fillna(100)
                elif '%' in col:
                    feature_data[col] = feature_data[col].fillna(feature_data[col].median())
                else:
                    feature_data[col] = feature_data[col].fillna(feature_data[col].median())

        print(f"Final feature data shape: {feature_data.shape}")
        print(f"Target distribution: {training_data['hr_target'].value_counts().to_dict()}")

        if 'hr_target' in training_data.columns:
            return feature_data, training_data['hr_target'], available_features
        else:
            print("❌ hr_target column missing after processing!")
            return None
        
    def train_model(self, features, targets, feature_names):
        """
        Train the HR prediction model and store feature importance.
        Also store the feature names for consistent prediction.
        """
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score

        X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X_train, y_train)

        # Store feature names for prediction consistency
        self.feature_names = feature_names

        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)[:, 1]
        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)

        # Store feature importance
        self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        self.model_metrics = {'val_accuracy': accuracy, 'val_auc': auc}

        return self.model_metrics

    def predict_team_hrs(self, data, ballpark=None):
        """
        Predict home runs for a specific team using the trained model.
        Returns a DataFrame with predictions and relevant info.
        """
        if self.model is None:
            print("❌ Model is not trained yet!")
            return pd.DataFrame()

        # Prepare data for prediction
        feature_data = self.prepare_prediction_data(data, ballpark)
        if feature_data is None or feature_data.shape[0] == 0:
            return pd.DataFrame()

        # Make predictions and probabilities
        hr_probs = self.model.predict_proba(feature_data)[:, 1]
        # Optionally, you can set a threshold for HR prediction (e.g., 0.5)
        hr_preds = (hr_probs > 0.5).astype(int)

        # Prepare output DataFrame
        results = data.copy()
        results = results.reset_index(drop=True)
        results['hr_probability'] = hr_probs
        results['hr_prediction'] = hr_preds

        # Add confidence (optional: use probability or other metric)
        results['confidence'] = np.abs(hr_probs - 0.5) * 2

        # Add power tier (optional)
        results['power_tier'] = pd.cut(hr_probs, bins=[-0.01, 0.1, 0.2, 0.3, 1.0], labels=['Low', 'Medium', 'High', 'Elite'])

        # Add power strengths (optional, based on features)
        strengths = []
        for i, row in results.iterrows():
            s = []
            if 'Power_Score' in row and row['Power_Score'] > 110:
                s.append('Power Score')
            if 'Barrel_Plus' in row and row['Barrel_Plus'] > 110:
                s.append('Barrel Rate')
            if 'HR_Launch_Score' in row and row['HR_Launch_Score'] > 1.0:
                s.append('Launch Angle')
            strengths.append(s if s else ['Balanced'])
        results['power_strengths'] = strengths

        # Ballpark boost (if ballpark provided)
        if ballpark and ballpark in self.ballpark_factors:
            boost = self.ballpark_factors[ballpark]
            results['ballpark_boost'] = boost
            results['hr_probability'] = results['hr_probability'] * boost
        else:
            results['ballpark_boost'] = 1.0

        # Return top 3 by HR probability
        results = results.sort_values('hr_probability', ascending=False).head(3)
        return results

    def prepare_prediction_data(self, data, ballpark=None):
        """
        Prepare features for prediction, ensuring columns match those used in training.
        Returns only the feature DataFrame for prediction.
        """
        if not hasattr(self, 'feature_names') or self.feature_names is None:
            raise ValueError("Model feature_names not set. Train the model first.")
        feature_data = data.reindex(columns=self.feature_names, fill_value=0)
        return feature_data
