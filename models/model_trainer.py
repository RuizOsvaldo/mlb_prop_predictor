"""
MLB Hit and HR Predictor Model Trainer

This module trains and validates the hit and home run prediction models using
the existing project architecture. It integrates with:
- data/data_collector.py for MLB data
- data/feature_engineer.py for sabermetric features  
- models/hit_predictor.py and models/hr_predictor.py for ML models
- data/database.py for prediction tracking

Usage:
    python models/model_trainer.py
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, date, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Import project modules
from data.data_collector import MLBDataCollector
from data.feature_engineer import SabermetricFeatureEngineer
from models.hit_predictor import HitPredictor
from models.hr_predictor import HomeRunPredictor
from data.database import MLBPredictionDatabase

class MLBModelTrainer:
    def __init__(self, models_dir='models/', db_path='mlb_predictions.db'):
        """
        Initialize the model trainer with existing project components
        """
        self.models_dir = models_dir
        self.db_path = db_path
        
        # Initialize project components
        self.data_collector = MLBDataCollector()
        self.feature_engineer = SabermetricFeatureEngineer()
        self.hit_predictor = HitPredictor()
        self.hr_predictor = HomeRunPredictor()
        self.database = MLBPredictionDatabase(db_path)
        
        # Training results
        self.training_results = {}
        
        print("MLB Model Trainer initialized")
        print(f"Models will be saved to: {models_dir}")
        print(f"Database: {db_path}")
    
    def collect_training_data(self, start_date=None, end_date=None, min_pa=50):
        """
        Collect comprehensive training data using the project's data collector
        
        Args:
            start_date: Start date for data collection (defaults to start of season)
            end_date: End date for data collection (defaults to yesterday)
            min_pa: Minimum plate appearances to include player
        """
        print("=" * 60)
        print("COLLECTING TRAINING DATA")
        print("=" * 60)
        
        if not start_date:
            start_date = f"{datetime.now().year}-03-01"
        if not end_date:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        print(f"Date range: {start_date} to {end_date}")
        print(f"Minimum PA filter: {min_pa}")
        
        try:
            # Collect hitting and pitching data
            hitting_data, pitching_data = self.data_collector.collect_all_data()
            
            if len(hitting_data) == 0:
                raise ValueError("No hitting data collected")
            
            print(f"✅ Collected data for {len(hitting_data)} players")
            
            # Filter by minimum plate appearances
            if 'PA' in hitting_data.columns:
                hitting_data = hitting_data[hitting_data['PA'] >= min_pa]
                print(f"✅ Filtered to {len(hitting_data)} players with {min_pa}+ PA")
            
            # Engineer all sabermetric features
            print("Engineering sabermetric features...")
            engineered_data = self.feature_engineer.engineer_all_features(hitting_data)
            
            print(f"✅ Feature engineering complete")
            print(f"   Original features: {len(hitting_data.columns)}")
            print(f"   Engineered features: {len(engineered_data.columns)}")
            
            # Create training targets
            engineered_data = self._create_training_targets(engineered_data)
            
            # Store training data info
            self.training_results['data_info'] = {
                'players': len(engineered_data),
                'features': len(engineered_data.columns),
                'date_range': f"{start_date} to {end_date}",
                'collection_time': datetime.now().isoformat()
            }
            
            return engineered_data, pitching_data
            
        except Exception as e:
            print(f"❌ Error collecting training data: {e}")
            raise
    
    def _create_training_targets(self, df):
        """
        Create binary training targets for hit and HR prediction
        """
        print("Creating training targets...")
        
        data_with_targets = df.copy()
        
        # Hit prediction target (above league average)
        league_avg = 0.243  # 2024 MLB batting average
        if 'AVG' in data_with_targets.columns:
            data_with_targets['hit_target'] = (data_with_targets['AVG'] > league_avg).astype(int)
            print(f"✅ Hit targets: {data_with_targets['hit_target'].sum()} above average, {len(data_with_targets) - data_with_targets['hit_target'].sum()} below")
        
        # HR prediction target (above league average HR rate)
        league_hr_rate = 0.034  # ~3.4% of PA result in HRs
        if 'HR' in data_with_targets.columns and 'PA' in data_with_targets.columns:
            data_with_targets['hr_rate'] = data_with_targets['HR'] / data_with_targets['PA']
            data_with_targets['hr_target'] = (data_with_targets['hr_rate'] > league_hr_rate).astype(int)
            print(f"✅ HR targets: {data_with_targets['hr_target'].sum()} above average, {len(data_with_targets) - data_with_targets['hr_target'].sum()} below")
        elif 'ISO' in data_with_targets.columns:
            # Fallback using ISO
            league_iso = 0.163
            data_with_targets['hr_target'] = (data_with_targets['ISO'] > league_iso).astype(int)
            print(f"✅ HR targets (ISO-based): {data_with_targets['hr_target'].sum()} above average, {len(data_with_targets) - data_with_targets['hr_target'].sum()} below")
        
        return data_with_targets
    
    def train_hit_model(self, training_data):
        """
        Train the hit prediction model using the existing HitPredictor class
        """
        print("=" * 60)
        print("TRAINING HIT PREDICTION MODEL")
        print("=" * 60)
        
        if 'hit_target' not in training_data.columns:
            raise ValueError("Hit targets not found in training data")
        
        try:
            # Prepare training data using HitPredictor's method
            X_hit, y_hit, hit_feature_names = self.hit_predictor.prepare_training_data(training_data)
            
            print(f"Hit prediction features: {len(hit_feature_names)}")
            print(f"Training samples: {len(X_hit)}")
            print(f"Class distribution: {y_hit.value_counts().to_dict()}")
            
            # Train the model
            hit_metrics = self.hit_predictor.train_model(X_hit, y_hit, hit_feature_names)
            
            # Store results
            self.training_results['hit_model'] = hit_metrics
            
            # Store feature importance in database
            if self.hit_predictor.feature_importance:
                self.database.store_feature_importance(
                    'hit', 
                    self.hit_predictor.feature_importance, 
                    'hit_predictor_v1.0',
                    len(training_data)
                )
            
            print("✅ Hit prediction model training complete!")
            print(f"   Validation AUC: {hit_metrics['val_auc']:.3f}")
            print(f"   Validation Accuracy: {hit_metrics['val_accuracy']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training hit model: {e}")
            return False
    
    def train_hr_model(self, training_data):
        """
        Train the home run prediction model using the existing HomeRunPredictor class
        """
        print("=" * 60)
        print("TRAINING HOME RUN PREDICTION MODEL")
        print("=" * 60)
        
        if 'hr_target' not in training_data.columns:
            raise ValueError("HR targets not found in training data")
        
        try:
            # Prepare training data using HomeRunPredictor's method
            X_hr, y_hr, hr_feature_names = self.hr_predictor.prepare_training_data(training_data)
            
            print(f"HR prediction features: {len(hr_feature_names)}")
            print(f"Training samples: {len(X_hr)}")
            print(f"Class distribution: {y_hr.value_counts().to_dict()}")
            
            # Train the model
            hr_metrics = self.hr_predictor.train_model(X_hr, y_hr, hr_feature_names)
            
            # Store results
            self.training_results['hr_model'] = hr_metrics
            
            # Store feature importance in database
            if self.hr_predictor.feature_importance:
                self.database.store_feature_importance(
                    'hr',
                    self.hr_predictor.feature_importance,
                    'hr_predictor_v1.0', 
                    len(training_data)
                )
            
            print("✅ HR prediction model training complete!")
            print(f"   Validation AUC: {hr_metrics['val_auc']:.3f}")
            print(f"   Validation Accuracy: {hr_metrics['val_accuracy']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training HR model: {e}")
            return False
    
    def save_models(self):
        """
        Save both trained models using their built-in save methods
        """
        print("=" * 60)
        print("SAVING TRAINED MODELS")
        print("=" * 60)
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        try:
            # Save hit prediction model
            hit_model_path = os.path.join(self.models_dir, 'hit_predictor_model.pkl')
            self.hit_predictor.save_model(hit_model_path)
            print(f"✅ Hit model saved: {hit_model_path}")
            
            # Save HR prediction model
            hr_model_path = os.path.join(self.models_dir, 'hr_predictor_model.pkl')
            self.hr_predictor.save_model(hr_model_path)
            print(f"✅ HR model saved: {hr_model_path}")
            
            # Save training results summary
            results_path = os.path.join(self.models_dir, 'training_results.pkl')
            joblib.dump(self.training_results, results_path)
            print(f"✅ Training results saved: {results_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return False
    
    def validate_models(self, validation_data):
        """
        Validate both models with cross-validation and additional metrics
        """
        print("=" * 60)
        print("VALIDATING MODELS")
        print("=" * 60)
        
        validation_results = {}
        
        try:
            # Validate hit model
            if self.hit_predictor.model is not None:
                print("Validating hit prediction model...")
                
                # Get top 5 hit predictions
                top_hitters = self.hit_predictor.predict_team_hits(validation_data)
                
                if len(top_hitters) > 0:
                    avg_hit_prob = top_hitters['hit_probability'].mean()
                    print(f"✅ Average top-5 hit probability: {avg_hit_prob:.3f}")
                    validation_results['hit_validation'] = {
                        'top_5_avg_probability': avg_hit_prob,
                        'predictions_count': len(top_hitters)
                    }
            
            # Validate HR model
            if self.hr_predictor.model is not None:
                print("Validating HR prediction model...")
                
                # Get top 3 HR predictions
                top_power_hitters = self.hr_predictor.predict_team_hrs(validation_data)
                
                if len(top_power_hitters) > 0:
                    avg_hr_prob = top_power_hitters['hr_probability'].mean()
                    print(f"✅ Average top-3 HR probability: {avg_hr_prob:.3f}")
                    validation_results['hr_validation'] = {
                        'top_3_avg_probability': avg_hr_prob,
                        'predictions_count': len(top_power_hitters)
                    }
            
            self.training_results['validation'] = validation_results
            return validation_results
            
        except Exception as e:
            print(f"❌ Error during validation: {e}")
            return {}
    
    def generate_sample_predictions(self, sample_data, ballpark=None):
        """
        Generate sample predictions to test the models
        """
        print("=" * 60)
        print("GENERATING SAMPLE PREDICTIONS")
        print("=" * 60)
        
        try:
            # Generate hit predictions
            if self.hit_predictor.model is not None:
                print("Generating hit predictions...")
                top_hitters = self.hit_predictor.predict_team_hits(sample_data.head(20))
                
                print("🎯 TOP 5 HIT PREDICTIONS:")
                for i, (_, player) in enumerate(top_hitters.iterrows(), 1):
                    print(f"  {i}. {player['player_name']}: {player['hit_probability']:.1%} "
                          f"(Confidence: {player['confidence']:.1%})")
            
            # Generate HR predictions
            if self.hr_predictor.model is not None:
                print("\nGenerating HR predictions...")
                top_power_hitters = self.hr_predictor.predict_team_hrs(sample_data.head(20), ballpark)
                
                print("💥 TOP 3 HR PREDICTIONS:")
                for i, (_, player) in enumerate(top_power_hitters.iterrows(), 1):
                    print(f"  {i}. {player['player_name']}: {player['hr_probability']:.1%} "
                          f"(Power Tier: {player['power_tier']})")
                    if ballpark:
                        print(f"     Ballpark boost: {player.get('ballpark_boost', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error generating sample predictions: {e}")
    
    def print_training_summary(self):
        """
        Print comprehensive training summary
        """
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        
        if 'data_info' in self.training_results:
            data_info = self.training_results['data_info']
            print(f"📊 Dataset: {data_info['players']} players, {data_info['features']} features")
            print(f"📅 Date range: {data_info['date_range']}")
        
        if 'hit_model' in self.training_results:
            hit_metrics = self.training_results['hit_model']
            print(f"\n🎯 HIT MODEL PERFORMANCE:")
            print(f"   Validation AUC: {hit_metrics['val_auc']:.3f}")
            print(f"   Validation Accuracy: {hit_metrics['val_accuracy']:.3f}")
            print(f"   Validation Precision: {hit_metrics['val_precision']:.3f}")
            print(f"   Validation Recall: {hit_metrics['val_recall']:.3f}")
        
        if 'hr_model' in self.training_results:
            hr_metrics = self.training_results['hr_model']
            print(f"\n💥 HR MODEL PERFORMANCE:")
            print(f"   Validation AUC: {hr_metrics['val_auc']:.3f}")
            print(f"   Validation Accuracy: {hr_metrics['val_accuracy']:.3f}")
            print(f"   Validation Precision: {hr_metrics['val_precision']:.3f}")
            print(f"   Validation Recall: {hr_metrics['val_recall']:.3f}")
            print(f"   HR Rate in Training: {hr_metrics.get('hr_rate_in_training', 'N/A'):.3f}")
        
        if 'validation' in self.training_results:
            validation = self.training_results['validation']
            if 'hit_validation' in validation:
                print(f"\n✅ Hit Validation: Top-5 avg probability = {validation['hit_validation']['top_5_avg_probability']:.3f}")
            if 'hr_validation' in validation:
                print(f"✅ HR Validation: Top-3 avg probability = {validation['hr_validation']['top_3_avg_probability']:.3f}")
        
        print("\n📁 Models and results saved to:", self.models_dir)
        print("🗄️  Prediction tracking database:", self.db_path)

def main():
    """
    Main training pipeline
    """
    print("🔥" * 20)
    print("MLB HIT & HR PREDICTOR - MODEL TRAINING")
    print("🔥" * 20)
    
    # Initialize trainer
    trainer = MLBModelTrainer()
    
    try:
        # Step 1: Collect training data
        training_data, pitching_data = trainer.collect_training_data(min_pa=75)
        
        # Step 2: Train hit prediction model
        hit_success = trainer.train_hit_model(training_data)
        
        # Step 3: Train HR prediction model
        hr_success = trainer.train_hr_model(training_data)
        
        if hit_success and hr_success:
            # Step 4: Validate models
            trainer.validate_models(training_data.sample(min(100, len(training_data))))
            
            # Step 5: Generate sample predictions
            trainer.generate_sample_predictions(
                training_data.sample(min(20, len(training_data))),
                ballpark="Yankee Stadium"
            )
            
            # Step 6: Save models
            trainer.save_models()
            
            # Step 7: Print summary
            trainer.print_training_summary()
            
            print("\n🎉 MODEL TRAINING COMPLETE!")
            print("✅ Both models trained and saved successfully")
            print("✅ Ready for daily predictions via app.py")
            
        else:
            print("\n❌ MODEL TRAINING FAILED")
            print("Check error messages above")
    
    except Exception as e:
        print(f"\n💥 TRAINING PIPELINE ERROR: {e}")
        print("Check your data sources and dependencies")

if __name__ == "__main__":
    main()