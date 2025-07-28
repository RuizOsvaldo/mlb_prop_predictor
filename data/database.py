"""
Database Module for MLB Prediction Tracking

This module handles:
1. Storing daily predictions (hits and HRs)
2. Recording actual game outcomes
3. Calculating prediction accuracy over time
4. Providing analytics on model performance
5. Tracking feature importance evolution

Database Schema:
- predictions: Daily predictions with probabilities
- outcomes: Actual game results
- model_performance: Aggregated accuracy metrics
- feature_tracking: Feature importance over time
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class MLBPredictionDatabase:
    def __init__(self, db_path='mlb_predictions.db'):
        """Initialize database connection and create tables"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        """Create all necessary database tables"""
        
        # Predictions table - stores daily predictions
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date DATE NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT,
                prediction_type TEXT NOT NULL, -- 'hit' or 'hr'
                probability REAL NOT NULL,
                confidence REAL,
                model_version TEXT,
                features_used TEXT, -- JSON string of features
                ballpark TEXT,
                opposing_pitcher TEXT,
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(prediction_date, player_name, prediction_type)
            )
        ''')
        
        # Outcomes table - stores actual game results
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_date DATE NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT,
                hits INTEGER DEFAULT 0,
                at_bats INTEGER DEFAULT 0,
                home_runs INTEGER DEFAULT 0,
                plate_appearances INTEGER DEFAULT 0,
                game_id TEXT,
                ballpark TEXT,
                recorded_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(game_date, player_name, game_id)
            )
        ''')
        
        # Model performance tracking
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_range_start DATE NOT NULL,
                date_range_end DATE NOT NULL,
                prediction_type TEXT NOT NULL,
                total_predictions INTEGER,
                correct_predictions INTEGER,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                auc_score REAL,
                avg_predicted_probability REAL,
                avg_actual_rate REAL,
                calibration_score REAL, -- How well probabilities match actual rates
                calculated_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Feature importance tracking over time
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS feature_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT NOT NULL, -- 'hit' or 'hr'
                feature_name TEXT NOT NULL,
                importance_score REAL,
                model_version TEXT,
                tracking_date DATE NOT NULL,
                sample_size INTEGER,
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Daily team predictions summary
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_team_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date DATE NOT NULL,
                team TEXT NOT NULL,
                top_5_hitters TEXT, -- JSON array of top 5 hit predictions
                top_3_power_hitters TEXT, -- JSON array of top 3 HR predictions
                team_hit_probability_avg REAL,
                team_hr_probability_avg REAL,
                ballpark TEXT,
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(prediction_date, team)
            )
        ''')
        
        self.conn.commit()
        print("Database tables created/verified successfully")
    
    def store_prediction(self, player_name, team, prediction_type, probability, 
                        confidence=None, model_version=None, features_used=None,
                        ballpark=None, opposing_pitcher=None, prediction_date=None):
        """
        Store a single prediction
        
        Args:
            player_name: Name of the player
            team: Player's team
            prediction_type: 'hit' or 'hr'
            probability: Predicted probability (0-1)
            confidence: Model confidence in prediction
            model_version: Version of model used
            features_used: Dict of features used in prediction
            ballpark: Ballpark where game will be played
            opposing_pitcher: Name of opposing pitcher
            prediction_date: Date of prediction (defaults to today)
        """
        if prediction_date is None:
            prediction_date = date.today()
        
        # Convert features to JSON string
        features_json = json.dumps(features_used) if features_used else None
        
        try:
            self.conn.execute('''
                INSERT OR REPLACE INTO predictions 
                (prediction_date, player_name, team, prediction_type, probability, 
                 confidence, model_version, features_used, ballpark, opposing_pitcher)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (prediction_date, player_name, team, prediction_type, probability,
                  confidence, model_version, features_json, ballpark, opposing_pitcher))
            
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error storing prediction: {e}")
    
    def store_team_predictions(self, team, prediction_date, top_hitters, top_power_hitters, 
                              ballpark=None):
        """
        Store daily team predictions summary
        
        Args:
            team: Team name
            prediction_date: Date of predictions
            top_hitters: List of top 5 hit predictions (from hit_predictor)
            top_power_hitters: List of top 3 HR predictions (from hr_predictor)
            ballpark: Home ballpark
        """
        # Calculate team averages
        hit_probs = [p['hit_probability'] for p in top_hitters] if top_hitters else [0]
        hr_probs = [p['hr_probability'] for p in top_power_hitters] if top_power_hitters else [0]
        
        team_hit_avg = np.mean(hit_probs)
        team_hr_avg = np.mean(hr_probs)
        
        # Convert to JSON
        top_hitters_json = json.dumps(top_hitters.to_dict('records') if hasattr(top_hitters, 'to_dict') else top_hitters)
        top_power_json = json.dumps(top_power_hitters.to_dict('records') if hasattr(top_power_hitters, 'to_dict') else top_power_hitters)
        
        try:
            self.conn.execute('''
                INSERT OR REPLACE INTO daily_team_predictions
                (prediction_date, team, top_5_hitters, top_3_power_hitters,
                 team_hit_probability_avg, team_hr_probability_avg, ballpark)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (prediction_date, team, top_hitters_json, top_power_json,
                  team_hit_avg, team_hr_avg, ballpark))
            
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error storing team predictions: {e}")
    
    def store_game_outcome(self, player_name, team, game_date, hits=0, at_bats=0, 
                          home_runs=0, plate_appearances=0, game_id=None, ballpark=None):
        """
        Store actual game outcome for a player
        
        Args:
            player_name: Name of the player
            team: Player's team
            game_date: Date of the game
            hits: Number of hits
            at_bats: Number of at-bats
            home_runs: Number of home runs
            plate_appearances: Number of plate appearances
            game_id: Unique game identifier
            ballpark: Ballpark where game was played
        """
        try:
            self.conn.execute('''
                INSERT OR REPLACE INTO outcomes
                (game_date, player_name, team, hits, at_bats, home_runs, 
                 plate_appearances, game_id, ballpark)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (game_date, player_name, team, hits, at_bats, home_runs,
                  plate_appearances, game_id, ballpark))
            
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error storing game outcome: {e}")
    
    def calculate_prediction_accuracy(self, start_date=None, end_date=None, prediction_type='hit'):
        """
        Calculate prediction accuracy over a date range
        
        Args:
            start_date: Start date for analysis (defaults to 30 days ago)
            end_date: End date for analysis (defaults to today)
            prediction_type: 'hit' or 'hr'
            
        Returns:
            Dict with accuracy metrics
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # Get predictions and outcomes for the date range
        query = '''
            SELECT p.player_name, p.prediction_date, p.probability, p.confidence,
                   o.hits, o.at_bats, o.home_runs, o.plate_appearances
            FROM predictions p
            LEFT JOIN outcomes o ON p.player_name = o.player_name 
                                AND p.prediction_date = o.game_date
            WHERE p.prediction_type = ? 
                AND p.prediction_date BETWEEN ? AND ?
                AND o.at_bats IS NOT NULL
        '''
        
        df = pd.read_sql_query(query, self.conn, params=(prediction_type, start_date, end_date))
        
        if len(df) == 0:
            return {'error': 'No data available for the specified date range'}
        
        # Calculate actual outcomes
        if prediction_type == 'hit':
            df['actual_outcome'] = (df['hits'] > 0).astype(int)
            df['actual_rate'] = df['hits'] / df['at_bats']
        else:  # HR prediction
            df['actual_outcome'] = (df['home_runs'] > 0).astype(int)
            df['actual_rate'] = df['home_runs'] / df['plate_appearances']
        
        # Calculate binary predictions (probability > 0.5)
        df['predicted_outcome'] = (df['probability'] > 0.5).astype(int)
        
        # Calculate metrics
        total_predictions = len(df)
        correct_predictions = sum(df['predicted_outcome'] == df['actual_outcome'])
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Precision and Recall
        true_positives = sum((df['predicted_outcome'] == 1) & (df['actual_outcome'] == 1))
        false_positives = sum((df['predicted_outcome'] == 1) & (df['actual_outcome'] == 0))
        false_negatives = sum((df['predicted_outcome'] == 0) & (df['actual_outcome'] == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Calibration (how well probabilities match actual rates)
        # Group predictions into probability bins and check if actual rate matches predicted rate
        df['prob_bin'] = pd.cut(df['probability'], bins=10, labels=False)
        calibration_data = df.groupby('prob_bin').agg({
            'probability': 'mean',
            'actual_outcome': 'mean'
        }).dropna()
        
        if len(calibration_data) > 0:
            calibration_score = np.mean(np.abs(calibration_data['probability'] - calibration_data['actual_outcome']))
        else:
            calibration_score = None
        
        metrics = {
            'prediction_type': prediction_type,
            'date_range': f"{start_date} to {end_date}",
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'avg_predicted_probability': df['probability'].mean(),
            'avg_actual_rate': df['actual_rate'].mean(),
            'calibration_score': calibration_score,
            'sample_data': df.head(10).to_dict('records')  # Sample for debugging
        }
        
        # Store metrics in database
        self.store_performance_metrics(start_date, end_date, prediction_type, metrics)
        
        return metrics
    
    def store_performance_metrics(self, start_date, end_date, prediction_type, metrics):
        """Store calculated performance metrics in database"""
        try:
            self.conn.execute('''
                INSERT INTO model_performance
                (date_range_start, date_range_end, prediction_type, total_predictions,
                 correct_predictions, accuracy, precision_score, recall_score,
                 avg_predicted_probability, avg_actual_rate, calibration_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (start_date, end_date, prediction_type, metrics['total_predictions'],
                  metrics['correct_predictions'], metrics['accuracy'], metrics['precision'],
                  metrics['recall'], metrics['avg_predicted_probability'],
                  metrics['avg_actual_rate'], metrics['calibration_score']))
            
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error storing performance metrics: {e}")
    
    def get_model_performance_history(self, prediction_type='hit', days=90):
        """
        Get historical model performance over time
        
        Returns performance trends for dashboard display
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        query = '''
            SELECT date_range_start, date_range_end, accuracy, precision_score, 
                   recall_score, calibration_score, total_predictions
            FROM model_performance
            WHERE prediction_type = ? 
                AND date_range_start >= ?
            ORDER BY date_range_start DESC
        '''
        
        df = pd.read_sql_query(query, self.conn, params=(prediction_type, start_date))
        
        return df
    
    def get_top_performing_predictions(self, prediction_type='hit', limit=10):
        """
        Get the most successful predictions to analyze what makes them work
        """
        query = '''
            SELECT p.player_name, p.prediction_date, p.probability, p.confidence,
                   o.hits, o.at_bats, o.home_runs, o.plate_appearances,
                   CASE 
                       WHEN ? = 'hit' AND o.hits > 0 THEN 1
                       WHEN ? = 'hr' AND o.home_runs > 0 THEN 1
                       ELSE 0
                   END as successful_prediction
            FROM predictions p
            JOIN outcomes o ON p.player_name = o.player_name 
                           AND p.prediction_date = o.game_date
            WHERE p.prediction_type = ?
                AND p.probability > 0.7  -- High confidence predictions only
            ORDER BY p.probability DESC, successful_prediction DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, self.conn, params=(prediction_type, prediction_type, prediction_type, limit))
        
        return df
    
    def store_feature_importance(self, model_type, feature_importance_dict, model_version=None, 
                                sample_size=None):
        """
        Store feature importance scores for tracking model evolution
        """
        tracking_date = date.today()
        
        for feature_name, importance_score in feature_importance_dict.items():
            try:
                self.conn.execute('''
                    INSERT INTO feature_tracking
                    (model_type, feature_name, importance_score, model_version, 
                     tracking_date, sample_size)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (model_type, feature_name, importance_score, model_version,
                      tracking_date, sample_size))
            except sqlite3.Error as e:
                print(f"Error storing feature importance: {e}")
        
        self.conn.commit()
    
    def get_recent_predictions(self, days=7):
        """Get recent predictions for dashboard display"""
        start_date = date.today() - timedelta(days=days)
        
        query = '''
            SELECT prediction_date, player_name, team, prediction_type, 
                   probability, confidence, ballpark
            FROM predictions
            WHERE prediction_date >= ?
            ORDER BY prediction_date DESC, probability DESC
        '''
        
        df = pd.read_sql_query(query, self.conn, params=(start_date,))
        return df
    
    def get_prediction_summary_stats(self):
        """Get summary statistics for dashboard"""
        stats = {}
        
        # Total predictions made
        total_predictions = self.conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        stats['total_predictions'] = total_predictions
        
        # Predictions by type
        hit_predictions = self.conn.execute('SELECT COUNT(*) FROM predictions WHERE prediction_type = "hit"').fetchone()[0]
        hr_predictions = self.conn.execute('SELECT COUNT(*) FROM predictions WHERE prediction_type = "hr"').fetchone()[0]
        stats['hit_predictions'] = hit_predictions
        stats['hr_predictions'] = hr_predictions
        
        # Recent accuracy (last 30 days)
        recent_hit_accuracy = self.calculate_prediction_accuracy(prediction_type='hit')
        recent_hr_accuracy = self.calculate_prediction_accuracy(prediction_type='hr')
        
        stats['recent_hit_accuracy'] = recent_hit_accuracy.get('accuracy', 0)
        stats['recent_hr_accuracy'] = recent_hr_accuracy.get('accuracy', 0)
        
        # Games with outcomes recorded
        total_outcomes = self.conn.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0]
        stats['total_outcomes'] = total_outcomes
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

# Example usage and testing
if __name__ == "__main__":
    # Initialize database
    db = MLBPredictionDatabase('test_mlb_predictions.db')
    
    print("=== MLB PREDICTION DATABASE INITIALIZED ===")
    print("Tables created:")
    print("1. predictions - Daily player predictions")
    print("2. outcomes - Actual game results") 
    print("3. model_performance - Accuracy tracking")
    print("4. feature_tracking - Feature importance evolution")
    print("5. daily_team_predictions - Team summaries")
    
    # Test storing a prediction
    db.store_prediction(
        player_name="Aaron Judge",
        team="NYY",
        prediction_type="hr",
        probability=0.85,
        confidence=0.9,
        model_version="hr_predictor_v1.0",
        ballpark="Yankee Stadium"
    )
    
    # Test storing outcome
    db.store_game_outcome(
        player_name="Aaron Judge",
        team="NYY", 
        game_date=date.today(),
        hits=2,
        at_bats=4,
        home_runs=1,
        plate_appearances=4,
        ballpark="Yankee Stadium"
    )
    
    print("\nSample prediction and outcome stored successfully!")
    print("Database ready for production use.")
    
    db.close()