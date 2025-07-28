"""
MLB Data Collection Module

This module collects comprehensive MLB data using pybaseball, focusing on:
1. Traditional stats (AVG, OBP, SLG)
2. Advanced sabermetrics (wOBA, xwOBA, wRC+, ISO)
3. Statcast data (Exit Velocity, Launch Angle, Barrel%, Hard Hit%)
4. Plate discipline (Whiff%, Chase%, Contact%, Zone%)
5. Situational data (vs LHP/RHP, ballpark factors)
"""

import pandas as pd
import pybaseball as pyb
from datetime import datetime, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class MLBDataCollector:
    def __init__(self):
        """Initialize the data collector with current season info"""
        self.current_year = datetime.now().year
        # Enable pybaseball cache for faster subsequent calls
        pyb.cache.enable()
    
    def get_player_hitting_stats(self, start_date=None, end_date=None):
        """
        Collect comprehensive hitting statistics
        
        Features included:
        - Traditional: AVG, OBP, SLG, OPS, ISO
        - Advanced: wOBA, xwOBA, wRC+, BABIP
        - Statcast: Exit Velocity, Launch Angle, Barrel%, Hard Hit%
        - Plate Discipline: Whiff%, Chase%, Contact%, Zone%, Swing%
        """
        print("Collecting hitting statistics...")
        
        if not start_date:
            start_date = f"{self.current_year}-03-01"
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get traditional and advanced stats from FanGraphs
        hitting_stats = pyb.batting_stats(start_date, end_date, qual=10)
        
        # Get Statcast data for exit velocity, launch angle, etc.
        statcast_data = self.get_statcast_hitting_data(start_date, end_date)
        
        # Merge datasets
        combined_stats = self.merge_hitting_data(hitting_stats, statcast_data)
        
        return combined_stats
    
    def get_statcast_hitting_data(self, start_date, end_date):
        """
        Get Statcast data including:
        - Exit Velocity (avg, max, 95th percentile)
        - Launch Angle (avg, sweet spot %)
        - Barrel Rate %
        - Hard Hit Rate % (95+ mph)
        - xwOBA, xSLG, xBA
        """
        print("Collecting Statcast data...")
        
        try:
            # Get Statcast data
            statcast = pyb.statcast_batter_exitvelo_barrels(start_date, end_date, min_bbe=25)
            
            # Get expected stats
            expected_stats = pyb.statcast_batter_expected_stats(start_date, end_date, min_pa=25)
            
            # Merge Statcast datasets
            if len(statcast) > 0 and len(expected_stats) > 0:
                statcast_merged = pd.merge(
                    statcast, expected_stats, 
                    on=['player_id', 'player_name'], 
                    how='outer'
                )
                return statcast_merged
            
        except Exception as e:
            print(f"Error collecting Statcast data: {e}")
            return pd.DataFrame()
    
    def get_plate_discipline_stats(self, start_date, end_date):
        """
        Get plate discipline metrics:
        - Whiff% (swings and misses / total swings)
        - Chase% (swings at pitches outside zone / pitches outside zone)
        - Contact% (contact made / total swings)
        - Zone% (pitches in strike zone / total pitches)
        - Swing% (swings / total pitches)
        """
        print("Collecting plate discipline data...")
        
        try:
            # Get pitch-by-pitch data for discipline metrics
            discipline_stats = pyb.batting_stats_bref(start_date, end_date)
            return discipline_stats
        except Exception as e:
            print(f"Error collecting plate discipline data: {e}")
            return pd.DataFrame()
    
    def get_pitcher_stats(self, start_date=None, end_date=None):
        """
        Get pitcher statistics for matchup analysis:
        - ERA, WHIP, FIP, xFIP
        - K%, BB%, GB%, FB%
        - Stuff+ metrics where available
        """
        print("Collecting pitcher statistics...")
        
        if not start_date:
            start_date = f"{self.current_year}-03-01"
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            pitcher_stats = pyb.pitching_stats(start_date, end_date, qual=10)
            return pitcher_stats
        except Exception as e:
            print(f"Error collecting pitcher data: {e}")
            return pd.DataFrame()
    
    def merge_hitting_data(self, traditional_stats, statcast_data):
        """
        Merge traditional and Statcast hitting data
        """
        if len(statcast_data) == 0:
            return traditional_stats
        
        # Create player mapping (handle name variations)
        merged_data = pd.merge(
            traditional_stats, 
            statcast_data,
            left_on='Name',
            right_on='player_name',
            how='left'
        )
        
        return merged_data
    
    def get_matchup_data(self, batter_name, pitcher_name):
        """
        Get historical batter vs pitcher matchup data
        This is crucial for prediction accuracy
        """
        try:
            # Get head-to-head stats if available
            matchup_data = pyb.statcast(
                start_dt=f"{self.current_year-2}-01-01",
                end_dt=datetime.now().strftime("%Y-%m-%d"),
                player_name=batter_name
            )
            
            # Filter for specific pitcher matchups
            if len(matchup_data) > 0:
                pitcher_matchups = matchup_data[
                    matchup_data['pitcher_name'] == pitcher_name
                ]
                return pitcher_matchups
            
        except Exception as e:
            print(f"Error getting matchup data: {e}")
        
        return pd.DataFrame()
    
    def get_ballpark_factors(self):
        """
        Get ballpark factors that affect hit and HR probability
        - Park Factor for runs, HRs
        - Dimensions, elevation, weather patterns
        """
        # This would typically come from a separate dataset
        ballpark_factors = {
            'Coors Field': {'hr_factor': 1.15, 'hit_factor': 1.08},
            'Fenway Park': {'hr_factor': 1.05, 'hit_factor': 1.02},
            'Yankee Stadium': {'hr_factor': 1.10, 'hit_factor': 1.01},
            # Add more ballparks...
        }
        return ballpark_factors
    
    def calculate_custom_metrics(self, df):
        """
        Calculate custom sabermetric features for prediction:
        
        Hit Prediction Features:
        - Contact-adjusted BABIP
        - Hard contact rate vs league average
        - Zone contact rate
        - Recent form (last 15 games)
        
        HR Prediction Features:
        - Barrel rate + launch angle optimization
        - Power index (ISO * Hard Hit%)
        - Pull rate on fly balls
        - Elevation-adjusted HR rate
        """
        print("Calculating custom sabermetric features...")
        
        if len(df) == 0:
            return df
        
        # Hit prediction features
        if 'BABIP' in df.columns and 'Contact%' in df.columns:
            df['Contact_Adj_BABIP'] = df['BABIP'] * (df['Contact%'] / 100)
        
        # HR prediction features
        if 'ISO' in df.columns and 'HardHit%' in df.columns:
            df['Power_Index'] = df['ISO'] * (df['HardHit%'] / 100)
        
        if 'Barrel%' in df.columns and 'avg_launch_angle' in df.columns:
            # Optimal launch angle for HRs is 25-35 degrees
            df['Launch_Angle_Opt'] = df['avg_launch_angle'].apply(
                lambda x: 1.0 if 25 <= x <= 35 else 0.5 if 15 <= x <= 45 else 0.1
            )
            df['Barrel_LA_Score'] = df['Barrel%'] * df['Launch_Angle_Opt']
        
        return df
    
    def get_recent_form(self, player_name, days=15):
        """
        Get player's recent performance (last 15 games)
        Recent form is crucial for daily predictions
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            recent_data = pyb.statcast(
                start_dt=start_date.strftime("%Y-%m-%d"),
                end_dt=end_date.strftime("%Y-%m-%d"),
                player_name=player_name
            )
            
            if len(recent_data) > 0:
                # Calculate recent metrics
                recent_metrics = {
                    'recent_avg_exit_velo': recent_data['launch_speed'].mean(),
                    'recent_avg_launch_angle': recent_data['launch_angle'].mean(),
                    'recent_barrel_rate': (recent_data['barrel'] == 1).mean() * 100,
                    'recent_hard_hit_rate': (recent_data['launch_speed'] >= 95).mean() * 100
                }
                return recent_metrics
            
        except Exception as e:
            print(f"Error getting recent form for {player_name}: {e}")
        
        return {}

    def collect_all_data(self):
        """
        Master function to collect all necessary data for modeling
        """
        print("Starting comprehensive data collection...")
        
        # Get current season data
        hitting_stats = self.get_player_hitting_stats()
        pitcher_stats = self.get_pitcher_stats()
        
        # Add custom metrics
        hitting_stats = self.calculate_custom_metrics(hitting_stats)
        
        print(f"Collected data for {len(hitting_stats)} batters and {len(pitcher_stats)} pitchers")
        
        return hitting_stats, pitcher_stats

# Example usage
if __name__ == "__main__":
    collector = MLBDataCollector()
    hitting_data, pitching_data = collector.collect_all_data()
    print("\nKey features for Hit Prediction:")
    hit_features = ['AVG', 'OBP', 'BABIP', 'Contact%', 'Zone%', 'avg_exit_velocity', 'Contact_Adj_BABIP']
    print([f for f in hit_features if f in hitting_data.columns])
    
    print("\nKey features for HR Prediction:")
    hr_features = ['ISO', 'SLG', 'Barrel%', 'HardHit%', 'avg_launch_angle', 'Power_Index', 'Barrel_LA_Score']
    print([f for f in hr_features if f in hitting_data.columns])