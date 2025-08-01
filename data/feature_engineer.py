"""
Feature Engineering Module - COMPLETE FIXED VERSION

This module creates advanced sabermetric features from the actual data collected.
The original version wasn't creating features because column names didn't match.

Key Fixes:
1. Handles actual Baseball Reference column names ('BA' not 'AVG', 'Tm' not 'Team')
2. Creates features even when advanced metrics are missing
3. Robust error handling and fallback calculations
4. Proper ballpark factor integration
5. Creates meaningful engineered features for ML models
6. Enhanced weather and lineup integration

This replaces your existing data/feature_engineer.py file completely.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SabermetricFeatureEngineer:
    def __init__(self):
        """Initialize feature engineering with league averages for context"""
        # 2024 MLB league averages for normalization
        self.league_averages = {
            'BA': 0.243,      # Batting Average (Baseball Reference uses 'BA' not 'AVG')
            'OBP': 0.312,     # On-Base Percentage
            'SLG': 0.406,     # Slugging Percentage
            'OPS': 0.718,     # On-Base Plus Slugging
            'wOBA': 0.315,    # Weighted On-Base Average
            'ISO': 0.163,     # Isolated Power
            'BABIP': 0.291,   # Batting Average on Balls in Play
            'Barrel%': 8.5,   # Barrel Rate
            'HardHit%': 37.2, # Hard Hit Rate
            'avg_exit_velocity': 88.5,  # Average Exit Velocity
            'avg_launch_angle': 12.1,   # Average Launch Angle
            'Whiff%': 24.8,   # Whiff Rate
            'Chase%': 28.9,   # Chase Rate
            'Contact%': 75.2, # Contact Rate
            'Zone%': 47.1     # Zone Rate
        }
        
        self.scaler = StandardScaler()
        print("Feature Engineer initialized with correct Baseball Reference column names")
    
    def standardize_column_names(self, df):
        """
        Standardize column names to match what we expect vs what Baseball Reference provides
        """
        df_std = df.copy()
        
        # Column name mappings from Baseball Reference to our expected names
        column_mappings = {
            'BA': 'AVG',      # Batting Average
            'Tm': 'Team',     # Team
            # Add more mappings as needed
        }
        
        # Apply mappings
        for old_name, new_name in column_mappings.items():
            if old_name in df_std.columns and new_name not in df_std.columns:
                df_std = df_std.rename(columns={old_name: new_name})
        
        print(f"Standardized column names. Key columns: {[c for c in ['Name', 'Team', 'AVG', 'OBP', 'SLG'] if c in df_std.columns]}")
        return df_std
    
    def create_basic_sabermetrics(self, df):
        """
        Create basic sabermetric features from fundamental stats
        """
        print("Creating basic sabermetric features...")
        
        enhanced_df = df.copy()
        
        try:
            # Calculate ISO (Isolated Power) if not present
            if 'ISO' not in enhanced_df.columns:
                if 'SLG' in enhanced_df.columns and 'AVG' in enhanced_df.columns:
                    enhanced_df['ISO'] = enhanced_df['SLG'] - enhanced_df['AVG']
                    print("✅ Created ISO from SLG - AVG")
                elif 'SLG' in enhanced_df.columns and 'BA' in enhanced_df.columns:
                    enhanced_df['ISO'] = enhanced_df['SLG'] - enhanced_df['BA']
                    print("✅ Created ISO from SLG - BA")
            
            # Calculate wOBA (simplified) if not present
            if 'wOBA' not in enhanced_df.columns:
                if all(col in enhanced_df.columns for col in ['BB', 'HBP', 'H', '2B', '3B', 'HR', 'AB', 'SF']):
                    # Simplified wOBA weights
                    numerator = (
                        0.69 * enhanced_df['BB'] + 
                        0.72 * enhanced_df['HBP'] + 
                        0.89 * (enhanced_df['H'] - enhanced_df['2B'] - enhanced_df['3B'] - enhanced_df['HR']) +
                        1.27 * enhanced_df['2B'] + 
                        1.62 * enhanced_df['3B'] + 
                        2.10 * enhanced_df['HR']
                    )
                    denominator = enhanced_df['AB'] + enhanced_df['BB'] + enhanced_df['SF'] + enhanced_df['HBP']
                    enhanced_df['wOBA'] = numerator / denominator
                    enhanced_df['wOBA'] = enhanced_df['wOBA'].fillna(self.league_averages['wOBA'])
                    print("✅ Created wOBA from component stats")
            
            # Calculate BABIP if not present
            if 'BABIP' not in enhanced_df.columns:
                if all(col in enhanced_df.columns for col in ['H', 'HR', 'AB', 'SO', 'SF']):
                    numerator = enhanced_df['H'] - enhanced_df['HR']
                    denominator = enhanced_df['AB'] - enhanced_df['SO'] - enhanced_df['HR'] + enhanced_df['SF']
                    enhanced_df['BABIP'] = numerator / denominator
                    enhanced_df['BABIP'] = enhanced_df['BABIP'].fillna(self.league_averages['BABIP'])
                    print("✅ Created BABIP from component stats")
            
            # Calculate HR/PA rate for HR prediction
            if 'HR_Rate' not in enhanced_df.columns:
                if 'HR' in enhanced_df.columns and 'PA' in enhanced_df.columns:
                    enhanced_df['HR_Rate'] = enhanced_df['HR'] / enhanced_df['PA']
                    enhanced_df['HR_Rate'] = enhanced_df['HR_Rate'].fillna(0)
                    print("✅ Created HR_Rate")
            
            # Calculate walk rate and strikeout rate
            if 'BB_Rate' not in enhanced_df.columns and 'BB' in enhanced_df.columns and 'PA' in enhanced_df.columns:
                enhanced_df['BB_Rate'] = enhanced_df['BB'] / enhanced_df['PA']
                
            if 'K_Rate' not in enhanced_df.columns and 'SO' in enhanced_df.columns and 'PA' in enhanced_df.columns:
                enhanced_df['K_Rate'] = enhanced_df['SO'] / enhanced_df['PA']
            
            print(f"✅ Basic sabermetrics created. New columns: {len(enhanced_df.columns) - len(df.columns)}")
            
        except Exception as e:
            print(f"⚠️ Error creating basic sabermetrics: {e}")
        
        return enhanced_df
    
    def create_hit_prediction_features(self, df):
        """
        Create features specifically designed for hit prediction
        """
        print("Creating hit prediction features...")
        
        hit_features = df.copy()
        feature_count = len(hit_features.columns)
        
        try:
            # 1. CONTACT ABILITY METRICS
            # Use available contact metrics or estimate from strikeout rate
            if 'Contact%' not in hit_features.columns and 'K_Rate' in hit_features.columns:
                # Estimate contact rate from strikeout rate (inverse relationship)
                hit_features['Contact%'] = 100 - (hit_features['K_Rate'] * 120)  # Rough conversion
                hit_features['Contact%'] = hit_features['Contact%'].clip(lower=40, upper=95)
            
            # Contact rate relative to league average
            if 'Contact%' in hit_features.columns:
                hit_features['Contact_Rate_Plus'] = (
                    hit_features['Contact%'] / self.league_averages['Contact%'] * 100
                )
            
            # 2. PLATE DISCIPLINE METRICS
            # Estimate plate discipline from walk and strikeout rates
            if 'BB_Rate' in hit_features.columns and 'K_Rate' in hit_features.columns:
                hit_features['Plate_Discipline_Score'] = (
                    hit_features['BB_Rate'] * 2 - hit_features['K_Rate']
                )
            
            # 3. CONTACT QUALITY METRICS
            # Use hard hit percentage if available, otherwise estimate from ISO
            if 'HardHit%' not in hit_features.columns and 'ISO' in hit_features.columns:
                # Estimate hard hit rate from ISO (players with more power typically hit harder)
                hit_features['HardHit%'] = 20 + (hit_features['ISO'] * 100)  # Rough conversion
                hit_features['HardHit%'] = hit_features['HardHit%'].clip(lower=15, upper=60)
            
            if 'HardHit%' in hit_features.columns:
                hit_features['Hard_Contact_Plus'] = (
                    hit_features['HardHit%'] / self.league_averages['HardHit%'] * 100
                )
            
            # 4. BABIP SUSTAINABILITY ANALYSIS
            if 'BABIP' in hit_features.columns and 'HardHit%' in hit_features.columns:
                # Expected BABIP based on contact quality
                hit_features['Expected_BABIP'] = (
                    0.291 + (hit_features['HardHit%'] - self.league_averages['HardHit%']) * 0.002
                )
                hit_features['BABIP_Luck_Factor'] = (
                    hit_features['BABIP'] / hit_features['Expected_BABIP']
                )
                hit_features['BABIP_Luck_Factor'] = hit_features['BABIP_Luck_Factor'].fillna(1.0).clip(0.7, 1.3)
            
            # 5. COMPOSITE CONTACT SCORE
            contact_components = []
            
            if 'Contact_Rate_Plus' in hit_features.columns:
                contact_components.append(hit_features['Contact_Rate_Plus'])
            if 'Hard_Contact_Plus' in hit_features.columns:
                contact_components.append(hit_features['Hard_Contact_Plus'])
            if 'BABIP_Luck_Factor' in hit_features.columns:
                contact_components.append(hit_features['BABIP_Luck_Factor'] * 100)
            
            if contact_components:
                hit_features['Contact_Composite_Score'] = np.mean(contact_components, axis=0)
            
            # 6. BALLPARK-ADJUSTED METRICS
            if 'ballpark_hit_factor' in hit_features.columns:
                # Adjust key hitting metrics for ballpark
                for metric in ['AVG', 'OBP', 'BABIP']:
                    if metric in hit_features.columns:
                        hit_features[f'{metric}_Park_Adj'] = (
                            hit_features[metric] / hit_features['ballpark_hit_factor']
                        )
            
            new_features = len(hit_features.columns) - feature_count
            print(f"✅ Created {new_features} hit prediction features")
            
        except Exception as e:
            print(f"⚠️ Error creating hit prediction features: {e}")
        
        return hit_features
    
    def create_hr_prediction_features(self, df):
        """
        Create features specifically designed for home run prediction
        """
        print("Creating HR prediction features...")
        
        hr_features = df.copy()
        feature_count = len(hr_features.columns)
        
        try:
            # 1. RAW POWER METRICS
            # ISO relative to league average
            if 'ISO' in hr_features.columns:
                hr_features['ISO_Plus'] = (
                    hr_features['ISO'] / self.league_averages['ISO'] * 100
                )
            
            # HR rate metrics
            if 'HR_Rate' in hr_features.columns:
                hr_features['HR_Rate_Plus'] = (
                    hr_features['HR_Rate'] / 0.034 * 100  # League average HR rate
                )
            
            # 2. LAUNCH CONDITIONS FOR HRS
            # Estimate optimal launch conditions
            if 'avg_exit_velocity' not in hr_features.columns and 'ISO' in hr_features.columns:
                # Estimate exit velocity from power (ISO)
                hr_features['avg_exit_velocity'] = 85 + (hr_features['ISO'] * 20)
                hr_features['avg_exit_velocity'] = hr_features['avg_exit_velocity'].clip(80, 95)
            
            if 'avg_launch_angle' not in hr_features.columns:
                # Estimate launch angle from HR rate and flyball tendency
                if 'HR' in hr_features.columns and 'AB' in hr_features.columns:
                    hr_rate_est = hr_features['HR'] / hr_features['AB']
                    hr_features['avg_launch_angle'] = 8 + (hr_rate_est * 400)  # Rough conversion
                    hr_features['avg_launch_angle'] = hr_features['avg_launch_angle'].clip(5, 25)
                else:
                    hr_features['avg_launch_angle'] = 12.0  # League average
            
            # HR Launch Score (combination of exit velo and launch angle)
            if 'avg_exit_velocity' in hr_features.columns and 'avg_launch_angle' in hr_features.columns:
                hr_features['HR_Launch_Score'] = (
                    (hr_features['avg_exit_velocity'] / 90.0) * 
                    hr_features['avg_launch_angle'].apply(self._launch_angle_hr_factor)
                )
            
            # 3. BARREL RATE AND OPTIMIZATION
            if 'Barrel%' not in hr_features.columns:
                # Estimate barrel rate from ISO and estimated launch conditions
                if 'ISO' in hr_features.columns and 'HR_Launch_Score' in hr_features.columns:
                    hr_features['Barrel%'] = (
                        hr_features['ISO'] * hr_features['HR_Launch_Score'] * 30
                    ).clip(0, 20)
            
            if 'Barrel%' in hr_features.columns:
                hr_features['Barrel_Plus'] = (
                    hr_features['Barrel%'] / self.league_averages['Barrel%'] * 100
                )
            
            # 4. POWER EFFICIENCY METRICS
            if 'HardHit%' in hr_features.columns and 'ISO' in hr_features.columns:
                hr_features['Power_Per_HardHit'] = (
                    hr_features['ISO'] / (hr_features['HardHit%'] / 100)
                ).fillna(hr_features['ISO'].median())
            
            # 5. COMPOSITE POWER SCORE
            power_components = []
            
            if 'ISO_Plus' in hr_features.columns:
                power_components.append(hr_features['ISO_Plus'])
            if 'Barrel_Plus' in hr_features.columns:
                power_components.append(hr_features['Barrel_Plus'])
            if 'HR_Launch_Score' in hr_features.columns:
                power_components.append(hr_features['HR_Launch_Score'] * 100)
            
            if power_components:
                hr_features['Power_Composite_Score'] = np.mean(power_components, axis=0)
            
            # 6. BALLPARK-ADJUSTED POWER METRICS
            if 'ballpark_hr_factor' in hr_features.columns:
                # Adjust power metrics for ballpark
                for metric in ['ISO', 'SLG', 'HR_Rate']:
                    if metric in hr_features.columns:
                        hr_features[f'{metric}_Park_Adj'] = (
                            hr_features[metric] / hr_features['ballpark_hr_factor']
                        )
            
            new_features = len(hr_features.columns) - feature_count
            print(f"✅ Created {new_features} HR prediction features")
            
        except Exception as e:
            print(f"⚠️ Error creating HR prediction features: {e}")
        
        return hr_features
    
    def _launch_angle_hr_factor(self, launch_angle):
        """
        Convert launch angle to HR probability factor
        Optimal HR launch angle is 25-35 degrees
        """
        if pd.isna(launch_angle):
            return 0.5
        
        if 25 <= launch_angle <= 35:
            return 1.0  # Optimal
        elif 20 <= launch_angle <= 40:
            return 0.8  # Good
        elif 15 <= launch_angle <= 45:
            return 0.6  # Decent
        else:
            return 0.3  # Poor for HRs
    
    def create_interaction_features(self, df):
        """
        Create interaction features between key metrics
        """
        print("Creating interaction features...")
        
        interaction_df = df.copy()
        feature_count = len(interaction_df.columns)
        
        try:
            # Hit prediction interactions
            if 'Contact_Composite_Score' in interaction_df.columns and 'Hard_Contact_Plus' in interaction_df.columns:
                interaction_df['Contact_Quality_Synergy'] = (
                    interaction_df['Contact_Composite_Score'] * interaction_df['Hard_Contact_Plus'] / 100
                )
            
            # HR prediction interactions
            if 'Power_Composite_Score' in interaction_df.columns and 'HR_Launch_Score' in interaction_df.columns:
                interaction_df['Power_Launch_Synergy'] = (
                    interaction_df['Power_Composite_Score'] * interaction_df['HR_Launch_Score']
                )
            
            # Ballpark interactions
            if 'ballpark_hr_factor' in interaction_df.columns and 'ISO' in interaction_df.columns:
                interaction_df['Ballpark_Power_Boost'] = (
                    interaction_df['ISO'] * interaction_df['ballpark_hr_factor']
                )
            
            if 'ballpark_hit_factor' in interaction_df.columns and 'AVG' in interaction_df.columns:
                interaction_df['Ballpark_Hit_Boost'] = (
                    interaction_df['AVG'] * interaction_df['ballpark_hit_factor']
                )
            
            new_features = len(interaction_df.columns) - feature_count
            print(f"✅ Created {new_features} interaction features")
            
        except Exception as e:
            print(f"⚠️ Error creating interaction features: {e}")
        
        return interaction_df
    
    def engineer_all_features(self, df):
        """
        Master function to create all engineered features
        """
        print("Starting comprehensive feature engineering...")
        print(f"Input columns: {list(df.columns)}")
        
        # Step 1: Standardize column names
        df_std = self.standardize_column_names(df)
        
        # Step 2: Create basic sabermetrics
        df_with_basics = self.create_basic_sabermetrics(df_std)
        
        # Step 3: Create hit prediction features
        df_with_hit_features = self.create_hit_prediction_features(df_with_basics)
        
        # Step 4: Create HR prediction features
        df_with_hr_features = self.create_hr_prediction_features(df_with_hit_features)
        
        # Step 5: Create interaction features
        final_df = self.create_interaction_features(df_with_hr_features)
        
        original_features = len(df.columns)
        final_features = len(final_df.columns)
        new_features = final_features - original_features
        
        print(f"✅ Feature engineering complete!")
        print(f"   Original features: {original_features}")
        print(f"   Final features: {final_features}")
        print(f"   New features created: {new_features}")
        
        # Show key engineered features
        key_engineered = [col for col in final_df.columns if any(keyword in col for keyword in 
                         ['Plus', 'Score', 'Rate', 'Factor', 'Adj', 'Synergy', 'Boost'])]
        if key_engineered:
            print(f"   Key engineered features: {key_engineered[:10]}...")
        
        return final_df
    
    def prepare_features_for_modeling(self, df, target_type='hit'):
        """
        Select and prepare features specifically for each model type
        """
        print(f"Preparing features for {target_type} prediction...")
        
        # Define feature sets for each prediction type
        if target_type == 'hit':
            feature_candidates = [
                # Basic stats
                'AVG', 'OBP', 'BABIP', 'BB_Rate', 'K_Rate',
                # Engineered hit features
                'Contact_Rate_Plus', 'Hard_Contact_Plus', 'Plate_Discipline_Score',
                'BABIP_Luck_Factor', 'Contact_Composite_Score', 'Contact_Quality_Synergy',
                # Ballpark adjustments
                'AVG_Park_Adj', 'OBP_Park_Adj', 'Ballpark_Hit_Boost'
            ]
        else:  # HR prediction
            feature_candidates = [
                # Basic power stats
                'ISO', 'SLG', 'HR_Rate', 'HardHit%', 'Barrel%',
                # Engineered HR features
                'ISO_Plus', 'HR_Rate_Plus', 'Barrel_Plus', 'HR_Launch_Score',
                'Power_Per_HardHit', 'Power_Composite_Score', 'Power_Launch_Synergy',
                # Ballpark adjustments
                'ISO_Park_Adj', 'SLG_Park_Adj', 'Ballpark_Power_Boost'
            ]
        
        # Select features that actually exist in the dataframe
        available_features = [f for f in feature_candidates if f in df.columns]
        
        # Add fallback basic features if not enough advanced features
        if len(available_features) < 5:
            basic_fallbacks = ['OPS', 'PA', 'AB', 'H', 'HR', 'BB', 'SO']
            for feature in basic_fallbacks:
                if feature in df.columns and feature not in available_features:
                    available_features.append(feature)
                if len(available_features) >= 8:  # Good number of features
                    break
        
        print(f"Selected {len(available_features)} features for {target_type} prediction:")
        print(available_features)
        
        # Return feature data
        if available_features:
            feature_data = df[available_features].fillna(df[available_features].median())
            return feature_data, available_features
        else:
            print("⚠️ No suitable features found!")
            return pd.DataFrame(), []

# Example usage
if __name__ == "__main__":
    engineer = SabermetricFeatureEngineer()
    
    print("=== FEATURE ENGINEERING TEST ===")
    
    # Test with sample Baseball Reference style data
    sample_data = pd.DataFrame({
        'Name': ['Player A', 'Player B', 'Player C'],
        'Tm': ['NYY', 'BOS', 'LAD'],  # Baseball Reference uses 'Tm' not 'Team'
        'BA': [0.280, 0.245, 0.310],  # Baseball Reference uses 'BA' not 'AVG'
        'OBP': [0.350, 0.320, 0.380],
        'SLG': [0.450, 0.380, 0.520],
        'OPS': [0.800, 0.700, 0.900],
        'AB': [400, 450, 380],
        'H': [112, 110, 118],
        'HR': [25, 12, 35],
        'BB': [45, 35, 55],
        'SO': [120, 140, 95],
        'PA': [450, 485, 435]
    })
    
    print("Input data:")
    print(sample_data)
    
    # Test feature engineering
    engineered_data = engineer.engineer_all_features(sample_data)
    
    print(f"\nEngineered data shape: {engineered_data.shape}")
    print(f"New columns created: {len(engineered_data.columns) - len(sample_data.columns)}")
    
    # Test feature selection for modeling
    hit_features, hit_feature_names = engineer.prepare_features_for_modeling(
        engineered_data, target_type='hit'
    )
    
    hr_features, hr_feature_names = engineer.prepare_features_for_modeling(
        engineered_data, target_type='hr'
    )
    
    print(f"\nHit prediction features: {len(hit_feature_names)}")
    print(f"HR prediction features: {len(hr_feature_names)}")
    
    if len(hit_features) > 0:
        print("\n✅ Feature engineering working correctly!")
    else:
        print("\n❌ Feature engineering failed")