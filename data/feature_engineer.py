"""
Feature Engineering Module for MLB Hit and HR Prediction

This module creates advanced sabermetric features specifically designed for:
1. Hit Prediction: Contact-based metrics, BABIP factors, plate discipline
2. HR Prediction: Power metrics, launch conditions, ballpark adjustments

Key Sabermetric Features Explained:
- wOBA: Weighted On-Base Average (weights hits by their run value)
- xwOBA: Expected wOBA based on quality of contact
- ISO: Isolated Power (SLG - AVG, measures extra-base hit ability)
- Barrel%: Rate of "barreled" balls (optimal exit velo + launch angle)
- Hard Hit%: Rate of balls hit 95+ mph
- Whiff%: Miss rate on swings (predictor of contact ability)
- Chase%: Rate of swinging at balls outside strike zone
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
            'AVG': 0.243,
            'OBP': 0.312,
            'SLG': 0.406,
            'wOBA': 0.315,
            'ISO': 0.163,
            'BABIP': 0.291,
            'Barrel%': 8.5,
            'HardHit%': 37.2,
            'avg_exit_velocity': 88.5,
            'avg_launch_angle': 12.1,
            'Whiff%': 24.8,
            'Chase%': 28.9,
            'Contact%': 75.2,
            'Zone%': 47.1
        }
        
        self.scaler = StandardScaler()
    
    def create_hit_prediction_features(self, df):
        """
        Create features specifically designed for hit prediction
        
        Hit Probability Factors:
        1. Contact Ability: How often does the player make contact?
        2. Contact Quality: When they make contact, how hard/well-placed is it?
        3. BABIP Sustainability: Is their average based on luck or skill?
        4. Situational Performance: How do they perform in key situations?
        5. Recent Form: Are they hot or cold?
        """
        print("Engineering hit prediction features...")
        
        hit_features = df.copy()
        
        # 1. CONTACT ABILITY METRICS
        if 'Contact%' in hit_features.columns:
            # Contact rate relative to league average
            hit_features['Contact_Rate_Plus'] = (
                hit_features['Contact%'] / self.league_averages['Contact%'] * 100
            )
        
        if 'Whiff%' in hit_features.columns:
            # Lower whiff rate = better contact ability (inverse relationship)
            hit_features['Contact_Skill'] = (
                (self.league_averages['Whiff%'] - hit_features['Whiff%']) / 
                self.league_averages['Whiff%'] * 100
            )
        
        if 'Zone%' in hit_features.columns and 'Chase%' in hit_features.columns:
            # Plate discipline composite score
            hit_features['Plate_Discipline'] = (
                (hit_features['Zone%'] / self.league_averages['Zone%']) - 
                (hit_features['Chase%'] / self.league_averages['Chase%'])
            )
        
        # 2. CONTACT QUALITY METRICS
        if 'avg_exit_velocity' in hit_features.columns:
            # Exit velocity relative to league average
            hit_features['Exit_Velo_Plus'] = (
                hit_features['avg_exit_velocity'] / self.league_averages['avg_exit_velocity'] * 100
            )
        
        if 'HardHit%' in hit_features.columns:
            # Hard contact rate advantage
            hit_features['Hard_Contact_Plus'] = (
                hit_features['HardHit%'] / self.league_averages['HardHit%'] * 100
            )
        
        # 3. BABIP SUSTAINABILITY
        if 'BABIP' in hit_features.columns and 'HardHit%' in hit_features.columns:
            # Expected BABIP based on contact quality
            hit_features['Expected_BABIP'] = (
                0.291 + (hit_features['HardHit%'] - self.league_averages['HardHit%']) * 0.003
            )
            # BABIP sustainability score
            hit_features['BABIP_Sustainability'] = (
                hit_features['Expected_BABIP'] / hit_features['BABIP']
            ).fillna(1.0)
        
        # 4. COMPOSITE CONTACT SCORE
        contact_components = []
        if 'Contact_Rate_Plus' in hit_features.columns:
            contact_components.append(hit_features['Contact_Rate_Plus'])
        if 'Hard_Contact_Plus' in hit_features.columns:
            contact_components.append(hit_features['Hard_Contact_Plus'])
        if 'Exit_Velo_Plus' in hit_features.columns:
            contact_components.append(hit_features['Exit_Velo_Plus'])
        
        if contact_components:
            hit_features['Contact_Score'] = np.mean(contact_components, axis=0)
        
        print(f"Created {len([c for c in hit_features.columns if c not in df.columns])} new hit prediction features")
        return hit_features
    
    def create_hr_prediction_features(self, df):
        """
        Create features specifically designed for home run prediction
        
        HR Probability Factors:
        1. Raw Power: ISO, SLG, HR rate
        2. Launch Conditions: Exit velocity + launch angle combination
        3. Barrel Rate: Optimal contact for HRs
        4. Pull Power: Direction of power hits
        5. Ballpark Factors: How friendly is the park?
        """
        print("Engineering home run prediction features...")
        
        hr_features = df.copy()
        
        # 1. RAW POWER METRICS
        if 'ISO' in hr_features.columns:
            # Isolated Power relative to league
            hr_features['ISO_Plus'] = (
                hr_features['ISO'] / self.league_averages['ISO'] * 100
            )
        
        if 'SLG' in hr_features.columns and 'AVG' in hr_features.columns:
            # Extra-base hit rate (alternative ISO calculation)
            hr_features['XBH_Rate'] = hr_features['SLG'] - hr_features['AVG']
        
        # 2. LAUNCH CONDITIONS FOR HRS
        if 'avg_exit_velocity' in hr_features.columns and 'avg_launch_angle' in hr_features.columns:
            # Optimal HR launch conditions (95+ mph exit velo, 25-35° launch angle)
            hr_features['HR_Launch_Score'] = (
                (hr_features['avg_exit_velocity'] / 95.0) * 
                hr_features['avg_launch_angle'].apply(self._launch_angle_hr_factor)
            )
        
        # 3. BARREL RATE AND OPTIMIZATION
        if 'Barrel%' in hr_features.columns:
            # Barrel rate relative to league
            hr_features['Barrel_Plus'] = (
                hr_features['Barrel%'] / self.league_averages['Barrel%'] * 100
            )
            
            # Power-speed combination (barrels that become HRs)
            if 'avg_exit_velocity' in hr_features.columns:
                hr_features['Barrel_Power_Score'] = (
                    hr_features['Barrel%'] * (hr_features['avg_exit_velocity'] / 100)
                )
        
        # 4. HARD CONTACT POWER INDEX
        if 'HardHit%' in hr_features.columns and 'ISO' in hr_features.columns:
            # Power efficiency: How much power per hard hit ball
            hr_features['Power_Efficiency'] = (
                hr_features['ISO'] / (hr_features['HardHit%'] / 100)
            ).fillna(0)
        
        # 5. COMPOSITE POWER SCORE
        power_components = []
        if 'ISO_Plus' in hr_features.columns:
            power_components.append(hr_features['ISO_Plus'])
        if 'Barrel_Plus' in hr_features.columns:
            power_components.append(hr_features['Barrel_Plus'])
        if 'HR_Launch_Score' in hr_features.columns:
            power_components.append(hr_features['HR_Launch_Score'] * 100)  # Scale to match others
        
        if power_components:
            hr_features['Power_Score'] = np.mean(power_components, axis=0)
        
        # 6. EXPECTED POWER METRICS
        if 'xSLG' in hr_features.columns and 'SLG' in hr_features.columns:
            # Power over-/under-performance
            hr_features['Power_Luck'] = hr_features['SLG'] - hr_features['xSLG']
        
        print(f"Created {len([c for c in hr_features.columns if c not in df.columns])} new HR prediction features")
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
    
    def add_situational_features(self, df):
        """
        Add situational and contextual features
        """
        print("Adding situational features...")
        
        situational_df = df.copy()
        
        # Recent performance trends (if available in data)
        # This would require game-by-game data
        
        # Handedness advantages (would need pitcher matchup data)
        # This will be added in the matchup analysis
        
        return situational_df
    
    def create_interaction_features(self, df):
        """
        Create interaction features between key metrics
        These capture the synergistic effects of different skills
        """
        print("Creating interaction features...")
        
        interaction_df = df.copy()
        
        # Hit prediction interactions
        if 'Contact_Score' in interaction_df.columns and 'Exit_Velo_Plus' in interaction_df.columns:
            interaction_df['Contact_Quality_Interaction'] = (
                interaction_df['Contact_Score'] * interaction_df['Exit_Velo_Plus'] / 100
            )
        
        # HR prediction interactions
        if 'Power_Score' in interaction_df.columns and 'Barrel_Plus' in interaction_df.columns:
            interaction_df['Power_Barrel_Interaction'] = (
                interaction_df['Power_Score'] * interaction_df['Barrel_Plus'] / 100
            )
        
        # Plate discipline and power combination
        if 'Plate_Discipline' in interaction_df.columns and 'ISO_Plus' in interaction_df.columns:
            interaction_df['Selective_Power'] = (
                interaction_df['Plate_Discipline'] * interaction_df['ISO_Plus'] / 100
            )
        
        return interaction_df
    
    def prepare_features_for_modeling(self, df, target_type='hit'):
        """
        Final feature preparation for modeling
        - Handle missing values
        - Scale features
        - Select relevant features for each target
        """
        print(f"Preparing features for {target_type} prediction modeling...")
        
        model_df = df.copy()
        
        # Define feature sets for each prediction type
        hit_features = [
            'AVG', 'OBP', 'BABIP', 'Contact%', 'Whiff%', 'Zone%', 'Chase%',
            'Contact_Rate_Plus', 'Contact_Skill', 'Plate_Discipline',
            'Exit_Velo_Plus', 'Hard_Contact_Plus', 'BABIP_Sustainability',
            'Contact_Score', 'Contact_Quality_Interaction'
        ]
        
        hr_features = [
            'ISO', 'SLG', 'Barrel%', 'HardHit%', 'avg_exit_velocity', 'avg_launch_angle',
            'ISO_Plus', 'Barrel_Plus', 'HR_Launch_Score', 'Power_Efficiency',
            'Power_Score', 'Barrel_Power_Score', 'Power_Barrel_Interaction'
        ]
        
        # Select features based on target type
        if target_type == 'hit':
            feature_cols = [f for f in hit_features if f in model_df.columns]
        else:  # HR prediction
            feature_cols = [f for f in hr_features if f in model_df.columns]
        
        # Handle missing values
        feature_data = model_df[feature_cols].fillna(model_df[feature_cols].median())
        
        # Scale features for modeling
        scaled_features = pd.DataFrame(
            self.scaler.fit_transform(feature_data),
            columns=feature_cols,
            index=feature_data.index
        )
        
        print(f"Selected {len(feature_cols)} features for {target_type} prediction:")
        print(feature_cols)
        
        return scaled_features, feature_cols
    
    def engineer_all_features(self, df):
        """
        Master function to create all engineered features
        """
        print("Starting comprehensive feature engineering...")
        # Debug: Show unique teams before feature engineering
        if 'Team' in df.columns:
            print("Teams in input df:", sorted(df['Team'].dropna().unique()))
        else:
            print("No 'Team' column in input df")

        # Create hit prediction features
        df_with_hit_features = self.create_hit_prediction_features(df)

        # Create HR prediction features
        df_with_all_features = self.create_hr_prediction_features(df_with_hit_features)

        # Add situational features
        df_with_situational = self.add_situational_features(df_with_all_features)

        # Create interaction features
        final_df = self.create_interaction_features(df_with_situational)

        # Debug: Show unique teams after feature engineering
        if 'Team' in final_df.columns:
            print("Teams in engineered final_df:", sorted(final_df['Team'].dropna().unique()))
        else:
            print("No 'Team' column in engineered final_df")

        print(f"Feature engineering complete. Created {len(final_df.columns) - len(df.columns)} new features.")

        return final_df

# Example usage
if __name__ == "__main__":
    # This would typically be called with real data from data_collector.py
    engineer = SabermetricFeatureEngineer()
    
    # Example feature importance for different prediction types
    print("\n=== HIT PREDICTION FEATURE IMPORTANCE ===")
    print("1. Contact% & Whiff% - Ability to make contact")
    print("2. Exit Velocity - Quality of contact when made")
    print("3. BABIP vs Expected BABIP - Luck vs skill component")
    print("4. Plate Discipline - Pitch selection ability")
    print("5. Hard Hit% - Consistency of solid contact")
    
    print("\n=== HOME RUN PREDICTION FEATURE IMPORTANCE ===")
    print("1. Barrel% - Optimal launch conditions")
    print("2. ISO (Isolated Power) - Raw power ability")
    print("3. Exit Velocity + Launch Angle - Physics of HRs")
    print("4. Power Score - Composite power metric")
    print("5. Hard Hit% in optimal launch angle range")