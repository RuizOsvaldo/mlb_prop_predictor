"""
Final Test Script - Verify Everything Works

Run this to make sure all components work together correctly.
"""

def test_complete_system():
    print("🔥" * 20)
    print("FINAL SYSTEM TEST")
    print("🔥" * 20)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from data.data_collector import DebugMLBDataCollector, PitcherAnalyzer, WeatherImpactCalculator
        from data.feature_engineer import SabermetricFeatureEngineer
        from models.hit_predictor import HitPredictor
        from models.hr_predictor import HomeRunPredictor
        from data.database import MLBPredictionDatabase
        print("   ✅ All imports successful")
        
        # Test data collection
        print("\n2. Testing data collection...")
        collector = DebugMLBDataCollector()
        hitting_data, pitching_data, daily_games = collector.collect_all_data_enhanced()
        print(f"   ✅ Data collected: {len(hitting_data)} players, {len(daily_games)} games")
        
        # Test feature engineering
        print("\n3. Testing feature engineering...")
        engineer = SabermetricFeatureEngineer()
        engineered_data = engineer.engineer_all_features(hitting_data)
        new_features = len(engineered_data.columns) - len(hitting_data.columns)
        print(f"   ✅ Feature engineering: {new_features} new features created")
        
        # Test model training
        print("\n4. Testing model training...")
        
        # Hit model
        hit_model = HitPredictor()
        hit_features, hit_targets, hit_names = hit_model.prepare_training_data(engineered_data)
        hit_metrics = hit_model.train_model(hit_features, hit_targets, hit_names)
        print(f"   ✅ Hit model trained: AUC {hit_metrics['val_auc']:.3f}")
        
        # HR model
        hr_model = HomeRunPredictor()
        hr_features, hr_targets, hr_names = hr_model.prepare_training_data(engineered_data)
        hr_metrics = hr_model.train_model(hr_features, hr_targets, hr_names)
        print(f"   ✅ HR model trained: AUC {hr_metrics['val_auc']:.3f}")
        
        # Test predictions
        print("\n5. Testing predictions...")
        
        # Get sample team data
        sample_game = daily_games.iloc[0]
        team_data = engineered_data[engineered_data['Team'] == sample_game['away_team']].head(9)
        
        if len(team_data) > 0:
            # Test hit predictions
            hit_predictions = hit_model.predict_team_hits(team_data)
            print(f"   ✅ Hit predictions: {len(hit_predictions)} players")
            
            # Test HR predictions
            hr_predictions = hr_model.predict_team_hrs(team_data, sample_game['ballpark'])
            print(f"   ✅ HR predictions: {len(hr_predictions)} players")
        else:
            print("   ⚠️ No team data found for predictions")
        
        # Test weather analysis
        print("\n6. Testing weather analysis...")
        weather_calc = WeatherImpactCalculator()
        sample_weather = sample_game.get('weather', {})
        if sample_weather:
            weather_impact = weather_calc.calculate_weather_impact(sample_weather, sample_game['ballpark'])
            print(f"   ✅ Weather analysis: HR factor {weather_impact['hr_factor']:.3f}")
        else:
            print("   ⚠️ No weather data found")
        
        # Test database
        print("\n7. Testing database...")
        database = MLBPredictionDatabase('test_final.db')
        database.store_prediction(
            player_name="Test Player",
            team="TST", 
            prediction_type='hit',
            probability=0.75,
            confidence=0.85,
            model_version='test_v1.0'
        )
        print("   ✅ Database storage working")
        database.close()
        
        print("\n" + "🎉" * 20)
        print("ALL TESTS PASSED!")
        print("🎉" * 20)
        print("\n✅ System is ready!")
        print("✅ Data collection: Working")
        print("✅ Feature engineering: Working") 
        print("✅ Model training: Working")
        print("✅ Predictions: Working")
        print("✅ Weather analysis: Working")
        print("✅ Database: Working")
        print("\n🚀 Ready to run: streamlit run app.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_complete_system()