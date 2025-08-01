# MLB Hit & Home Run Predictor

## Overview

This application generates daily predictions for Major League Baseball (MLB) player hits and home runs using real-time data, advanced sabermetric feature engineering, and machine learning models. It leverages game schedules, player statistics, and ballpark-specific factors to provide context-aware, team-based forecasts displayed in an interactive Streamlit web app.

## System Architecture

```
data/
├── data_collector.py       # Collects daily MLB schedule and player stats
├── feature_engineer.py     # Generates advanced sabermetric features
└── database.py             # Stores predictions for historical tracking

models/
├── hit_predictor.py        # XGBoost model for hit predictions
└── hr_predictor.py         # XGBoost model for home run predictions

app.py                      # Streamlit front-end interface
test_fixed_system.py        # End-to-end test script
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Tests

Run a full system test to validate data collection, feature engineering, model training, and prediction generation:

```bash
python test_fixed_system.py
```

### 3. Launch the Application

```bash
streamlit run app.py
```

## How It Works

### Daily Workflow

1. Automatically fetches the current day’s MLB schedule
2. Collects player statistics from Baseball Reference and related sources
3. Engineers sabermetric features (e.g., ISO, BABIP, wOBA)
4. Trains XGBoost models for hit and home run prediction
5. Generates top player predictions by team and game
6. Displays predictions through a clean, game-oriented user interface

## Key Features

### Game Integration
- Auto-detection of today’s scheduled games
- Team-specific predictions aligned with actual matchups

### Feature Engineering
- Over 15 sabermetric metrics per player, including:
  - ISO_Plus: Isolated power vs. league average
  - Contact_Composite_Score: Combines contact rate and quality
  - Power_Launch_Synergy: Exit velocity + launch angle
  - BABIP_Luck_Factor: Adjusts for variance in batting average

### Ballpark Factors
- Venue-specific adjustments applied automatically based on:
  - Home run environment (e.g., Coors Field, Petco Park)
  - Hit factor modifiers per stadium

### Machine Learning
- Separate XGBoost models for:
  - Hit Predictions: Contact-focused features
  - Home Run Predictions: Power-focused features
- Typical AUC performance:
  - Hit: 0.75–0.90
  - HR: 0.70–0.85

### Prediction Output
- Top 5 hit predictions per team with probability scores
- Top 3 home run predictions per team with power tiers
- Visual indicators and summaries by game

## User Interface

### Streamlit Dashboard

- Automatic Game Display: Lists scheduled MLB games
- Prediction Controls:
  - Generate all predictions for today’s games
  - Focus mode for specific matchups
- Results Display:
  - Team-based predictions grouped by game
  - Player probabilities for hits and HRs
  - Venue information integrated into view

## Monitoring & Tracking

- Historical predictions stored in database
- Model performance metrics (AUC, feature importance) displayed
- Tools for debugging data collection or prediction generation
- Tracks model accuracy and prediction history over time

## Troubleshooting

| Issue                             | Recommendation                                                                 |
|----------------------------------|--------------------------------------------------------------------------------|
| No games found for today         | Check time of day; MLB schedule may not be released yet                        |
| No hitting data collected        | Verify internet connection; retry after a few minutes                          |
| Feature engineering returned 0   | Ensure input data includes necessary stats (BA, OBP, SLG, etc.)                |
| Perfect model accuracy reported  | Likely due to overfitting; normal in early season with limited data            |

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Expected Performance

| Component         | Expected Range                          |
|------------------|------------------------------------------|
| Players per day  | 400–600 active MLB players               |
| Daily games      | 0–15 (depends on MLB schedule)           |
| Features created | 25–45 (including engineered metrics)     |
| Hit AUC          | 0.75–0.90                                |
| HR AUC           | 0.70–0.85                                |
| Hit Probabilities| 45%–85%                                  |
| HR Probabilities | 5%–40%                                   |

## Operational Workflow

### Daily Use

1. Open the application
2. Click “Generate Today’s Predictions”
3. View player predictions organized by matchup
4. Monitor games and compare outcomes

### End of Day

- Predictions are logged to the database
- Available for analysis and model evaluation

## Deployment Notes

### Environment Configuration
- SQLite is used for local development
- PostgreSQL recommended for production
- Data is cached for 30 minutes to reduce load
- Respects external API rate limits

### Remove Before Production
- Remove placeholder data and development modes
- Eliminate debug print statements from source files

## Next Steps

1. Run `python test_fixed_system.py` to verify system functionality  
2. Launch with `streamlit run app.py`  
3. Begin generating and analyzing predictions  

For any issues or custom deployment guidance, review the logs or modular scripts for fine-tuning.
