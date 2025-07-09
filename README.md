# ⚾ MLB Prop Predictor

This tool uses **live MLB Statcast and API data** to predict player prop outcomes, focusing on:

- 💥 Hit Probability
- 🧨 Home Run Probability
- ⚾ Player vs Pitcher Matchups
- 🌡️ Real-time Statcast Metrics (EV, LA, xSLG, Barrel%)

It automatically pulls today’s **confirmed lineups**, applies a formula-based model using real Statcast data, and surfaces top hitters and likely HR candidates **per team**.

### 🔧 Features
- Live data from MLB API and Statcast
- Player-specific launch angle, exit velocity, and more
- Filters only starting lineups for today
- Formulaic HR score and hit probability
- Streamlit interface (coming soon!)

### 🚀 Getting Started
```bash
pip install -r requirements.txt
python main.py
