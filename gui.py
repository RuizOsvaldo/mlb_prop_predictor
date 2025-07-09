import streamlit as st
import pandas as pd
from main import load_today_lineup_data, train_and_predict_with_hr_score  # ensure these are imported correctly

def app():
    st.title("MLB Prop Predictor")
    st.markdown("### Today's Top Hit & HR Predictions")
    st.info("Top 3 Hitters and 1 HR Candidate Per Team")

    with st.spinner("Loading data..."):
        df = load_today_lineup_data()
        df = train_and_predict_with_hr_score(df)

    teams = df['team'].dropna().unique()

    for team in sorted(teams):
        team_df = df[df['team'] == team].copy()

        st.subheader(f"🏟️ {team}")
        if team_df.empty:
            st.warning("No data available for this team.")
            continue

        # Top 3 Hitters
        top_hitters = team_df.sort_values(by="hit_prob", ascending=False).head(3)
        st.markdown("**Top 3 Hitters (Hit Probability):**")
        for _, row in top_hitters.iterrows():
            st.markdown(
                f"- {row['player_name']} | Hit Prob: `{row['hit_prob']:.3f}` | "
                f"EV: `{row.get('avg_exit_velocity', 'N/A'):.2f}` | "
                f"LA: `{row.get('avg_launch_angle', 'N/A'):.2f}` | "
                f"Hard Hit%: `{row.get('hard_hit_pct', 0) * 100:.1f}%` | "
                f"xSLG: `{row.get('xSLG', 0):.3f}` | Barrel%: `{row.get('barrel_pct', 0) * 100:.1f}%`"
            )

        # HR Candidate
        top_hr = team_df.sort_values(by="hr_prob", ascending=False).head(1)
        if not top_hr.empty:
            hr_player = top_hr.iloc[0]
            st.markdown("**💣 HR Candidate:**")
            st.markdown(
                f"- {hr_player['player_name']} | HR Prob: `{hr_player['hr_prob']:.3f}` | "
                f"EV: `{hr_player.get('avg_exit_velocity', 'N/A'):.2f}` | "
                f"LA: `{hr_player.get('avg_launch_angle', 'N/A'):.2f}` | "
                f"Barrel%: `{hr_player.get('barrel_pct', 0) * 100:.1f}%`"
            )

if __name__ == "__main__":
    app()
