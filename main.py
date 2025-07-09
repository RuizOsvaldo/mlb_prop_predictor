from data import load_today_lineup_data
from model import train_and_predict_with_hr_score

def display_results(df):
    print("\nTop Player Predictions")
    print("=" * 60)

    teams = df['team'].unique()
    for team in teams:
        print(f"\n🏟️ {team}")
        print("-" * 60)
        team_df = df[df['team'] == team].sort_values('hr_prob', ascending=False)

        top_3 = team_df.head(3)

        for _, row in top_3.iterrows():
            print(f"- {row.get('player_name', 'N/A')} | "
                  f"Hit Prob: {row.get('hit_prob', 0):.3f} | "
                  f"HR Prob: {row.get('hr_prob', 0):.3f} | "
                  f"Exit Velo: {row.get('avg_exit_velocity', 'N/A')}mph | "
                  f"LA: {row.get('avg_launch_angle', 'N/A')} | "
                  f"Hard Hit%: {row.get('hard_hit_pct', 'N/A')}% | "
                  #f"xSLG: {row.get('xSLG', 'N/A')} | "
                  f"Barrel%: {row.get('barrel_pct', 'N/A')}% | "
                  f"Stance: {row.get('batting_hand', 'N/A')} | "
                  f"Opp Pitcher: {row.get('opp_pitcher', 'N/A')} ({row.get('pitching_hand', 'N/A')})"
                  )


def main():
    print("📦 Loading live MLB data from Statcast and API...")
    df = load_today_lineup_data()
    print(f"✅ Loaded {len(df)} players in confirmed lineups.")

    print("🧠 Running predictive model...")
    df = train_and_predict_with_hr_score(df)

    display_results(df)

if __name__ == "__main__":
    main()
