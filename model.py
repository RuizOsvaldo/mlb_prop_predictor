import numpy as np

def train_and_predict_with_hr_score(df):
    def hit_prob(row):
        try:
            if row['avg_exit_velocity'] is None or row['avg_launch_angle'] is None:
                return 0.1
            base = (row['avg_exit_velocity'] / 120) + (abs(row['avg_launch_angle']) / 45)
            return min(max(base / 2, 0), 1)
        except:
            return 0.1

    def hr_prob(row):
        ev = row.get('avg_exit_velocity', 0)
        la = row.get('avg_launch_angle', 0)
        barrel = row.get('barrel_pct', 0)
        hh = row.get('hard_hit_pct', 0)

        # Missing key data? Assign conservative floor.
        if ev == 0 or la == 0:
            return 0.01

        # Score based on mix of inputs
        score = 0
        if la > 10:
            score += 0.1
        if 20 < la < 35:
            score += 0.2
        if ev > 90:
            score += 0.2
        if barrel > 5:
            score += 0.2
        if hh > 45:
            score += 0.2

        print(f"{row['player_name']} | EV: {ev} | LA: {la} | Barrel%: {barrel} | HH%: {hh} | HR Prob: {hr_prob(row)}")
        
        return round(min(0.01 + score, 0.99), 3)


    def hr_score(row):
        score = 0
        score += (1 if (row.get('avg_launch_angle') or 0) > 10 else 0.2)
        score += (1 if (row.get('avg_exit_velocity') or 0) > 90 else 0.2)
        score += (1 if (row.get('hard_hit_pct') or 0) > 0.5 else 0.2)
        # score += (1 if (row.get('xSLG') or 0) > 0.47 else 0.2)
        score += (1 if (row.get('barrel_pct') or 0) > 0.12 else 0.2)
        score += (1 if row.get('batting_hand') and row.get('pitching_hand') and row['batting_hand'] != row['pitching_hand'] else 0)

        # Pitcher vulnerabilities
        score += (1 if (row.get('pitcher_la') or 0) > 10 else 0)
        score += (1 if (row.get('pitcher_ev') or 0) > 75 else 0)
        score += (1 if (row.get('pitcher_hard_hit_pct') or 0) > 0.5 else 0)
        score += (1 if (row.get('pitcher_barrel_pct') or 0) > 0.12 else 0)

        # Environmental
        score += row.get('park_factor', 1.0)
        score += row.get('weather_factor', 1.0)

        return round(score, 3)

    df['hit_prob'] = df.apply(hit_prob, axis=1)
    df['hr_prob'] = df.apply(hr_prob, axis=1)
    df['hr_score'] = df.apply(hr_score, axis=1)

    return df