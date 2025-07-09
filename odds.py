def get_mock_odds(players):
    return {player: 2.1 for player in players}  # decimal odds

def compare_odds(predictions, odds):
    for player, prob in predictions.items():
        implied = 1 / odds[player]
        edge = prob - implied
        if edge > 0.05:
            print(f"BET: {player} - Model: {prob:.2%}, Implied: {implied:.2%}, Edge: {edge:.2%}")
        else:
            print(f"PASS: {player} - Edge too small ({edge:.2%})")
