import pandas as pd
from data import load_data, preprocess
from backtest import run_single_backtest, calculate_metrics


def run_sensitivity():
    train, _ = load_data()
    data = preprocess(train)

    # el mejor resultado
    best = {
        'rsi_window': 24,
        'rsi_lower': 14,
        'take_profit': 0.052,
        'stop_loss': 0.025
    }

    # +- 20%
    variations = [0.8, 1.0, 1.2]

    print(f"{'Variación':<15} | {'RSI Window':<12} | {'Calmar Ratio':<12}")
    print("-" * 45)

    for v in variations:
        test_window = int(best['rsi_window'] * v)
        print(f"{int((v - 1) * 100):>+3}% {'':<10} | {test_window:<12} | proceso Manual")


if __name__ == "__main__":
    run_sensitivity()