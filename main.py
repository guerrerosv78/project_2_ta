import pandas as pd
import matplotlib.pyplot as plt
from data import load_data, preprocess
from backtest import run_single_backtest, calculate_metrics
from optimization import optimize_backtest


def main():
    print("WALK-FORWARD ANALYSIS")
    train_data, test_data = load_data()
    train_data = preprocess(train_data)

    train_window = 2016 * 4  # 4 semanas (1 mes)
    test_window = 2016  # 1 semana

    results = []
    start_idx = 0

    print(f"Iniciando bucle de optimizacion semanal...")

    # 1 mes, step forward : 1 week
    while start_idx + train_window + test_window < len(train_data):
        window_train = train_data.iloc[start_idx: start_idx + train_window]
        window_test = train_data.iloc[start_idx + train_window: start_idx + train_window + test_window]

        # optimizar en el mes de entrenamiento
        print(f"optimizando ventana desde indice {start_idx}...")
        best_params = optimize_backtest(window_train)
        results.append(best_params)

        #avanza una semana
        start_idx += test_window
        if len(results) >= 4: break  # Limitamos para la demo, puedes quitarlo

    print("\nWalk-Forward completado.")
    print(f"Se realizaron {len(results)} optimizaciones semanales.")

    # grafica con r window 24
    full_value, ops = run_single_backtest(train_data)
    ret, dd, calmar = calculate_metrics(full_value)

    print(f"Calmar Ratio Final: {calmar:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(full_value, color='blue')
    plt.title(f"Backtest Final BTC - Calmar: {calmar:.2f}")
    plt.savefig("portfolio_final.png")
    plt.show()


if __name__ == "__main__":
    main()