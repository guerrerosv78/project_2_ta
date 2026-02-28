import pandas as pd
import matplotlib.pyplot as plt
from data import load_data, preprocess
from backtest import run_single_backtest, calculate_metrics
from optimization import optimize_backtest


def main():
    print("--- INICIANDO SISTEMA DE TRADING BTC ---")
    train_raw, _ = load_data()
    train_data = preprocess(train_raw)
    if train_data is None: return

    print("\nPASO 1: Optimizando parámetros (200 trials)...")
    best_params = optimize_backtest(train_data)

    print("\nPASO 2: Generando Resultados y Sensibilidad...")
    variations = [0.8, 1.0, 1.2]
    summary = []

    for v in variations:
        p = best_params.copy()
        p['rsi_window'] = int(best_params['rsi_window'] * v)
        res = run_single_backtest(train_data, p)
        ret, dd, calmar, sharpe = calculate_metrics(res['portfolio'])
        summary.append({
            "Escenario": "-20%" if v < 1 else ("+20%" if v > 1 else "ÓPTIMO"),
            "RSI_Win": p['rsi_window'],
            "Cap_Final": res['portfolio'][-1],
            "Comisiones": res['total_fees'],
            "Retorno_Pct": ret,
            "Calmar": calmar
        })

    opt = summary[1]
    print("\n" + "=" * 55)
    print("DESGLOSE FINANCIERO FINAL")
    print("=" * 55)
    print(f"Capital Inicial:  $1,000,000.00")
    print(f"Capital Final:    ${opt['Cap_Final']:,.2f}")
    print(f"Ganancia Neta:    ${(opt['Cap_Final'] - 1000000):,.2f}")
    print(f"Total Comisiones: ${opt['Comisiones']:,.2f}")
    print(f"RENDIMIENTO TOTAL: {opt['Retorno_Pct']:.2%}")
    print(f"Calmar Ratio:     {opt['Calmar']:.4f}")

    print("\n--- TABLA DE SENSIBILIDAD ---")
    df_sens = pd.DataFrame(summary)
    df_sens['Retorno %'] = df_sens['Retorno_Pct'].map(lambda x: f"{x:.2%}")
    print(df_sens[["Escenario", "RSI_Win", "Retorno %", "Calmar"]])

    plt.figure(figsize=(10, 5))
    plt.plot(run_single_backtest(train_data, best_params)['portfolio'], color='blue')
    plt.title(f"Evolución del Portafolio (Retorno: {opt['Retorno_Pct']:.2%})")
    plt.savefig("portfolio_final.png")
    plt.show()


if __name__ == "__main__":
    main()