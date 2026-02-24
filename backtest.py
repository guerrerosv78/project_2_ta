import pandas as pd
import ta
import numpy as np


def run_single_backtest(data: pd.DataFrame) -> list[float]:
    data = data.copy()
    cash = 1_000_000
    COM = 0.125 / 100
    active_long_positions = []
    strategy_value = [cash]
    num_operations = 0

    # ventanas
    rsi_window = 24
    rsi_lower = 14

    # 20% del capital (resultados de los 200 trials que hice al inciio)
    n_shares = (cash * 0.20) / data.Close.iloc[0]
    take_profit = 0.0525
    stop_loss = 0.0254

    rsi_ind = ta.momentum.RSIIndicator(data.Close, window=rsi_window)
    data["rsi"] = rsi_ind.rsi()
    data = data.dropna()

    for i, row in data.iterrows():
        for position in active_long_positions.copy():
            current_val = row.Close * position["shares"]
            if current_val < position["sl"] or current_val > position["tp"]:
                cash += current_val * (1 - COM)
                active_long_positions.remove(position)

        if row.rsi < rsi_lower:
            cost = row.Close * n_shares * (1 + COM)
            if cash > cost:
                cash -= cost
                active_long_positions.append({
                    "shares": n_shares,
                    "sl": row.Close * n_shares * (1 - stop_loss),
                    "tp": row.Close * n_shares * (1 + take_profit)
                })
                num_operations += 1

        long_values = sum([p["shares"] * row.Close for p in active_long_positions])
        strategy_value.append(cash + long_values)

    return strategy_value, num_operations


def objective(data: pd.DataFrame, trial) -> float:
    data = data.copy()
    cash = 1000000
    COM = 0.125 / 100
    active_long_positions = []
    strategy_value = [cash]

    # Sugerencias de Optuna
    rsi_window = trial.suggest_int("rsi_window", 4, 40)
    rsi_lower = trial.suggest_int("rsi_lower", 5, 45)
    take_profit = trial.suggest_float("take_profit", 0.02, 0.15)
    stop_loss = trial.suggest_float("stop_loss", 0.02, 0.15)

    # 20% del capital para n_shares
    n_shares = (cash * 0.20) / data.Close.iloc[0]

    rsi_ind = ta.momentum.RSIIndicator(data.Close, window=rsi_window)
    data["rsi"] = rsi_ind.rsi()
    data = data.dropna()

    for i, row in data.iterrows():
        for position in active_long_positions.copy():
            current_val = row.Close * position["shares"]
            if current_val < position["sl"] or current_val > position["tp"]:
                cash += current_val * (1 - COM)
                active_long_positions.remove(position)

        if row.rsi < rsi_lower:
            cost = row.Close * n_shares * (1 + COM)
            if cash > cost:
                cash -= cost
                active_long_positions.append({
                    "shares": n_shares,
                    "sl": row.Close * n_shares * (1 - stop_loss),
                    "tp": row.Close * n_shares * (1 + take_profit)
                })

        long_values = sum([p["shares"] * row.Close for p in active_long_positions])
        strategy_value.append(cash + long_values)

    # calculo de Calmar Ratio
    vals = pd.Series(strategy_value)
    ret = (vals.iloc[-1] / vals.iloc[0]) - 1
    dd = (vals - vals.cummax()) / vals.cummax()
    max_dd = abs(dd.min())

    return ret / max_dd if max_dd != 0 else 0


def calculate_metrics(strategy_value):
    df_val = pd.DataFrame(strategy_value, columns=['value'])

    # retorno total
    total_return = (df_val['value'].iloc[-1] / df_val['value'].iloc[0]) - 1

    # max drawdown
    df_val['peak'] = df_val['value'].cummax()
    df_val['drawdown'] = (df_val['value'] - df_val['peak']) / df_val['peak']
    max_drawdown = abs(df_val['drawdown'].min())

    # calmar ratio
    calmar_ratio = total_return / max_drawdown if max_drawdown != 0 else 0

    return total_return, max_drawdown, calmar_ratio