import pandas as pd
import ta
import numpy as np

def run_single_backtest(data: pd.DataFrame, params: dict) -> dict:
    df = data.copy()
    initial_cash = 1_000_000
    current_cash = initial_cash
    COM = 0.00125  # 0.125%
    active_pos = None
    strategy_value = []
    total_fees = 0
    num_ops = 0

    # Indicadores
    df['rsi'] = ta.momentum.RSIIndicator(df.Close, window=int(params['rsi_window'])).rsi()
    df['ma'] = ta.trend.sma_indicator(df.Close, window=int(params['ma_window']))
    df['macd_hist'] = ta.trend.macd_diff(df.Close)
    df = df.dropna().reset_index(drop=True)

    for i, row in df.iterrows():
        v_ma = 1 if row.close > row.ma else -1
        v_rsi = 1 if row.rsi < params['rsi_lower'] else (-1 if row.rsi > (100 - params['rsi_lower']) else 0)
        v_macd = 1 if row.macd_hist > 0 else -1
        score = v_ma + v_rsi + v_macd

        unrealized_pnl = 0

        if active_pos:
            p, e, s = row.close, active_pos['entry'], active_pos['shares']
            unrealized_pnl = s * (p - e) if active_pos['type'] == 'L' else s * (e - p)
            pnl_pct = unrealized_pnl / (s * e)

            if pnl_pct >= params['tp'] or pnl_pct <= -params['sl'] or (active_pos['type'] == 'L' and score <= -2) or (active_pos['type'] == 'S' and score >= 2):
                fee_salida = (s * p) * COM
                total_fees += fee_salida
                current_cash += unrealized_pnl - fee_salida
                active_pos = None
                num_ops += 1
                unrealized_pnl = 0

        if not active_pos:
            if score >= 2 or score <= -2:
                tipo = 'L' if score >= 2 else 'S'
                n_shares = (current_cash * 0.20) / row.close
                fee_entrada = (n_shares * row.close) * COM
                total_fees += fee_entrada
                current_cash -= fee_entrada
                active_pos = {'type': tipo, 'shares': n_shares, 'entry': row.close}

        strategy_value.append(current_cash + unrealized_pnl)

    return {"portfolio": strategy_value, "total_fees": total_fees, "num_ops": num_ops}

def calculate_metrics(strategy_value):
    df_v = pd.Series(strategy_value)
    if len(df_v) < 2: return 0, 0, 0, 0
    ret_total = (df_v.iloc[-1] / df_v.iloc[0]) - 1
    peak = df_v.cummax()
    dd = (df_v - peak) / peak
    max_dd = abs(dd.min())
    calmar = ret_total / max_dd if max_dd > 0 else 0
    returns = df_v.pct_change().dropna()
    ann = np.sqrt(288 * 365)
    sharpe = (returns.mean() / returns.std()) * ann if returns.std() != 0 else 0
    return ret_total, max_dd, calmar, sharpe