import optuna
from backtest import run_single_backtest, calculate_metrics

def optimize_backtest(data):
    def objective(trial):
        params = {
            'rsi_window': trial.suggest_int('rsi_window', 10, 50),
            'rsi_lower': trial.suggest_int('rsi_lower', 15, 35),
            'ma_window': trial.suggest_int('ma_window', 20, 150),
            'tp': trial.suggest_float('tp', 0.02, 0.10),
            'sl': trial.suggest_float('sl', 0.01, 0.05)
        }
        res = run_single_backtest(data, params)
        ret, dd, calmar, sharpe = calculate_metrics(res['portfolio'])
        return calmar

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)
    return study.best_params