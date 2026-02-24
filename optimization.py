import optuna
from backtest import objective

def optimize_backtest(data):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(data, trial), n_trials=200)
    return study.best_params