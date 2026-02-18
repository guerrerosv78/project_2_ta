import optuna
from backtest import objective

def optimize_backtest(data):
    study = optuna.create_study(drection='maximize')
    study.optimize(lambda trial: objective(data, trial),
                   n_trials=50)
    return study.best_params
