import matplotlib.pyplot as plt
import pandas as pd

def plot_portfolio_value(strategy_value):
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_value, label='Valor del Portafolio', color='blue')
    plt.title('Evolución del Portfolio a través del tiempo')
    plt.xlabel('Periodos')
    plt.ylabel('USD')
    plt.legend()
    plt.grid(True)
    plt.savefig('portfolio_value.png')
    plt.show()

def generate_returns_table(strategy_value):
    series = pd.Series(strategy_value)
    retornos = series.pct_change().dropna()
    print("Retornos:")
    print(retornos.describe())