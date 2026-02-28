import matplotlib.pyplot as plt
import pandas as pd


def generate_returns_tables(strategy_value, datetimes):
    df = pd.DataFrame({'value': strategy_value})
    # Ajustar longitud de fechas al valor del portafolio
    df['date'] = pd.to_datetime(datetimes.iloc[:len(df)])
    df.set_index('date', inplace=True)

    #  diario
    daily = df['value'].resample('D').last().ffill()
    daily_ret = daily.pct_change()

    # Mensual
    monthly = daily_ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
    print("\n--- RETORNOS MENSUALES ---")
    print(monthly)

    # Trimestral
    quarterly = daily_ret.resample('Q').apply(lambda x: (1 + x).prod() - 1)
    print("\n--- RETORNOS TRIMESTRALES ---")
    print(quarterly)

    # Anual
    annual = daily_ret.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    print("\n--- RETORNOS ANUALES ---")
    print(annual)