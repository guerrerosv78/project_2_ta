import pandas as pd


def load_data():
    try:
        data_train = pd.read_csv("data/btc_project_train.csv").dropna()
        data_test = pd.read_csv("data/btc_project_test.csv").dropna()
        return data_train, data_test
    except FileNotFoundError:
        print("Error: No se encontraron los archivos en la carpeta 'data'.")
        return None, None


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [col.lower() for col in data.columns]

    data["timeStamp"] = data["timestamp"]
    data["open"] = data["open"]
    data["high"] = data["high"]
    data["low"] = data["low"]
    data["close"] = data["close"]
    data["Close"] = data["close"]

    return data