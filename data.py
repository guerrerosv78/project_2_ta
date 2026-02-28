import pandas as pd

def load_data():
    try:
        data_train = pd.read_csv("data/btc_project_train.csv")
        data_test = pd.read_csv("data/btc_project_test.csv")
        return data_train, data_test
    except FileNotFoundError:
        print("Error: No se encontraron los archivos en la carpeta 'data/'.")
        return None, None

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    if data is None: return None
    data = data.copy()
    data.columns = [col.lower() for col in data.columns]
    data["datetime"] = pd.to_datetime(data["datetime"], errors='coerce')
    data["Close"] = data["close"] # Para compatibilidad con ta-lib
    return data.dropna(subset=['close', 'datetime']).reset_index(drop=True)