
from data import load_data, preprocess_data
from optimization import optimize_backtest


#Entrypoint
def main():
    print("Starting...")
    data_train, data_test = load_data()

    data_train = preprocess_data(data_train)
    data_test = preprocess_data(data_test)

    #print(data_train.head())
    #print(data_test.head())

    #strategy_val, num_operations = run_single_backtest()

    best_params = optimize_backtest(data_train)
    print(best_params)

    #print(strategy_val[-1])
    #print(num_operations)

if __name__ == "__main__":
    main()

