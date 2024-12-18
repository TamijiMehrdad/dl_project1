
from pathlib import Path
import copy
import json
from datetime import date, timedelta, datetime
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle


def save_pickle(data, address):
    with open(address, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(address):
    with open(address, 'rb') as handle:
        data = pickle.load(handle)
    return data


def produce_result_pandas(model, model_name, result, test_label, probs, test_symbol, test_date_starting, max_iter):
    today = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M')
    address_model = f"{config['path']['addr_ml_model']}{model_name}_{max_iter}_iter_{today}.pkl"
    address_result = f"{config['path']['addr_ml_model']}{model_name}_result_{max_iter}_iter_{today}.pkl"
    save_pickle(model, address_model)
    result = pd.DataFrame(result, columns=["prediction"])
    result["truth"] = test_label
    result["probs"] = probs
    result["symbol"] = test_symbol
    result["date_starting"] = test_date_starting
    result = result.sort_values(['symbol', 'date_starting'], ascending=[True, True])
    result.index = result["date_starting"]
    result = result[["symbol", "prediction", "truth", "probs"]]
    save_pickle(result, address_result)
    print(f"Results: {address_result}")
    return address_result

def multi_layer_percepton(test_data, test_label, train_data, train_label, max_iter, lr):
    model = MLPClassifier(hidden_layer_sizes=(1024, 1024 ,256),  # Two hidden layers with 64 neurons each
                        max_iter=max_iter,  # Maximum number of iterations
                        alpha=0.0001,  # L2 penalty (regularization term) parameter
                        learning_rate_init=lr,
                        solver='adam',
                        verbose=True,
                        random_state=42,  # Seed for random number generation
                         ).fit(train_data, train_label)
    result = model.predict(test_data)
    probs = model.predict_proba(test_data)
    # score = mlp.score(test_data, test_label)
    return model, result, probs

def mlp_model(train_data, test_data, train_label, test_label, selected_columns, max_iter, lr, model_name, type_label):
    tr_data = train_data[selected_columns]
    ts_data = test_data[selected_columns]
    model, result, probs = multi_layer_percepton(test_data=ts_data,
                               test_label=test_label[type_label],
                               train_data=tr_data,
                               train_label=train_label[type_label],
                               max_iter=max_iter, lr=lr)

    address_result = produce_result_pandas(model, model_name, result, test_label[type_label], probs[:, 1], test_data["symbol"], test_label["date_starting"], max_iter)
    return address_result

def  mlp_buy(address_of_data, folder_addr):

    x_train = load_pickle(f"{folder_addr}x_train")
    y_train = load_pickle(f"{folder_addr}y_train")
    x_test = load_pickle(f"{folder_addr}x_test")
    y_test = load_pickle(f"{folder_addr}y_test")

    dic_symbol_embedding = {}
    for symbol in list(ml_data.keys()):
        embedding = int(ml_data[symbol]["symbol_embedding"].iloc[0])
        dic_symbol_embedding[embedding] = symbol
    lr = 0.0001
    epochs = 50001
    selected_columns = ['count', 'rsi', 'macd', 'tsi'] #FOR BUY
    address_result_mlp = mlp_model(x_train, x_test, y_train, y_test, selected_columns, max_iter=epochs, lr=lr, model_name="mlp", type_label="low")

if __name__ == "__main__":
    mlp_buy(address_of_data, folder_addr)
