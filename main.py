import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tools import save_pickle, load_pickle
from pathlib import Path
import os
from machine_learning import produce_result_pandas, result_torch_to_pandas, mlp_model, split_data
# from visualization import vidualize_ml_output

############################################
# Example Dataset
############################################
class StockProfitDataset(Dataset):
    def __init__(self, X, y):
        """
        X: Tensor or numpy array of shape (num_samples, seq_length=60, num_features)
        y: Tensor or numpy array of shape (num_samples,) representing profit percentage
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


############################################
# Positional Encoding for Transformers
############################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_length, d_model)
        seq_length = x.size(1)
        x = x + self.pe[:, :seq_length, :]
        return self.dropout(x)


############################################
# Transformer-based Model for Regression
############################################
# class TransformerRegressor(nn.Module):
#     def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
#         super(TransformerRegressor, self).__init__()
#         # Project input features to d_model
#         self.input_proj = nn.Linear(num_features, d_model)
#
#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(d_model, dropout)
#
#         # Transformer Encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
#                                                    dim_feedforward=dim_feedforward,
#                                                    dropout=dropout, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#         # Final regression layer
#         self.fc = nn.Linear(d_model, 1)
#
#
#     def forward(self, x):
#         # x: (batch_size, seq_length, num_features)
#         x = self.input_proj(x)  # (batch_size, seq_length, d_model)
#         x = self.pos_encoder(x)  # apply positional encoding
#         x = self.transformer_encoder(x)  # (batch_size, seq_length, d_model)
#
#         # Pooling: average over time dimension
#         x = x.mean(dim=1)  # (batch_size, d_model)
#
#         # Final regression output
#         out = self.fc(x)  # (batch_size, 1)
#         return out.squeeze(-1)  # (batch_size,)

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention_weights = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch_size, seq_length, d_model)
        attn_scores = self.attention_weights(x).softmax(dim=1)  # (batch_size, seq_length, 1)
        pooled = torch.sum(x * attn_scores, dim=1)  # Weighted sum: (batch_size, d_model)
        return pooled
class Transformer_Sell(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(Transformer_Sell, self).__init__()
        # Project input features to d_model
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = AttentionPooling(d_model=d_model)


    def forward(self, x):
        x = self.input_proj(x)  # (batch_size, seq_length, d_model)
        x = self.pos_encoder(x)
        # Pass through transformer
        x = self.transformer_encoder(x) #(batch_size, seq_length, d_model).
        # # For simplicity, take the mean pooling of the sequence dimension
        # features = torch.mean(x, dim=1) # (batch_size, d_model)
        pooled_features = self.attention_pool(x)  # Replace torch.mean(x, dim=1)
        return pooled_features


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLPClassifier, self).__init__()

        # Create a list of layers from input_dim -> h1 -> h2 -> h3 -> output
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hdim
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CombinedModel(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_transformer_layers, mlp_hidden_dims, dropout, dim_feedforward):
        super(CombinedModel, self).__init__()
        self.feature_extractor = Transformer_Sell(
            num_features=num_features,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_transformer_layers,
            dropout=dropout
        )
        self.classifier = MLPClassifier(input_dim=d_model, hidden_dims=mlp_hidden_dims, output_dim=1)

    def forward(self, x):
        features = self.feature_extractor(x) # (batch_size, d_model)
        logits = self.classifier(features) # (batch_size, 1)
        return logits

############################################
# Training Loop
############################################
def test_data_evaluate(model, test_loader, criterion, device):
    all_preds = []
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            logits = model(X_test).squeeze(-1)
            loss = criterion(logits, y_test)
            total_test_loss += loss.item() * X_test.size(0)
            total_test_loss += loss.item() * X_test.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.detach().cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader.dataset)

    # Concatenate all predictions into a single NumPy array
    all_preds = np.concatenate(all_preds, axis=0)
    return avg_test_loss, all_preds


def test_model(model, test_loader, criterion, device):
    model.eval()
    total_test_loss = 0
    correct = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            logits = model(X_test).squeeze(-1)
            loss = criterion(logits, y_test)
            total_test_loss += loss.item() * X_test.size(0)

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds * y_test).sum().item()
    avg_test_loss = total_test_loss / len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return avg_test_loss, test_accuracy

def train_model(model, train_loader, test_loader, epochs, start_epoch, lr, device, save_step):
    if epochs==start_epoch:
        print("ALL EPOCHS ARE FINISHED")
        exit()
    criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch).squeeze(-1) # (batch_size,)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_test_loss, test_accuracy = test_model(model, test_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.4f}")
        if (epoch + 1) % save_step == 0:
            model_path = f"{config['path']['addr_ml_model']}Transformer_Sell_epoch_{epoch}_date_{datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M')}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        if epoch+1==epochs:
            avg_test_loss, all_preds =  test_data_evaluate(model, test_loader, criterion, device)
    return avg_test_loss, all_preds

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import psutil
from tools import load_pickle
from machine_learning import prepare_labels_and_data, select_low_oscillation_stocks, prepare_labels_and_data_sequential
from datetime import datetime
import setting
config = setting.config
############################################
# Example Usage
############################################

def load_and_continue_training(num_features, d_model, nhead, num_transformer_layers, mlp_hidden_dims, dropout, dim_feedforward, model_name):
    model = CombinedModel(
        num_features=num_features,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_transformer_layers=num_transformer_layers,
        mlp_hidden_dims=mlp_hidden_dims,
        dropout=dropout
    )

    lst_checkpoints_addr = []
    for filename in os.listdir(config['path']['addr_ml_model']):
        if filename.startswith(f"{model_name}_epoch_"):
            file_path = os.path.join(config['path']['addr_ml_model'], filename)
            lst_checkpoints_addr.append(file_path)
    start_epoch = -1
    if len(lst_checkpoints_addr) != 0:
        for checkpoint_addr in lst_checkpoints_addr:
            temp = int(checkpoint_addr.split("epoch_")[1].split("_")[0])
            if start_epoch < temp:
                start_epoch = temp
                model_path = checkpoint_addr
        model.load_state_dict(torch.load(model_path))
    else:
        pass
    return model, start_epoch+1

# def trandfer_regression():
#     load_from_file = False
#     address_of_data = "/home/hoho/Downloads/ml_data"
#     ml_data = load_pickle(address_of_data)
#     seq_length = 60
#     if load_from_file==True:
#         # address_result_mlp = "/home/hoho/Downloads/mlp_result_2000_iter_2024_12_15_05_23.pkl"
#         # lst_resutl = [[address_result_mlp, "mlp"]]
#         temp = {}
#         temp["AAPL"] = ml_data["AAPL"]
#         temp["ABBV"] = ml_data["ABBV"]
#         ml_data = temp
#         ml_data, labels_dic = prepare_labels_and_data(ml_data, label_address= config["path"]["addr_labels"], simu_annealing_label_address= config["path"]["addr_simulated_annealing_labels"] )
#         test_start = datetime(2023, 6, 1, 0, 0, 0)
#         test_end = datetime(2025, 12, 1, 0, 0, 0)
#         x_train, y_train, x_test, y_test, test_data = prepare_labels_and_data_sequential(ml_data, labels_dic, test_start, test_end, seq_length)
#         save_pickle(x_train, "/home/hoho/OneDrive/stock/Stock/data/sequential/x_train")
#         save_pickle(y_train, "/home/hoho/OneDrive/stock/Stock/data/sequential/y_train")
#         save_pickle(x_test, "/home/hoho/OneDrive/stock/Stock/data/sequential/x_test")
#         save_pickle(y_test, "/home/hoho/OneDrive/stock/Stock/data/sequential/y_test")
#         save_pickle(test_data, "/home/hoho/OneDrive/stock/Stock/data/sequential/test_data")
#         exit()
#     else:
#         x_train = load_pickle("/home/hoho/OneDrive/stock/Stock/data/sequential/x_train")
#         y_train = load_pickle( "/home/hoho/OneDrive/stock/Stock/data/sequential/y_train")
#         x_test = load_pickle( "/home/hoho/OneDrive/stock/Stock/data/sequential/x_test")
#         y_test = load_pickle( "/home/hoho/OneDrive/stock/Stock/data/sequential/y_test")
#         test_data = load_pickle( "/home/hoho/OneDrive/stock/Stock/data/sequential/test_data")
#
#     dic_symbol_embedding = {}
#     for symbol in list(ml_data.keys()):
#         embedding = int(ml_data[symbol]["symbol_embedding"].iloc[0])
#         dic_symbol_embedding[embedding] = symbol
#     # num_days_oscillation = 365  # number of days that we check for oscillation
#     # max_oscillation = 5  # max oscillation that we accept.The lower this number is, the less risk we take.
#     # num_days_up_down = 365 * 5
#     # special_list = list(ml_data.keys())
#     #
#     # lst_symbols, oscillation = select_low_oscillation_stocks(special_list, ml_data, labels_dic, num_days_oscillation=num_days_oscillation, max_oscillation=max_oscillation, num_days_up_down=num_days_up_down)
#
#
#     num_features = x_train.shape[2]  # e.g., price, RSI, TSI, and other indicators
#
#     num_cores = psutil.cpu_count()
#     torch.set_num_threads(num_cores)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # X_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
#     # y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
#     # X_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
#     # y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
#
#     # Create DataLoader
#     batch_size = 32
#     epochs = 1
#     lr = 1e-4
#     train_dataset = StockProfitDataset(x_train, y_train)
#     test_dataset = StockProfitDataset(x_test, y_test)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     # Example of continuing training for 10 more epochs from epoch 100
#     model, start_epoch = load_and_continue_training(num_features)
#     avg_test_loss, all_preds = train_model(model, train_loader, test_loader, epochs=epochs, start_epoch=start_epoch, lr=lr, device=device)
#     print(all_preds)
#     # result = all_preds > 0.5
#     model_name = "TransformerRegressor"
#     address_result = result_torch_to_pandas(model=model, model_name=model_name, result=all_preds, test_label=y_test,
#                                             dic_symbol_embedding=dic_symbol_embedding, test_data=test_data, max_iter=epochs)

def mlp_sell(address_of_data, folder_addr):
    load_from_file = False

    ml_data = load_pickle(address_of_data)
    seq_length = 1
    if load_from_file==False:

        temp = {}
        temp["AAPL"] = ml_data["AAPL"]
        temp["ABBV"] = ml_data["ABBV"]
        ml_data = temp
        ml_data, labels_dic = prepare_labels_and_data(ml_data, label_address= config["path"]["addr_labels"], simu_annealing_label_address= config["path"]["addr_simulated_annealing_labels"] )
        test_start = datetime(2023, 6, 1, 0, 0, 0)
        test_end = datetime(2025, 12, 1, 0, 0, 0)
        x_train, y_train, x_test, y_test, test_data = prepare_labels_and_data_sequential(ml_data, labels_dic, test_start, test_end, seq_length, "sa_high")
        save_pickle(x_train, f"{folder_addr}x_train")
        save_pickle(y_train, f"{folder_addr}y_train")
        save_pickle(x_test, f"{folder_addr}x_test")
        save_pickle(y_test, f"{folder_addr}y_test")
        save_pickle(test_data, f"{folder_addr}test_data")
        exit()
    else:
        x_train = load_pickle(f"{folder_addr}x_train")
        y_train = load_pickle(f"{folder_addr}y_train")
        x_test = load_pickle(f"{folder_addr}x_test")
        y_test = load_pickle(f"{folder_addr}y_test")
        test_data = load_pickle(f"{folder_addr}test_data")

    dic_symbol_embedding = {}
    for symbol in list(ml_data.keys()):
        embedding = int(ml_data[symbol]["symbol_embedding"].iloc[0])
        dic_symbol_embedding[embedding] = symbol
    # num_days_oscillation = 365  # number of days that we check for oscillation
    # max_oscillation = 5  # max oscillation that we accept.The lower this number is, the less risk we take.
    # num_days_up_down = 365 * 5
    # special_list = list(ml_data.keys())
    #
    # lst_symbols, oscillation = select_low_oscillation_stocks(special_list, ml_data, labels_dic, num_days_oscillation=num_days_oscillation, max_oscillation=max_oscillation, num_days_up_down=num_days_up_down)


    num_features = x_train.shape[2]  # e.g., price, RSI, TSI, and other indicators

    # num_cores = psutil.cpu_count()
    # torch.set_num_threads(num_cores)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # X_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    # y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    # X_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    # y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)


    num_cores = psutil.cpu_count()
    torch.set_num_threads(num_cores)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # X_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    # y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    # X_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    # y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Create DataLoader
    batch_size = 32
    epochs = 1
    lr = 1e-4
    num_features = x_train.shape[2]  # e.g., price, RSI, TSI, and other indicators
    d_model = 72 # IMPOERTANTd_model must be divisible by num_heads
    nhead = 4
    num_transformer_layers = 2
    mlp_hidden_dims = [32, 16, 4] #[128, 64, 32]
    dropout = 0.1
    dim_feedforward = 128# 2048
    save_step = 10
    model_name = "Transformer_Sell"

    train_dataset = StockProfitDataset(x_train, y_train)
    test_dataset = StockProfitDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Example of continuing training for 10 more epochs from epoch 100
    model, start_epoch = load_and_continue_training(num_features, d_model, nhead, num_transformer_layers, mlp_hidden_dims, dropout, dim_feedforward, model_name)
    avg_test_loss, all_preds = train_model(model, train_loader, test_loader, epochs=epochs, start_epoch=start_epoch, lr=lr, device=device, save_step=save_step)
    # print(all_preds)
    # result = all_preds > 0.5
    address_result = result_torch_to_pandas(model=model, model_name=model_name, result=all_preds, test_label=y_test,
                                            dic_symbol_embedding=dic_symbol_embedding, test_data=test_data, max_iter=epochs)




def mlp_buy_seq(address_of_data, folder_addr):
    load_from_file = False

    ml_data = load_pickle(address_of_data)
    seq_length = 5
    if load_from_file==False:

        # temp = {}
        # temp["AAPL"] = ml_data["AAPL"]
        # temp["ABBV"] = ml_data["ABBV"]
        # ml_data = temp
        ml_data, labels_dic = prepare_labels_and_data(ml_data, label_address= config["path"]["addr_labels"], simu_annealing_label_address= config["path"]["addr_simulated_annealing_labels"] )
        test_start = datetime(2023, 6, 1, 0, 0, 0)
        test_end = datetime(2025, 12, 1, 0, 0, 0)
        x_train, y_train, x_test, y_test, test_data = prepare_labels_and_data_sequential(ml_data, labels_dic, test_start, test_end, seq_length, "low")
        save_pickle(x_train, f"{folder_addr}x_train")
        save_pickle(y_train, f"{folder_addr}y_train")
        save_pickle(x_test, f"{folder_addr}x_test")
        save_pickle(y_test, f"{folder_addr}y_test")
        save_pickle(test_data, f"{folder_addr}test_data")
        exit()
    else:
        x_train = load_pickle(f"{folder_addr}x_train")
        y_train = load_pickle(f"{folder_addr}y_train")
        x_test = load_pickle(f"{folder_addr}x_test")
        y_test = load_pickle(f"{folder_addr}y_test")
        test_data = load_pickle(f"{folder_addr}test_data")

    dic_symbol_embedding = {}
    for symbol in list(ml_data.keys()):
        embedding = int(ml_data[symbol]["symbol_embedding"].iloc[0])
        dic_symbol_embedding[embedding] = symbol

    num_cores = psutil.cpu_count()
    torch.set_num_threads(num_cores)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create DataLoader
    batch_size = 32
    epochs = 1
    lr = 1e-4
    num_features = x_train.shape[2]  # e.g., price, RSI, TSI, and other indicators
    d_model = 72 # IMPOERTANTd_model must be divisible by num_heads
    nhead = 4
    num_transformer_layers = 2
    mlp_hidden_dims = [32, 16, 4] #[128, 64, 32]
    dropout = 0.1
    dim_feedforward = 128# 2048
    save_step = 10
    model_name = "Transformer_Sell"

    train_dataset = StockProfitDataset(x_train, y_train)
    test_dataset = StockProfitDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Example of continuing training for 10 more epochs from epoch 100
    model, start_epoch = load_and_continue_training(num_features, d_model, nhead, num_transformer_layers, mlp_hidden_dims, dropout, dim_feedforward, model_name)
    avg_test_loss, all_preds = train_model(model, train_loader, test_loader, epochs=epochs, start_epoch=start_epoch, lr=lr, device=device, save_step=save_step)
    # print(all_preds)
    # result = all_preds > 0.5
    address_result = result_torch_to_pandas(model=model, model_name=model_name, result=all_preds, test_label=y_test,
                                            dic_symbol_embedding=dic_symbol_embedding, test_data=test_data, max_iter=epochs)

def  mlp_buy(address_of_data, folder_addr):
    load_from_file = True

    ml_data = load_pickle(address_of_data)
    seq_length = 5
    if load_from_file==False:

        temp = {}
        temp["AAPL"] = ml_data["AAPL"]
        temp["ABBV"] = ml_data["ABBV"]
        ml_data = temp
        ml_data, labels_dic = prepare_labels_and_data(ml_data, label_address= config["path"]["addr_labels"], simu_annealing_label_address= config["path"]["addr_simulated_annealing_labels"] )
        test_start = datetime(2023, 6, 1, 0, 0, 0)
        test_end = datetime(2025, 12, 1, 0, 0, 0)
        x_train, x_test, y_train, y_test = split_data(ml_data, labels_dic, test_start, test_end)
        # os.makedirs(config['path']['addr_ml_model'], exist_ok=True)
        # x_train, y_train, x_test, y_test, test_data = prepare_labels_and_data_sequential(ml_data, labels_dic, test_start, test_end, seq_length, "low")
        save_pickle(x_train, f"{folder_addr}x_train")
        save_pickle(y_train, f"{folder_addr}y_train")
        save_pickle(x_test, f"{folder_addr}x_test")
        save_pickle(y_test, f"{folder_addr}y_test")
        # save_pickle(test_data, f"{folder_addr}test_data")
        exit()
    else:
        x_train = load_pickle(f"{folder_addr}x_train")
        y_train = load_pickle(f"{folder_addr}y_train")
        x_test = load_pickle(f"{folder_addr}x_test")
        y_test = load_pickle(f"{folder_addr}y_test")
        # test_data = load_pickle(f"{folder_addr}test_data")

    dic_symbol_embedding = {}
    for symbol in list(ml_data.keys()):
        embedding = int(ml_data[symbol]["symbol_embedding"].iloc[0])
        dic_symbol_embedding[embedding] = symbol
    lr = 0.0001
    epochs = 50001
    selected_columns = ['count', 'rsi', 'so20', 'so50', 'ppo_exponential', 'ppo_exponential_6', 'ppo', 'cci', 'cmo', 'adl', 'dpo','tsi41',
     'tsi25'] #FOR BUY
    address_result_mlp = mlp_model(x_train, x_test, y_train, y_test, selected_columns, max_iter=epochs, lr=lr, model_name="mlp", type_label="low")

    # selected_columns =['symbol_embedding', 'count', 'so50', 'ppo_exponential', 'cci', 'cmo','tsi41', 'tsi25', 'profit'] # FOR SELL
    # address_result_mlp = mlp_model(x_train, x_test, y_train, y_test, selected_columns, max_iter=epochs, lr=lr, model_name="mlp", type_label="sa_high")
    # lst_resutl = [[address_result_mlp, "mlp"]]
    # vidualize_ml_output(ml_data, lst_resutl, prob=0.5)

if __name__ == "__main__":
    # trandfer_regression()
    address_of_data = "/home/test/projects/PromptKD/dl_project/ml_data"
    folder_addr = "/home/test/projects/PromptKD/dl_project/"
    config['path']['addr_ml_model'] = "/home/test/projects/PromptKD/dl_project/"
    # mlp_sell(address_of_data, folder_addr)
    # mlp_sell(address_of_data, folder_addr)
    # mlp_buy_seq(address_of_data, folder_addr)
    mlp_buy(address_of_data, folder_addr)
