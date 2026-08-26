from pathlib import Path
import json
import pandas as pd
import random
import numpy as np
import torch
from main.source_code import esn_model
from main.source_code import lstm_model
from main.source_code import utils


# ======================================
# LOAD DATA
# ======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "main" / "input" / "input.csv"
OUTPUT_DIR = BASE_DIR / "main" / "output"

CONFIG_PATH = BASE_DIR / "main" / "input" / "config.json"

WEIGHTS_PATH_LSTM = OUTPUT_DIR / "best_params_lstm.pt"
WEIGHTS_PATH_ENS = OUTPUT_DIR / "best_params_ens.pt"

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

data_config = config["data"]
lstm_config = config["lstm"]
esn_config = config["esn"]

input_series = pd.read_csv(DATA_PATH)['value']

# ======================================
# SET SEED FOR REPRODUCIBILITY
# ======================================
SEED = config["seed"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ======================================
# DATA PREPROCESSING
# ======================================
train, val, test = utils.split_data(input_series, data_config['train_end'], data_config['val_end'])
train_norm, val_norm, test_norm, train_mean, train_std = utils.zscore_normalize(train, val, test)

# ======================================
# LSTM
# ======================================
# Run LSTM
lstm_results = lstm_model.run_lstm(train_norm, val_norm, test_norm, lstm_config, device, WEIGHTS_PATH_LSTM)

# Return predictions and targets to the original scale
y_train_pred_lstm = utils.inverse_zscore(lstm_results['train_pred'], train_mean, train_std)
y_val_pred_lstm = utils.inverse_zscore(lstm_results['val_pred'], train_mean, train_std)
y_test_pred_lstm = utils.inverse_zscore(lstm_results['test_pred'], train_mean, train_std)
y_train_actual_lstm = train[lstm_config["sequence_length"]:].to_numpy().reshape(-1, 1)
y_val_actual_lstm = val[lstm_config["sequence_length"]:].to_numpy().reshape(-1, 1)
y_test_actual_lstm = test[lstm_config["sequence_length"]:].to_numpy().reshape(-1, 1)

# Metrics computation
train_mse_lstm = utils.mse(y_train_actual_lstm, y_train_pred_lstm)
val_mse_lstm = utils.mse(y_val_actual_lstm, y_val_pred_lstm)
test_mse_lstm  = utils.mse(y_test_actual_lstm, y_test_pred_lstm)

# ======================================
# ESN
# ======================================
# Run ESN
esn_results = esn_model.run_esn(train_norm, val_norm, test_norm, esn_config, device, WEIGHTS_PATH_ENS)

# Return predictions and targets to the original scale
y_train_pred_esn = utils.inverse_zscore(esn_results['train_pred'], train_mean, train_std)
y_val_pred_esn = utils.inverse_zscore(esn_results['val_pred'], train_mean, train_std)
y_test_pred_esn = utils.inverse_zscore(esn_results['test_pred'], train_mean, train_std)
y_train_actual_esn = torch.tensor(train[1:].to_numpy(), dtype=torch.float32).unsqueeze(1)
y_val_actual_esn = torch.tensor(val[1:].to_numpy(), dtype=torch.float32).unsqueeze(1)
y_test_actual_esn = torch.tensor(test[1:].to_numpy(), dtype=torch.float32).unsqueeze(1)

# Metrics computation
train_mse_esn = utils.mse(y_train_actual_esn, y_train_pred_esn)
val_mse_esn = utils.mse(y_val_actual_esn, y_val_pred_esn)
test_mse_esn  = utils.mse(y_test_actual_esn, y_test_pred_esn)

# ======================================
# SAVE METRICS AND PLOTS IN OUTPUT DIR
# ======================================
# Save metrics
results_df = pd.DataFrame({
    "Model": ["LSTM", "ESN"],
    "Train_MSE": [train_mse_lstm, train_mse_esn],
    "Validation_MSE": [val_mse_lstm, val_mse_esn],
    "Test_MSE": [test_mse_lstm, test_mse_esn],
    "Training_Time_s": [
        lstm_results["training_time"],
        esn_results["training_time"]
    ],
    "Inference_Time_s": [
        lstm_results["inference_time"],
        esn_results["inference_time"]
    ]
})

results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)

# Save graphs
utils.save_original_time_series(
    input_series,
    save_path=OUTPUT_DIR / "input_series.png"
)

utils.save_losses(
    lstm_results["train_losses"],
    lstm_results["val_losses"],
    save_path=OUTPUT_DIR / "lstm_losses.png"
)

utils.save_predictions(
    y_test_actual_lstm,
    y_test_pred_lstm,
    model_name="LSTM",
    save_path=OUTPUT_DIR / "lstm_predictions.png"
)

utils.save_predictions(
    y_test_actual_esn,
    y_test_pred_esn,
    model_name="ESN",
    save_path=OUTPUT_DIR / "esn_predictions.png"
)