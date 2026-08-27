import torch
import json
import pandas as pd
from pathlib import Path
from main.source_code import utils
from fine_tuning.source_code import tuning

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "main" / "input" / "input.csv"
CONFIG_PATH = BASE_DIR / "main" / "input" / "config.json"
SEARCH_SPACE_PATH_LSTM = BASE_DIR / "fine_tuning" / "input" / "search_spaces" / "lstm_search_space.json"
SEARCH_SPACE_PATH_ESN = BASE_DIR / "fine_tuning" / "input" / "search_spaces" / "esn_search_space.json" 
CONFIGS_PATH_LSTM = BASE_DIR / "fine_tuning" / "input" / "configs" / "configs_lstm.csv"
CONFIGS_PATH_ESN = BASE_DIR / "fine_tuning" / "input" / "configs" / "configs_esn.csv"
RESULTS_PATH_LSTM = BASE_DIR / "fine_tuning" / "output" / "results_lstm.csv"
BEST_RESULTS_PATH_LSTM = BASE_DIR / "fine_tuning" / "output" / "best_result_lstm.csv"
RESULTS_PATH_ESN = BASE_DIR / "fine_tuning" / "output" / "results_ens.csv"
BEST_RESULTS_PATH_ESN = BASE_DIR / "fine_tuning" / "output" / "best_result_esn.csv"

# Parameters
MODEL_LSTM =  "lstm"
N_CONFIG_LSTM = 2
SEEDS_LSTM = range(1)
RESTART_LSTM = True

MODEL_ESN = "esn"
N_CONFIG_ESN = 10
SEEDS_ESN = range(5)
RESTART_ESN = True

# Data preprocessing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

data_config = config["data"]

mg17_series = pd.read_csv(DATA_PATH)['value']

train, val, test = utils.split_data(mg17_series, data_config['train_end'], data_config['val_end'])
train_norm, val_norm, test_norm, train_mean, train_std = utils.zscore_normalize(train, val, test)

#=============================
# LSTM FINE TUNING
#=============================

print("Fine tuning LSTM...")

# Generate parameter configurations to test
configs_lstm = tuning.generate_configs(MODEL_LSTM, SEARCH_SPACE_PATH_LSTM, N_CONFIG_LSTM, CONFIGS_PATH_LSTM)

# Calculate best parameter configuration and best loss
best_config_lstm, best_val_loss_lstm = tuning.random_search(MODEL_LSTM, train_norm, val_norm, device, configs_lstm, SEEDS_LSTM, RESULTS_PATH_LSTM, BEST_RESULTS_PATH_LSTM, RESTART_LSTM)

#=============================
# ESN FINE TUNING
#=============================

print("\n Fine tuning ESN...")

# Generate parameter configurations to test
configs_esn = tuning.generate_configs("esn", SEARCH_SPACE_PATH_ESN, N_CONFIG_ESN, CONFIGS_PATH_ESN)

# Calculate best parameter configuration and best loss
best_config_esn, best_val_loss_esn = tuning.random_search(MODEL_ESN, train_norm, val_norm, device, configs_esn, SEEDS_ESN, RESULTS_PATH_ESN, BEST_RESULTS_PATH_ESN, RESTART_ESN)