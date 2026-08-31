import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import shutil
import json

from main.source_code import utils
from main.source_code import lstm_model
from main.source_code import esn_model
from fine_tuning.source_code import tuning

NUM_EPOCHS = 100
PATIENCE = 5
SEED = 0

BASE_DIR = Path(__file__).resolve().parent.parent

# Base directories
GUI_FINE_TUNING_DIR = BASE_DIR / "gui" / "fine_tuning"
GUI_INPUT_DIR = BASE_DIR / "gui" / "input"
GUI_WEIGHTS_DIR = BASE_DIR / "gui" / "weights"
GUI_CONFIG_DIR = BASE_DIR / "gui" / "config"

# Training LSTM
CONFIG_PATH_LSTM = GUI_CONFIG_DIR / "lstm_config.json"
WEIGHTS_PATH_LSTM = GUI_WEIGHTS_DIR / "lstm_weights.pt"

# Training ESN
CONFIG_PATH_ESN =  GUI_CONFIG_DIR/ "esn_config.json"
WEIGHTS_PATH_ESN = GUI_WEIGHTS_DIR / "esn_weights.pt"

# Parameter tuning LSTM
SEARCH_SPACE_PATH_LSTM = BASE_DIR / "fine_tuning"/ "input" / "search_spaces" / "lstm_search_space.json"
CONFIGS_PATH_LSTM = GUI_FINE_TUNING_DIR / "configs_lstm.csv"
RESULTS_PATH_LSTM = GUI_FINE_TUNING_DIR / "results_lstm.csv"
BEST_RESULTS_PATH_LSTM = GUI_FINE_TUNING_DIR / "best_result_lstm.csv"
BEST_CONFIG_PATH_LSTM = GUI_FINE_TUNING_DIR / "best_config_lstm.json"

# Parameter tuning ESN
SEARCH_SPACE_PATH_ESN = BASE_DIR / "fine_tuning" / "input" / "search_spaces" / "esn_search_space.json"
CONFIGS_PATH_ESN = GUI_FINE_TUNING_DIR / "configs_esn.csv"
RESULTS_PATH_ESN = GUI_FINE_TUNING_DIR / "results_esn.csv"
BEST_RESULTS_PATH_ESN = GUI_FINE_TUNING_DIR / "best_result_esn.csv"
BEST_CONFIG_PATH_ESN = GUI_FINE_TUNING_DIR / "best_config_esn.json"

def reset_gui():
    """
    Clear all GUI directories.
    """

    directories = [
        GUI_INPUT_DIR,
        GUI_CONFIG_DIR,
        GUI_WEIGHTS_DIR
    ]

    for directory in directories:
        utils.clear_directory(directory)

def load_dataset(file):
    """
    Load the uploaded CSV dataset and save a copy in the GUI input directory.

    Parameters
    ----------
    file : str
        Path to the CSV file selected by the user.

    Returns
    -------
    input_series : pd.Series
        Time series contained in the "value" column.
    info : str
        Basic information about the uploaded dataset.
    """

    if file is None:
        return None, "No dataset loaded."

    # Load CSV file
    df = pd.read_csv(file)

    # Check required column
    if "value" not in df.columns:
        raise gr.Error(
            "The CSV file must contain a column named 'value'."
        )

    # Get uploaded file path
    source_path = Path(file)

    # Save a copy inside gui/input
    destination_path = GUI_INPUT_DIR / source_path.name

    shutil.copy2(
        source_path,
        destination_path
    )

    # Extract input time series
    input_series = df["value"]

    # Dataset information
    info = (
        f"Dataset loaded successfully.\n"
        f"Number of samples: {len(df)}\n"
        f"Number of columns: {len(df.columns)}\n"
        f"Selected column: value"
    )

    return input_series, info

def save_dataset(file):
    """
    Save the uploaded CSV file in the GUI input directory.

    Parameters
    ----------
    file : str
        Path to the CSV file uploaded through the GUI.

    Returns
    -------
    message : str
        Confirmation message containing the saved file name.
    """

    if file is None:
        raise gr.Error(
            "No dataset loaded. Please upload a CSV file first."
        )

    # Clear files from the previous experiment
    reset_gui()

    # Get original file name
    source_path = Path(file)

    # Destination inside gui/input
    destination_path = GUI_INPUT_DIR / source_path.name

    # Copy uploaded file
    shutil.copy2(
        source_path,
        destination_path
    )

    return f"Dataset saved as `{source_path.name}`."

def load_lstm_config(config_file):
    """
    Load LSTM hyperparameters from a JSON configuration file.

    The configuration file is validated to ensure that all required
    hyperparameters are present.

    Parameters
    ----------
    config_file : str
        Path to the uploaded JSON configuration file.

    Returns
    -------
    hidden_size : int
        Number of hidden units in the LSTM layer.
    sequence_length : int
        Number of previous time steps used to predict the next value.
    batch_size : int
        Number of training samples per batch.
    learning_rate : float
        Optimizer learning rate.
    num_epochs : int
        Maximum number of training epochs.
    patience : int
        Number of consecutive epochs without validation loss improvement
        allowed before early stopping.

    Raises
    ------
    gr.Error
        If no configuration file is provided or if one or more required
        hyperparameters are missing.
    """

    if config_file is None:
        raise gr.Error(
            "Please upload a JSON configuration file."
        )

    # Load configuration
    with open(config_file, "r", encoding="utf-8") as file:
        config = json.load(file)

    # Check required parameters
    required_parameters = [
        "hidden_size",
        "sequence_length",
        "batch_size",
        "learning_rate",
        "num_epochs",
        "patience"
    ]

    missing_parameters = [
        parameter
        for parameter in required_parameters
        if parameter not in config
    ]

    if missing_parameters:
        raise gr.Error(
            "Invalid LSTM configuration. Missing parameters: "
            + ", ".join(missing_parameters)
        )

    return (
        int(config["hidden_size"]),
        int(config["sequence_length"]),
        int(config["batch_size"]),
        float(config["learning_rate"]),
        int(config["num_epochs"]),
        int(config["patience"])
    )

def load_esn_config(config_file):
    """
    Load an ESN configuration from a JSON file.

    Parameters
    ----------
    config_file : str
        Path to the uploaded JSON configuration file.

    Returns
    -------
    reservoir_size : int
        Number of reservoir units.
    spectral_radius : float
        Spectral radius of the reservoir matrix.
    reservoir_connectivity : float
        Fraction of non-zero reservoir connections.
    input_scaling : float
        Scaling factor applied to the input weights.
    washout : int
        Number of initial reservoir states discarded.
    alpha : float
        Leaky integration parameter.
    ridge : float
        Ridge regularization coefficient.
    """

    if config_file is None:
        raise gr.Error(
            "Please upload a JSON configuration file."
        )

    # Load configuration
    with open(config_file, "r", encoding="utf-8") as file:
        config = json.load(file)

    # Check required parameters
    required_parameters = [
        "reservoir_size",
        "spectral_radius",
        "reservoir_connectivity",
        "input_scaling",
        "washout",
        "alpha",
        "ridge"
    ]

    missing_parameters = [
        parameter
        for parameter in required_parameters
        if parameter not in config
    ]

    if missing_parameters:
        raise gr.Error(
            "Invalid ESN configuration. Missing parameters: "
            + ", ".join(missing_parameters)
        )

    # Return hyperparameters in the same order as the GUI outputs
    return (
        int(config["reservoir_size"]),
        float(config["spectral_radius"]),
        float(config["reservoir_connectivity"]),
        float(config["input_scaling"]),
        int(config["washout"]),
        float(config["alpha"]),
        float(config["ridge"])
    )

def get_lstm_config_file():
    """
    Return the path to the current LSTM configuration file.

    Returns
    -------
    config_path : str
        Path to the LSTM JSON configuration file.

    Raises
    ------
    gr.Error
        If no LSTM configuration file is available.
    """
    if not CONFIG_PATH_LSTM.exists():
        raise gr.Error(
            "No LSTM configuration is available. Run the model first."
        )

    return str(CONFIG_PATH_LSTM)

def get_lstm_weights_file():
    """
    Return the path to the current LSTM weights file.

    Returns
    -------
    weights_path : str
        Path to the LSTM weights file.

    Raises
    ------
    gr.Error
        If no LSTM weights file is available.
    """
    if not WEIGHTS_PATH_LSTM.exists():
        raise gr.Error(
            "No LSTM weights are available. Run the model first."
        )

    return str(WEIGHTS_PATH_LSTM)

def get_esn_config_file():
    """
    Return the path to the current ESN configuration file.

    Returns
    -------
    config_path : str
        Path to the ESN JSON configuration file.

    Raises
    ------
    gr.Error
        If no ESN configuration file is available.
    """

    if not CONFIG_PATH_ESN.exists():
        raise gr.Error(
            "No ESN configuration is available. "
            "Run the model first."
        )

    return str(CONFIG_PATH_ESN)

def get_esn_weights_file():
    """
    Return the path to the current ESN weights file.

    Returns
    -------
    weights_path : str
        Path to the ESN weights file.

    Raises
    ------
    gr.Error
        If no ESN weights file is available.
    """

    if not WEIGHTS_PATH_ESN.exists():
        raise gr.Error(
            "No ESN weights are available. "
            "Run the model first."
        )

    return str(WEIGHTS_PATH_ESN)

def get_best_lstm_config_file():
    """
    Return the path to the best LSTM configuration found by random search.

    Returns
    -------
    config_path : str
        Path to the JSON file containing the best LSTM configuration
        found by random search.

    Raises
    ------
    gr.Error
        If no tuned LSTM configuration file is available.
    """

    if not BEST_CONFIG_PATH_LSTM.exists():
        raise gr.Error(
            "No tuned LSTM configuration is available. "
            "Run the hyperparameter tuning first."
        )

    return str(BEST_CONFIG_PATH_LSTM)

def get_best_esn_config_file():
    """
    Return the path to the best ESN configuration found by random search.

    Returns
    -------
    config_path : str
        Path to the JSON file containing the best ESN configuration
        found by random search.

    Raises
    ------
    gr.Error
        If no tuned ESN configuration file is available.
    """

    if not BEST_CONFIG_PATH_ESN.exists():
        raise gr.Error(
            "No tuned ESN configuration is available. "
            "Run the hyperparameter tuning first."
        )

    return str(BEST_CONFIG_PATH_ESN)

def split_and_normalize(input_series, train_end, val_end):
    """
    Split the time series into training, validation, and test sets
    and normalize them using training-set statistics.

    Parameters
    ----------
    input_series : pd.Series
        Original input time series.
    train_end : int
        End index of the training set.
    val_end : int
        End index of the validation set.

    Returns
    -------
    train, val, test : pd.Series
        Original-scale data splits.
    train_norm, val_norm, test_norm : pd.Series
        Normalized data splits.
    train_mean : float
        Mean of the training set.
    train_std : float
        Standard deviation of the training set.
    split_info : str
        Summary of the dataset split.
    """

    if input_series is None:
        raise gr.Error("Please upload a dataset first.")

    train_end = int(train_end)
    val_end = int(val_end)

    n_samples = len(input_series)

    # Check split indices
    if not 0 < train_end < val_end < n_samples:
        raise gr.Error(
            "The split must satisfy: "
            "0 < train_end < val_end < number of samples."
        )

    # Split original data
    train, val, test = utils.split_data(input_series, train_end, val_end)

    # Normalize using training-set statistics
    train_norm, val_norm, test_norm, train_mean, train_std = utils.zscore_normalize(train, val, test)

    # Split summary displayed in the GUI
    split_info = (
        f"Train: {len(train)} samples "
        f"({len(train) / n_samples:.1%})\n"
        f"Validation: {len(val)} samples "
        f"({len(val) / n_samples:.1%})\n"
        f"Test: {len(test)} samples "
        f"({len(test) / n_samples:.1%})"
    )

    return train, val, test, train_norm, val_norm, test_norm, train_mean, train_std, split_info

def plot_time_series(input_series, start, end):
    """
    Plot a selected range of the input time series.

    Parameters
    ----------
    input_series : pd.Series
        Input time series.
    start : int
        Initial sample index.
    end : int
        Final sample index.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Time-series plot.
    """

    if input_series is None:
        raise gr.Error("Please upload a dataset first.")

    start = int(start)
    end = int(end)

    # Check selected range
    if start < 0:
        raise gr.Error("Start index cannot be negative.")

    if end > len(input_series):
        raise gr.Error(
            "End index cannot exceed the number of samples."
        )

    if start >= end:
        raise gr.Error(
            "Start index must be smaller than end index."
        )

    # Select requested range
    series = input_series.iloc[start:end]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        series.index,
        series.values
    )

    ax.set_title("Time Series")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Value")

    fig.tight_layout()

    return fig

def plot_predictions(
    y_actual,
    y_pred,
    model_name,
    save_path=None
):
    """
    Plot actual and predicted values and optionally save the figure.
    """

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(y_actual, label="Actual")
    ax.plot(y_pred, label="Predicted")

    ax.set_title(f"{model_name} - Actual vs Predicted")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Test Value")
    ax.legend()

    fig.tight_layout()

    # Save only if a path is provided
    if save_path is not None:
        fig.savefig(save_path)

    return fig

def run_lstm_from_gui(
    input_series,
    train,
    val,
    test,
    train_norm,
    val_norm,
    test_norm,
    train_mean,
    train_std,
    lstm_mode,
    hidden_size,
    sequence_length,
    batch_size,
    learning_rate,
    num_epochs,
    patience,
    pretrained_config_file,
    pretrained_weights_file
):
    """
    Run the LSTM from the GUI using either a newly trained model
    or a pretrained model.

    Parameters
    ----------
    input_series : pd.Series
        Original input time series.
    train, val, test : pd.Series
        Original-scale training, validation, and test sets.
    train_norm, val_norm, test_norm : pd.Series
        Normalized training, validation, and test sets.
    train_mean : float
        Mean of the training set used for normalization.
    train_std : float
        Standard deviation of the training set used for normalization.
    lstm_mode : str
        Selected LSTM mode: "Train New Model" or "Use Pretrained Model".
    hidden_size : int
        Number of LSTM hidden units.
    sequence_length : int
        Number of time steps in each input sequence.
    batch_size : int
        Number of samples per training batch.
    learning_rate : float
        Optimizer learning rate.
    pretrained_config_file : str or None
        Path to the uploaded pretrained configuration file.
    pretrained_weights_file : str or None
        Path to the uploaded pretrained weights file.
    num_epochs : int
        Maximum number of training epochs.
    patience : int
        Early-stopping patience.

    Returns
    -------
    train_mse : float
        Training mean squared error.
    val_mse : float
        Validation mean squared error.
    test_mse : float
        Test mean squared error.
    training_time : float or None
        Training time in seconds. None when a pretrained model is used.
    inference_time : float
        Test-set inference time in seconds.
    prediction_plot : matplotlib.figure.Figure
        Plot comparing actual and predicted test-set values.
    """

    # Check that a dataset has been uploaded
    if input_series is None:
        raise gr.Error(
            "No dataset loaded. Please upload a CSV file in the Data tab "
            "before running the LSTM."
        )

    # Check that the dataset has been split
    if train_norm is None or val_norm is None or test_norm is None:
        raise gr.Error(
            "The dataset has not been split yet. Please define and apply "
            "the train/validation/test split in the Data tab before "
            "running the LSTM."
        )

    # Select computation device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Remove files from the previous model run
    # reset_model_run()

    # ====================================================
    # TRAIN NEW MODEL
    # ====================================================

    if lstm_mode == "Train New Model":

        load_pretrained = False

        # Build configuration from GUI values
        lstm_config = {
            "input_size": 1,
            "hidden_size": int(hidden_size),
            "output_size": 1,
            "sequence_length": int(sequence_length),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "num_epochs": int(num_epochs),
            "patience": int(patience)
        }

        # Save the configuration actually used for training
        with open(CONFIG_PATH_LSTM, "w", encoding="utf-8") as file:
            json.dump(
                lstm_config,
                file,
                indent=4
            )

    # ====================================================
    # USE PRETRAINED MODEL
    # ====================================================

    elif lstm_mode == "Use Pretrained Model":

        load_pretrained = True

        # Check required files
        if pretrained_config_file is None:
            raise gr.Error(
                "Please upload the JSON configuration associated "
                "with the pretrained LSTM."
            )

        if pretrained_weights_file is None:
            raise gr.Error(
                "Please upload the .pt file containing the pretrained "
                "LSTM weights."
            )

        # Copy uploaded files to fixed GUI paths
        shutil.copy2(
            pretrained_config_file,
            CONFIG_PATH_LSTM
        )

        shutil.copy2(
            pretrained_weights_file,
            WEIGHTS_PATH_LSTM
        )

        # Load configuration
        with open(CONFIG_PATH_LSTM, "r", encoding="utf-8") as file:
            lstm_config = json.load(file)

    else:
        raise gr.Error("Invalid LSTM mode.")

    # ====================================================
    # Set seed for reproducibility
    # ====================================================

    utils.set_seed(SEED)

    # ====================================================
    # RUN LSTM
    # ====================================================

    lstm_results = lstm_model.run_lstm(
        train_norm,
        val_norm,
        test_norm,
        lstm_config,
        device,
        WEIGHTS_PATH_LSTM,
        load_pretrained
    )

    # Return predictions to original scale
    y_train_pred_lstm = utils.inverse_zscore(
        lstm_results["train_pred"],
        train_mean,
        train_std
    )

    y_val_pred_lstm = utils.inverse_zscore(
        lstm_results["val_pred"],
        train_mean,
        train_std
    )

    y_test_pred_lstm = utils.inverse_zscore(
        lstm_results["test_pred"],
        train_mean,
        train_std
    )

    # Align actual values with sliding-window predictions
    seq_len = int(lstm_config["sequence_length"])

    y_train_actual_lstm = (
        train[seq_len:]
        .to_numpy()
        .reshape(-1, 1)
    )

    y_val_actual_lstm = (
        val[seq_len:]
        .to_numpy()
        .reshape(-1, 1)
    )

    y_test_actual_lstm = (
        test[seq_len:]
        .to_numpy()
        .reshape(-1, 1)
    )

    # Compute MSE
    train_mse = utils.mse(
        y_train_actual_lstm,
        y_train_pred_lstm
    )

    val_mse = utils.mse(
        y_val_actual_lstm,
        y_val_pred_lstm
    )

    test_mse = utils.mse(
        y_test_actual_lstm,
        y_test_pred_lstm
    )

    # Create test prediction plot
    prediction_plot = plot_predictions(
        y_test_actual_lstm,
        y_test_pred_lstm,
        model_name="LSTM"
    )

    # Training time is None when pretrained weights are used
    training_time = lstm_results["training_time"]

    if training_time is not None:
        training_time = float(training_time)

    return (
        float(train_mse),
        float(val_mse),
        float(test_mse),
        training_time,
        float(lstm_results["inference_time"]),
        prediction_plot
    )

def run_esn_from_gui(
    input_series,
    train,
    val,
    test,
    train_norm,
    val_norm,
    test_norm,
    train_mean,
    train_std,
    esn_mode,
    reservoir_size,
    spectral_radius,
    reservoir_connectivity,
    input_scaling,
    washout,
    alpha,
    ridge,
    pretrained_config_file,
    pretrained_weights_file
):
    """
    Run the ESN from the GUI using either a newly trained model
    or a pretrained model.

    Parameters
    ----------
    input_series : pd.Series
        Original input time series.
    train, val, test : pd.Series
        Original-scale training, validation, and test sets.
    train_norm, val_norm, test_norm : pd.Series
        Normalized training, validation, and test sets.
    train_mean : float
        Mean of the training set used for normalization.
    train_std : float
        Standard deviation of the training set used for normalization.
    esn_mode : str
        Selected ESN mode: "Train New Model" or "Use Pretrained Model".
    reservoir_size : int
        Number of reservoir units.
    spectral_radius : float
        Spectral radius of the reservoir weight matrix.
    reservoir_connectivity : float
        Fraction of non-zero reservoir connections.
    input_scaling : float
        Scaling factor applied to the input weights.
    washout : int
        Number of initial reservoir states discarded during training.
    alpha : float
        Leaky integration parameter.
    ridge : float
        Ridge regularization coefficient.
    pretrained_config_file : str or None
        Path to the uploaded pretrained configuration file.
    pretrained_weights_file : str or None
        Path to the uploaded pretrained weights file.

    Returns
    -------
    train_mse : float
        Training mean squared error.
    val_mse : float
        Validation mean squared error.
    test_mse : float
        Test mean squared error.
    training_time : float or None
        Training time in seconds. None when a pretrained model is used.
    inference_time : float
        Test-set inference time in seconds.
    prediction_plot : matplotlib.figure.Figure
        Plot comparing actual and predicted test-set values.
    """

    # Check that a dataset has been uploaded
    if input_series is None:
        raise gr.Error(
            "No dataset loaded. Please upload a CSV file in the Data tab "
            "before running the ESN."
        )

    # Check that the dataset has been split
    if train_norm is None or val_norm is None or test_norm is None:
        raise gr.Error(
            "The dataset has not been split yet. Please define and apply "
            "the train/validation/test split in the Data tab before "
            "running the ESN."
        )

    # Select computation device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ====================================================
    # TRAIN NEW MODEL
    # ====================================================

    if esn_mode == "Train New Model":

        load_pretrained = False

        # Build configuration from GUI values
        esn_config = {
            "reservoir_size": int(reservoir_size),
            "spectral_radius": float(spectral_radius),
            "reservoir_connectivity": float(reservoir_connectivity),
            "input_scaling": float(input_scaling),
            "washout": int(washout),
            "alpha": float(alpha),
            "ridge": float(ridge)
        }

        # Save the configuration actually used for training
        with open(CONFIG_PATH_ESN, "w", encoding="utf-8") as file:
            json.dump(
                esn_config,
                file,
                indent=4
            )

    # ====================================================
    # USE PRETRAINED MODEL
    # ====================================================

    elif esn_mode == "Use Pretrained Model":

        load_pretrained = True

        # Check required files
        if pretrained_config_file is None:
            raise gr.Error(
                "Please upload the JSON configuration associated "
                "with the pretrained ESN."
            )

        if pretrained_weights_file is None:
            raise gr.Error(
                "Please upload the .pt file containing the pretrained "
                "ESN weights."
            )

        # Copy uploaded files to fixed GUI paths
        shutil.copy2(
            pretrained_config_file,
            CONFIG_PATH_ESN
        )

        shutil.copy2(
            pretrained_weights_file,
            WEIGHTS_PATH_ESN
        )

        # Load configuration
        with open(CONFIG_PATH_ESN, "r", encoding="utf-8") as file:
            esn_config = json.load(file)

    else:
        raise gr.Error("Invalid ESN mode.")

    # ====================================================
    # Set seed for reproducibility
    # ====================================================

    utils.set_seed(SEED)

    # ====================================================
    # RUN ESN
    # ====================================================

    esn_results = esn_model.run_esn(
        train_norm,
        val_norm,
        test_norm,
        esn_config,
        device,
        WEIGHTS_PATH_ESN,
        load_pretrained
    )

    # Return predictions to original scale
    y_train_pred_esn = utils.inverse_zscore(
        esn_results["train_pred"],
        train_mean,
        train_std
    )

    y_val_pred_esn = utils.inverse_zscore(
        esn_results["val_pred"],
        train_mean,
        train_std
    )

    y_test_pred_esn = utils.inverse_zscore(
        esn_results["test_pred"],
        train_mean,
        train_std
    )

    # Create actual one-step-ahead targets
    y_train_actual_esn = torch.tensor(
        train[1:].to_numpy(),
        dtype=torch.float32
    ).unsqueeze(1)

    y_val_actual_esn = torch.tensor(
        val[1:].to_numpy(),
        dtype=torch.float32
    ).unsqueeze(1)

    y_test_actual_esn = torch.tensor(
        test[1:].to_numpy(),
        dtype=torch.float32
    ).unsqueeze(1)

    # Compute MSE
    train_mse = utils.mse(
        y_train_actual_esn,
        y_train_pred_esn
    )

    val_mse = utils.mse(
        y_val_actual_esn,
        y_val_pred_esn
    )

    test_mse = utils.mse(
        y_test_actual_esn,
        y_test_pred_esn
    )

    # Create test prediction plot
    prediction_plot = plot_predictions(
        y_test_actual_esn,
        y_test_pred_esn,
        model_name="ESN"
    )

    # Training time is None when pretrained weights are used
    training_time = esn_results["training_time"]

    if training_time is not None:
        training_time = float(training_time)

    return (
        float(train_mse),
        float(val_mse),
        float(test_mse),
        training_time,
        float(esn_results["inference_time"]),
        prediction_plot
    )

def run_random_search_from_gui(
    model,
    input_series,
    train_norm,
    val_norm,
    n_trials,
    n_seeds,
    search_space_path,
    configs_path,
    results_path,
    best_results_path
):
    """
    Run random-search hyperparameter tuning from the GUI.

    A new random search is started every time the function is called.
    Random hyperparameter configurations are generated from the specified
    search space, and each configuration is evaluated over multiple random
    seeds. The configuration with the lowest mean validation loss is selected.

    Parameters
    ----------
    model : str
        Model to optimize. Must be either "lstm" or "esn".
    input_series : pd.Series
        Original input time series.
    train_norm : pd.Series
        Normalized training set.
    val_norm : pd.Series
        Normalized validation set.
    n_trials : int
        Number of random hyperparameter configurations to evaluate.
    n_seeds : int
        Number of random seeds used to evaluate each configuration.
    search_space_path : str or Path
        Path to the JSON file containing the hyperparameter search space.
    configs_path : str or Path
        Path where the generated configurations are saved.
    results_path : str or Path
        Path where the random-search results are saved.
    best_results_path : str or Path
        Path where the best result found is saved.

    Returns
    -------
    best_config : dict
        Best hyperparameter configuration found.

    Raises
    ------
    gr.Error
        If no dataset has been uploaded, if the dataset has not been split,
        or if the number of trials or seeds is smaller than 1.
    """

    # Check that a dataset has been uploaded
    if input_series is None:
        raise gr.Error(
            "No dataset loaded. Please upload a CSV file in the Data tab "
            "before starting hyperparameter tuning."
        )

    # Check that the dataset has been split
    if train_norm is None or val_norm is None:
        raise gr.Error(
            "The dataset has not been split yet. Please define and apply "
            "the train/validation/test split in the Data tab before "
            "starting hyperparameter tuning."
        )

    # Check tuning parameters
    n_trials = int(n_trials)
    n_seeds = int(n_seeds)

    if n_trials < 1:
        raise gr.Error(
            "The number of trials must be at least 1."
        )

    if n_seeds < 1:
        raise gr.Error(
            "The number of seeds must be at least 1."
        )

    # Select computation device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    seeds = range(n_seeds)

    # Generate random hyperparameter configurations
    configs = tuning.generate_configs(
        model,
        search_space_path,
        n_trials,
        configs_path,
        search_seed=42
    )

    # Run a new random search from scratch
    best_config,_ = tuning.random_search(
        model,
        train_norm,
        val_norm,
        device,
        configs,
        seeds,
        results_path,
        best_results_path,
        restart=True
    )

    # Convert Pandas Series to dictionary
    if hasattr(best_config, "to_dict"):
        best_config = best_config.to_dict()

    # Convert NumPy scalar values to standard Python values
    best_config = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in best_config.items()
    }

    return best_config

def run_lstm_tuning_gui(
    input_series,
    train_norm,
    val_norm,
    n_trials,
    n_seeds
):
    """
    Run LSTM random search from the GUI and return only the
    user-editable hyperparameters of the best configuration.
    """

    best_config = run_random_search_from_gui(
        model="lstm",
        input_series=input_series,
        train_norm=train_norm,
        val_norm=val_norm,
        n_trials=n_trials,
        n_seeds=n_seeds,
        search_space_path=SEARCH_SPACE_PATH_LSTM,
        configs_path=CONFIGS_PATH_LSTM,
        results_path=RESULTS_PATH_LSTM,
        best_results_path=BEST_RESULTS_PATH_LSTM
    )

    # Keep only hyperparameters editable from the GUI
    editable_parameters = [
        "hidden_size",
        "sequence_length",
        "batch_size",
        "learning_rate",
        "num_epochs",
        "patience"
    ]

    best_config = {
        key: best_config[key]
        for key in editable_parameters
    }

    # Save best configuration found by random search
    with open(BEST_CONFIG_PATH_LSTM, "w", encoding="utf-8") as file:
        json.dump(
            best_config,
            file,
            indent=4
        )

    return best_config

def run_esn_tuning_gui(
    input_series,
    train_norm,
    val_norm,
    n_trials,
    n_seeds
):
    """
    Run ESN random search from the GUI and return only the
    user-editable hyperparameters of the best configuration.
    """

    best_config = run_random_search_from_gui(
        model="esn",
        input_series=input_series,
        train_norm=train_norm,
        val_norm=val_norm,
        n_trials=n_trials,
        n_seeds=n_seeds,
        search_space_path=SEARCH_SPACE_PATH_ESN,
        configs_path=CONFIGS_PATH_ESN,
        results_path=RESULTS_PATH_ESN,
        best_results_path=BEST_RESULTS_PATH_ESN
    )

    # Save best configuration found by random search
    with open(BEST_CONFIG_PATH_ESN, "w", encoding="utf-8") as file:
        json.dump(
            best_config,
            file,
            indent=4
        )

    return best_config