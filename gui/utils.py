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

BASE_DIR = Path(__file__).resolve().parent.parent
GUI_OUTPUT_DIR = BASE_DIR / "gui" / "output"

GUI_INPUT_DIR = BASE_DIR / "gui" / "input"

SEARCH_SPACE_PATH_LSTM = BASE_DIR / "fine_tuning"/ "input" / "search_spaces" / "lstm_search_space.json"
CONFIGS_PATH_LSTM = GUI_OUTPUT_DIR / "configs_lstm.csv"
RESULTS_PATH_LSTM = GUI_OUTPUT_DIR / "results_lstm.csv"
BEST_RESULTS_PATH_LSTM = GUI_OUTPUT_DIR / "best_result_lstm.csv"

SEARCH_SPACE_PATH_ESN = BASE_DIR / "fine_tuning" / "input" / "search_spaces" / "esn_search_space.json"
CONFIGS_PATH_ESN = GUI_OUTPUT_DIR / "configs_esn.csv"
RESULTS_PATH_ESN = GUI_OUTPUT_DIR / "results_esn.csv"
BEST_RESULTS_PATH_ESN = GUI_OUTPUT_DIR / "best_result_esn.csv"

def load_dataset(file):
    """
    Load a CSV dataset and extract the input time series.

    Parameters
    ----------
    file : str
        Path to the uploaded CSV file.

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

    # Check expected time-series column
    if "value" not in df.columns:
        raise gr.Error(
            "The CSV file must contain a column named 'value'."
        )

    input_series = df["value"]

    # Dataset information displayed in the GUI
    info = (
        f"Dataset loaded successfully.\n"
        f"Num ber of samples: {len(df)}\n"
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
    Load an LSTM configuration from a JSON file.

    Parameters
    ----------
    config_file : str
        Path to the uploaded JSON configuration file.

    Returns
    -------
    hidden_size : int
        Number of LSTM hidden units.
    sequence_length : int
        Length of the input sequences.
    batch_size : int
        Training batch size.
    learning_rate : float
        Optimizer learning rate.
    """

    if config_file is None:
        raise gr.Error(
            "Please upload a JSON configuration file."
        )

    # Load configuration
    with open(config_file, "r") as file:
        config = json.load(file)

    # Check required parameters
    required_parameters = [
        "hidden_size",
        "sequence_length",
        "batch_size",
        "learning_rate"
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
        float(config["learning_rate"])
    )

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
    hidden_size,
    sequence_length,
    batch_size,
    learning_rate,
    num_epochs = NUM_EPOCHS,
    patience = PATIENCE
):
    """
    Run the LSTM using hyperparameters provided through the GUI
    and return train, validation, and test MSE.

    Parameters
    ----------
    input_series: pd.Series
        Input data series.
    train, val, test : pd.Series
        Original-scale training, validation, and test sets.
    train_norm, val_norm, test_norm : pd.Series
        Normalized training, validation, and test sets.
    train_mean : float
        Mean of the training set used for normalization.
    train_std : float
        Standard deviation of the training set used for normalization.
    hidden_size : int
        Number of LSTM hidden units.
    sequence_length : int
        Number of time steps in each input sequence.
    batch_size : int
        Number of samples per training batch.
    learning_rate : float
        Optimizer learning rate.
    num_epochs : int
        Maximum number of training epochs.
    patience : int
        Early-stopping patience.
    training_time: float
        Time for training (s).
    inference_time: float
        Time for inference (s).

    Returns
    -------
    train_mse : float
        Training mean squared error.
    val_mse : float
        Validation mean squared error.
    test_mse : float
        Test mean squared error.
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
            "the train/validation/test split in the Data tab before running the LSTM."
        )
    # Select computation device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Build configuration from GUI inputs
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

    # Run the existing LSTM pipeline
    lstm_results = lstm_model.run_lstm(
        train_norm,
        val_norm,
        test_norm,
        lstm_config,
        device,
        weights_path=None,
        load_pretrained=False,
        save_weights = False
    )

    # Return predictions to the original scale
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

    # Align actual values with the sliding-window predictions
    seq_len = lstm_config["sequence_length"]

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

    prediction_plot = plot_predictions(
    y_test_actual_lstm,
    y_test_pred_lstm,
    model_name="LSTM"
)

    return (
        float(train_mse),
        float(val_mse),
        float(test_mse),
        float(lstm_results["training_time"]), 
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
    reservoir_size,
    spectral_radius,
    reservoir_connectivity,
    input_scaling,
    washout,
    alpha,
    ridge
):
    """
    Run the ESN using hyperparameters provided through the GUI
    and return model performance metrics and the prediction plot.

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

    Returns
    -------
    train_mse : float
        Training mean squared error.
    val_mse : float
        Validation mean squared error.
    test_mse : float
        Test mean squared error.
    training_time : float
        ESN training time in seconds.
    inference_time : float
        ESN inference time in seconds.
    prediction_plot : matplotlib.figure.Figure
        Actual vs predicted values on the test set.
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
            "the train/validation/test split in the Data tab before running the ESN."
        )

    # Select computation device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Build ESN configuration from GUI inputs
    esn_config = {
        "reservoir_size": int(reservoir_size),
        "spectral_radius": float(spectral_radius),
        "reservoir_connectivity": float(reservoir_connectivity),
        "input_scaling": float(input_scaling),
        "washout": int(washout),
        "alpha": float(alpha),
        "ridge": float(ridge)
    }

    # Run ESN
    esn_results = esn_model.run_esn(
        train_norm,
        val_norm,
        test_norm,
        esn_config,
        device
    )

    # Return predictions to the original scale
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

    # Create test-set prediction plot
    prediction_plot = plot_predictions(
        y_test_actual_esn,
        y_test_pred_esn,
        model_name="ESN"
    )

    return (
        float(train_mse),
        float(val_mse),
        float(test_mse),
        float(esn_results["training_time"]),
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
    Run random hyperparameter tuning from the GUI.

    A new random search is started every time the function is called.
    Each sampled configuration is evaluated over multiple random seeds,
    and the configuration with the lowest mean validation loss is returned.

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
        Number of random seeds used for each configuration.
    search_space_path : str or Path
        Path to the JSON file containing the hyperparameter search space.
    configs_path : str or Path
        Path where generated configurations are saved.
    results_path : str or Path
        Path where tuning results are saved.
    best_results_path : str or Path
        Path where the best tuning result is saved.

    Returns
    -------
    best_config : dict
        Best hyperparameter configuration found.
    best_val_loss : float
        Mean validation loss of the best configuration.
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
        "learning_rate"
    ]

    best_config = {
        key: best_config[key]
        for key in editable_parameters
    }

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

    return best_config