import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import torch

from main.source_code import utils
from main.source_code import lstm_model
from main.source_code import esn_model

NUM_EPOCHS = 100
PATIENCE = 5 

# ============================================================
# DATA FUNCTIONS
# ============================================================

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
        f"Number of samples: {len(df)}\n"
        f"Number of columns: {len(df.columns)}\n"
        f"Selected column: value"
    )

    return input_series, info


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
    train, val, test = utils.split_data(
        input_series,
        train_end,
        val_end
    )

    # Normalize using training-set statistics
    (
        train_norm,
        val_norm,
        test_norm,
        train_mean,
        train_std
    ) = utils.zscore_normalize(
        train,
        val,
        test
    )

    # Split summary displayed in the GUI
    split_info = (
        f"Train: {len(train)} samples "
        f"({len(train) / n_samples:.1%})\n"
        f"Validation: {len(val)} samples "
        f"({len(val) / n_samples:.1%})\n"
        f"Test: {len(test)} samples "
        f"({len(test) / n_samples:.1%})"
    )

    return (
        train,
        val,
        test,
        train_norm,
        val_norm,
        test_norm,
        train_mean,
        train_std,
        split_info
    )

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
        device
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

# ============================================================
# GRAPHICAL USER INTERFACE
# ============================================================

with gr.Blocks(title="Time Series Prediction") as demo:

    gr.Markdown("# Time Series Prediction")

    # --------------------------------------------------------
    # Internal states shared between GUI events
    # --------------------------------------------------------

    input_series_state = gr.State()

    train_state = gr.State()
    val_state = gr.State()
    test_state = gr.State()

    train_norm_state = gr.State()
    val_norm_state = gr.State()
    test_norm_state = gr.State()

    train_mean_state = gr.State()
    train_std_state = gr.State()

    with gr.Tabs():

        # ====================================================
        # DATA TAB
        # ====================================================

        with gr.Tab("Data"):

            gr.Markdown("## Data")

            # -------------------------
            # 1. Upload dataset
            # -------------------------

            gr.Markdown("### 1. Upload Dataset")

            gr.Markdown(
                """
                **Input format:** Upload a CSV file containing the time series in a column named `value`. Do not include a time column. Values must be equally spaced in time, for example one value every second, minute, or hour. The x-axis of the plots represents these time units.
                """
            )

            csv_file = gr.File(
                label="Upload CSV",
                file_types=[".csv"],
                type="filepath"
            )

            dataset_info = gr.Textbox(
                label="Dataset Info",
                interactive=False
            )

            # Load dataset when a CSV file is uploaded
            csv_file.change(
                fn=load_dataset,
                inputs=csv_file,
                outputs=[
                    input_series_state,
                    dataset_info
                ]
            )

            # -------------------------
            # 2. Visualize time series
            # -------------------------

            gr.Markdown("### 2. Visualize Time Series")

            with gr.Row():

                start_index = gr.Number(
                    value=0,
                    label="Start index",
                    precision=0
                )

                end_index = gr.Number(
                    value=1000,
                    label="End index",
                    precision=0
                )

            plot_button = gr.Button(
                "Display Range",
                variant="primary"
            )

            time_series_plot = gr.Plot(
                label="Time Series"
            )

            # Plot the selected range
            plot_button.click(
                fn=plot_time_series,
                inputs=[
                    input_series_state,
                    start_index,
                    end_index
                ],
                outputs=time_series_plot
            )

            # -------------------------
            # 3. Train / Validation / Test split
            # -------------------------

            gr.Markdown(
                "### 3. Train / Validation / Test Split"
            )

            with gr.Row():

                train_end = gr.Number(
                    value=6000,
                    label="Train end index",
                    precision=0
                )

                val_end = gr.Number(
                    value=7000,
                    label="Validation end index",
                    precision=0
                )

            split_button = gr.Button(
                "Apply Split",
                variant="primary"
            )

            split_info = gr.Textbox(
                label="Split Info",
                interactive=False
            )

            # Split and normalize the dataset
            split_button.click(
                fn=split_and_normalize,
                inputs=[
                    input_series_state,
                    train_end,
                    val_end
                ],
                outputs=[
                    train_state,
                    val_state,
                    test_state,
                    train_norm_state,
                    val_norm_state,
                    test_norm_state,
                    train_mean_state,
                    train_std_state,
                    split_info
                ]
            )

        # ====================================================
        # LSTM
        # ====================================================

        with gr.Tab("LSTM"):
            gr.Markdown("## LSTM")
            gr.Markdown("### Hyperparameters")
            gr.Markdown(
                "Fixed training parameters: "
                "`num_epochs = 100`, `patience = 5`"
            )
            # Inputs
            lstm_hidden_size = gr.Number(
                value=128,
                label="Hidden size",
                info="Suggested for MG17: 8–128",
                precision=0
            )

            lstm_sequence_length = gr.Number(
                value=100,
                label="Sequence length",
                info="Suggested for MG17: 10–100",
                precision=0
            )

            lstm_batch_size = gr.Number(
                value=16,
                label="Batch size",
                info="Suggested for MG17: 16–128",
                precision=0
            )

            lstm_learning_rate = gr.Number(
                value=0.001,
                label="Learning rate",
                info="Suggested for MG17: 1e-4 – 1e-2"
            )

            # Button for running LSTM 
            run_lstm_button = gr.Button(
                "Run LSTM",
                variant="primary"
            )
            gr.Markdown(
                """
                **Note:** LSTM training may take a few minutes when running on CPU.
                """
            )

            # Outputs
            gr.Markdown(
                "### Results "
            )

            train_mse_output = gr.Number(
                label="Train MSE",
                interactive=False
            )

            val_mse_output = gr.Number(
                label="Validation MSE",
                interactive=False
            )

            test_mse_output = gr.Number(
                label="Test MSE",
                interactive=False
            )

            training_time_output = gr.Number(
                label="Training Time (s)",
                interactive=False
            )

            inference_time_output = gr.Number(
                label="Inference Time (s)",
                interactive=False
            )

            lstm_prediction_plot = gr.Plot(
                label="Actual vs Predicted — Test Set"
            )

            run_lstm_button.click(
                fn=run_lstm_from_gui,
                inputs=[
                    input_series_state,
                    train_state,
                    val_state,
                    test_state,
                    train_norm_state,
                    val_norm_state,
                    test_norm_state,
                    train_mean_state,
                    train_std_state,
                    lstm_hidden_size,
                    lstm_sequence_length,
                    lstm_batch_size,
                    lstm_learning_rate,
                ],
                outputs=[
                    train_mse_output,
                    val_mse_output,
                    test_mse_output,
                    training_time_output,
                    inference_time_output,
                    lstm_prediction_plot
                ]
            )

        # ====================================================
        # ESN
        # ====================================================
        with gr.Tab("ESN"):
            gr.Markdown("## ESN")

            gr.Markdown("### Hyperparameters")

            # Reservoir size
            esn_reservoir_size = gr.Number(
                value=100,
                label="Reservoir size",
                info="Suggested search range for MG17: 50–200",
                precision=0
            )

            # Spectral radius
            esn_spectral_radius = gr.Number(
                value=0.9,
                label="Spectral radius",
                info="Suggested search range for MG17: 0.70–0.99"
            )

            # Reservoir connectivity
            esn_reservoir_connectivity = gr.Number(
                value=0.05,
                label="Reservoir connectivity",
                info="Suggested search range for MG17: 0.01–0.10"
            )

            # Input scaling
            esn_input_scaling = gr.Number(
                value=0.5,
                label="Input scaling",
                info="Suggested search range for MG17: 0.10–1.00"
            )
            # Washout
            esn_washout = gr.Number(
                value=100,
                label="Washout",
                info="Suggested search range for MG17: 50–500",
                precision=0
            )

            # Leaky integration parameter
            esn_alpha = gr.Number(
                value=0.1,
                label="Alpha",
                info="Suggested search range for MG17: 0.05–1.00"
            )

            # Ridge regularization coefficient
            esn_ridge = gr.Number(
                value=1e-6,
                label="Ridge",
                info="Suggested search range for MG17: 1e-8–1e-4"
            )

            # Button for running LSTM 
            run_esn_button = gr.Button(
                "Run ESN",
                variant="primary"
            )

            # Outputs
            gr.Markdown(
                "### Results "
            )

            train_mse_esn_output = gr.Number(
                label="Train MSE",
                interactive=False
            )

            val_mse_esn_output = gr.Number(
                label="Validation MSE",
                interactive=False
            )

            test_mse_esn_output = gr.Number(
                label="Test MSE",
                interactive=False
            )

            training_time_esn_output = gr.Number(
                label="Training Time (s)",
                interactive=False
            )

            inference_time_esn_output = gr.Number(
                label="Inference Time (s)",
                interactive=False
            )

            esn_prediction_plot = gr.Plot(
                label="Actual vs Predicted — Test Set"
            )

            run_esn_button.click(
                fn=run_esn_from_gui,
                inputs=[
                    input_series_state,
                    train_state,
                    val_state,
                    test_state,
                    train_norm_state,
                    val_norm_state,
                    test_norm_state,
                    train_mean_state,
                    train_std_state,
                    esn_reservoir_size,
                    esn_spectral_radius,
                    esn_reservoir_connectivity,
                    esn_input_scaling,
                    esn_washout,
                    esn_alpha,
                    esn_ridge
                ],
                outputs=[
                    train_mse_esn_output,
                    val_mse_esn_output,
                    test_mse_esn_output,
                    training_time_esn_output,
                    inference_time_esn_output,
                    esn_prediction_plot
                ]
            )


# ============================================================
# LAUNCH APPLICATION
# ============================================================

demo.launch()