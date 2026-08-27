import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

def split_data(series, train_end, val_end):
    """
    Chronologically split a time series into training, validation,
    and test sets.

    Parameters
    ----------
    series : pd.Series
        Complete time series.
    train_end : int
        Index marking the end of the training set.
    val_end : int
        Index marking the end of the validation set.

    Returns
    -------
    train, val, test : pd.Series
        Training, validation, and test subsets.
    """

    train = series[:train_end]
    val = series[train_end:val_end]
    test = series[val_end:]

    return train, val, test

def zscore_normalize(train, val, test):
    """
    Normalize training, validation, and test sets using z-score
    standardization based only on training-set statistics.

    Parameters
    ----------
    train, val, test : pd.Series
        Original training, validation, and test subsets.

    Returns
    -------
    train_norm, val_norm, test_norm : pd.Series
        Standardized datasets.
    train_mean : float
        Mean of the training set.
    train_std : float
        Standard deviation of the training set.
    """

    train_mean = train.mean()
    train_std = train.std()

    train_norm = (train - train_mean) / train_std
    val_norm = (val - train_mean) / train_std
    test_norm = (test - train_mean) / train_std

    return train_norm, val_norm, test_norm, train_mean, train_std

def inverse_zscore(data, mean, std):
    """
    Transform standardized data back to the original scale.

    Parameters
    ----------
    data : torch.Tensor or array-like
        Data in the standardized scale.
    mean : float
        Mean used for z-score normalization.
    std : float
        Standard deviation used for z-score normalization.

    Returns
    -------
    data_orig : same type as input when possible
        Data transformed back to the original scale.
    """

    return data * std + mean

def mse(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE) between actual and predicted values.

    If the input data are not already PyTorch tensors, they are automatically
    converted to tensors with dtype torch.float32.

    Parameters
    ----------
    y_true : torch.Tensor or array-like
        Actual target values.
    y_pred : torch.Tensor or array-like
        Predicted values.

    Returns
    -------
    mse : torch.Tensor
        Mean Squared Error between the actual and predicted values.
    """

    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.float32)

    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    criterion = nn.MSELoss()

    return criterion(y_pred, y_true).item()

def save_original_time_series(data, save_path, n_points=1000):
    """
    Save the first n points of a time series.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Time-series data to plot.
    save_path: str
        Path where to save the plot.
    n_points : int
        Number of initial time steps to display.
    """

    plt.figure(figsize=(12, 5))

    plt.plot(data[:n_points])

    plt.title("Mackey-Glass Time Series")
    plt.xlabel("Time step")
    plt.ylabel("Mackey-Glass value")

    plt.xlim(0, n_points - 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
def save_losses(train_losses, val_losses, save_path):
    """
    Save the training and validation loss over the training epochs.

    Parameters
    ----------
    train_losses : list
        Training loss recorded at each epoch.
    val_losses : list
        Validation loss recorded at each epoch.
    save_path: str
        Path where to save the plot.
    """

    plt.figure(figsize=(12, 6))

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.title("Training and Validation Loss over Epochs (LSTM)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

def save_predictions(y_actual, y_pred, model_name, save_path, n_points=1000):
    """
    Save actual and predicted values of the Mackey-Glass time series.

    Parameters
    ----------
    y_actual : array-like
        Actual target values.
    y_pred : array-like
        Model predictions.
    model_name : str
        Name of the model used for the predictions.
    save_path: str
        Path where to save the plot.
    n_points : int
        Number of time steps to display.
    """

    plt.figure(figsize=(12, 5))

    plt.plot(
        y_actual[:n_points],
        label="Actual",
        linewidth=2
    )

    plt.plot(
        y_pred[:n_points],
        label=f"{model_name} Prediction",
        linestyle="--",
        linewidth=2
    )

    plt.title(f"Mackey-Glass: Actual vs {model_name} Prediction")
    plt.xlabel("Time step")
    plt.ylabel("Mackey-Glass value")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, n_points - 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

def clear_output_directory(output_dir):
    """
    Remove all files from the specified output directory.

    Parameters
    ----------
    output_dir : str or Path
        Path to the output directory to clear.
    """

    output_dir = Path(output_dir)

    # Remove all files in the output directory
    for file in output_dir.iterdir():
        if file.is_file():
            file.unlink()