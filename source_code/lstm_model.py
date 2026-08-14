import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import copy
import time

def create_sequences(data, seq_len):
    """
    Create input-target pairs from a univariate time series using
    a sliding window.

    Parameters
    ----------
    data : pd.DataFrame
        Input time series.
    seq_len : int
        Number of time steps in each input sequence.

    Returns
    -------
    X : torch.Tensor
        Shape (N, seq_len, 1), where N is the number of generated
        sequences and 1 is the number of input features.

    y : torch.Tensor
        Shape (N, 1), where N is the number of targets and 1 is the
        number of predicted features.
    """
    X = []
    y = []

    data = data.values
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])

    X = np.expand_dims(np.array(X), axis=-1)
    y = np.expand_dims(np.array(y), axis=-1)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

def create_lstm_data(train_norm, val_norm, test_norm, seq_len):
    """
    Convert normalized time-series splits into sliding-window
    input-target pairs for the LSTM.

    Parameters
    ----------
    train_norm, val_norm, test_norm : pd.Series
        Normalized time-series subsets.
    seq_len : int
        Number of time steps in each input sequence.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : torch.Tensor
        Sliding-window inputs and corresponding one-step-ahead targets.
    """
    X_train, y_train = create_sequences(train_norm, seq_len)
    X_val, y_val = create_sequences(val_norm, seq_len)
    X_test, y_test = create_sequences(test_norm, seq_len)

    return X_train, y_train, X_val, y_val, X_test, y_test

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    """
    Create PyTorch datasets and data loaders for training,
    validation, testing.

    Returns
    -------
    train_loader : DataLoader
        Training loader with shuffled samples.
    val_loader : DataLoader
        Validation loader with samples in chronological order.
    test_loader : DataLoader
        Test loader with samples in chronological order.
    train_loader_plot : DataLoader
        Training loader without shuffling, used to generate
        chronologically ordered predictions.
    """

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    train_loader_plot = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, train_loader_plot

class TimeSeriesLSTM(nn.Module):
    """
    Single-layer LSTM model for one-step-ahead time-series prediction.

    The LSTM processes the input sequence and produces a hidden
    representation at each time step. The output corresponding to the
    last time step is passed to a linear layer to predict the next value.

    Parameters
    ----------
    input_size : int
        Number of input features at each time step.
    hidden_size : int
        Number of hidden units in the LSTM layer.
    output_size : int
        Number of values predicted by the model.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)

        # Keep only the output at the last time step
        out = out[:, -1, :]

        # Predict the next value
        return self.fc(out)

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    patience
):
    """
    Train the model using mini-batch gradient descent and evaluate it
    on the validation set after each epoch.

    Early stopping is applied when the validation loss does not improve
    for a specified number of consecutive epochs.

    Returns
    -------
    train_losses : list
        Average training loss for each epoch.
    val_losses : list
        Average validation loss for each epoch.
    """
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    train_losses = []
    val_losses = []

    print("Starting Training...")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():

            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f}"
        )
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(best_state)
                break

    print("Training complete.")

    return train_losses, val_losses

def get_predictions(model, loader, device):
    """
    Generate model predictions and collect the corresponding target values.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    loader : DataLoader
        DataLoader containing the input-target pairs.
    device : torch.device
        Device used for computation.

    Returns
    -------
    predictions : np.ndarray
        Model predictions with shape (N, 1).
    actuals : np.ndarray
        Corresponding target values with shape (N, 1).
    """

    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            pred = model(X)

            predictions.append(pred.cpu().numpy())
            actuals.append(y.numpy())

    return np.vstack(predictions), np.vstack(actuals)

def run_lstm(train_norm, val_norm, test_norm, config, device):
    """
    Run the complete LSTM training and inference pipeline.

    The function creates sliding-window input-target pairs, builds the
    DataLoaders, initializes and trains the LSTM model, and generates
    predictions for the training, validation, and test sets.

    Parameters
    ----------
    train_norm : pd.Series
        Normalized training time series.
    val_norm : pd.Series
        Normalized validation time series.
    test_norm : pd.Series
        Normalized test time series.
    config : dict
        LSTM configuration containing the model and training
        hyperparameters.
    device : torch.device
        Device used for model training and inference.

    Returns
    -------
    results : dict
        Dictionary containing:
        - model: trained LSTM model.
        - train_pred: training predictions in the normalized scale.
        - val_pred: validation predictions in the normalized scale.
        - test_pred: test predictions in the normalized scale.
        - train_losses: training loss recorded at each epoch.
        - val_losses: validation loss recorded at each epoch.
    """
    # Create sliding window data
    X_train, y_train, X_val, y_val, X_test, y_test = create_lstm_data(train_norm, val_norm, test_norm, config["sequence_length"])

    # Create DataLoaders
    train_loader, val_loader, test_loader, train_loader_plot = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, config["batch_size"])

    # Model initialization
    model = TimeSeriesLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"]
    ).to(device)

    # Loss
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )

    # Training model
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, device, config["num_epochs"], config["patience"]) 

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_time = time.perf_counter() - start_time

    # Inference
    y_train_pred,_ = get_predictions(model, train_loader_plot, device)
    y_val_pred,_ = get_predictions(model, val_loader, device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    y_test_pred,_ = get_predictions(model, test_loader, device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - start_time

    return {
        "model": model,
        "train_pred": y_train_pred,
        "val_pred": y_val_pred,
        "test_pred": y_test_pred,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "training_time": training_time,
        "inference_time": inference_time
    }
        