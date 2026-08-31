import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def create_input_target(series):
    """
    Create input-target pairs for one-step-ahead time-series prediction.

    The input contains the time series up to the penultimate value, while 
    the target is the same series shifted forward by one time step.

    Parameters
    ----------
    series : pd.Series
        Univariate time series.

    Returns
    -------
    X : torch.Tensor
        Input values with shape (N, 1), where N is the number of
        input-target pairs and 1 is the number of input features.
    y : torch.Tensor
        Target values with shape (N, 1), corresponding to the
        next time step of each input value.
    """
    series = series.to_numpy()
    
    X = torch.tensor(
        series[:-1],
        dtype=torch.float32
    ).unsqueeze(1)

    y = torch.tensor(
        series[1:],
        dtype=torch.float32
    ).unsqueeze(1)

    return X, y

def create_esn_data(train_norm, val_norm, test_norm):
    """
    Create one-step-ahead input-target pairs for ESN training and evaluation.

    Each normalized dataset is transformed so that the value at time t
    is used as input to predict the value at time t+1.

    Parameters
    ----------
    train_norm : pd.Series
        Normalized training time series.
    val_norm : pd.Series
        Normalized validation time series.
    test_norm : pd.Series
        Normalized test time series.

    Returns
    -------
    X_train, y_train : torch.Tensor
        Training inputs and targets with shape (N_train, 1).
    X_val, y_val : torch.Tensor
        Validation inputs and targets with shape (N_val, 1).
    X_test, y_test : torch.Tensor
        Test inputs and targets with shape (N_test, 1).
    """
    X_train, y_train = create_input_target(train_norm)
    X_val, y_val = create_input_target(val_norm)
    X_test, y_test = create_input_target(test_norm)

    return X_train, y_train, X_val, y_val, X_test, y_test

class EchoStateNetwork(nn.Module):

    """
    Echo State Network for univariate one-step-ahead time-series prediction.

    The input and recurrent reservoir weights are randomly initialized
    and kept fixed. Only the output weights are fitted using ridge regression.

    Parameters
    ----------
    reservoir_size : int
        Number of recurrent units in the reservoir.
    spectral_radius : float
        Desired spectral radius of the recurrent reservoir weight matrix.
    reservoir_connectivity : float
        Fraction of non-zero recurrent connections in the reservoir.
    input_scaling : float
        Scaling factor applied to the input weights.
    alpha : float
        Leaky integration parameter controlling the contribution of the
        previous reservoir state.
    washout : int
        Number of initial reservoir states discarded before fitting
        the output weights.
    ridge : float
        Ridge regularization coefficient used to fit the output weights.
    """

    def __init__(self, reservoir_size, spectral_radius, reservoir_connectivity, input_scaling, alpha, washout, ridge):
        super().__init__()

        self.reservoir_size = reservoir_size
        self.alpha = alpha
        self.washout = washout
        self.ridge = ridge

        # Random reservoir weights in [-1, 1]
        W_res = 2 * torch.rand(
            reservoir_size,
            reservoir_size,
            dtype=torch.float32
        ) - 1

        # Random connectivity mask
        mask = torch.rand(
            reservoir_size,
            reservoir_size
        ) <= reservoir_connectivity

        # Sparsity requirement of reservoir weights
        W_res = W_res * mask

        # Compute current spectral radius
        eigenvalues = torch.linalg.eigvals(W_res)
        current_radius = torch.max(torch.abs(eigenvalues))

        # Rescale reservoir weights
        W_res = W_res * (spectral_radius / current_radius)

        # Fixed weights: registered as buffers, not trainable parameters
        self.register_buffer("W_res", W_res)

        # Input weights
        W_in = (
            2 * torch.rand(
                reservoir_size,
                1,
                dtype=torch.float32
            ) - 1
        ) * input_scaling

        self.register_buffer("W_in", W_in)

        # Output weights are computed during training
        self.register_buffer(
            "W_out",
            torch.zeros(
                reservoir_size,
                1,
                dtype=torch.float32
            )
        )

    def run_reservoir(self, input_data):
        """
        Propagate an input sequence through the reservoir.

        Parameters
        ----------
        input_data : torch.Tensor
            Input sequence with shape (T, 1).

        Returns
        -------
        reservoir_states : torch.Tensor
            Reservoir states with shape (T, reservoir_size).
        """

        reservoir_states = torch.zeros(
            len(input_data),
            self.reservoir_size,
            device=input_data.device
        )

        # Leaky integration
        for t in range(1, len(input_data)):
            reservoir_states[t] = (
                self.alpha * reservoir_states[t - 1]
                + (1 - self.alpha) * torch.tanh(
                    self.W_res @ reservoir_states[t - 1]
                    + (self.W_in @ input_data[t]).squeeze()
                )
            )

        return reservoir_states

    def fit(self, input_data, target_data):
        """
        Fit the output weights using the reservoir states.

        Parameters
        ----------
        input_data : torch.Tensor
            Training input with shape (T, 1).
        target_data : torch.Tensor
            Training targets with shape (T, 1).
        """
        reservoir_states = self.run_reservoir(input_data)

        # Discard the washout phase
        X = reservoir_states[self.washout:]
        Y = target_data[self.washout:]

        # Identity matrix for ridge regularization
        I = torch.eye(
            X.shape[1],
            device=X.device,
            dtype=X.dtype
        )

        # Compute output weights using ridge regression
        self.W_out = torch.linalg.solve(
            X.T @ X + self.ridge * I,
            X.T @ Y
        )

    def forward(self, input_data):
        """
        Generate predictions for an input sequence.

        Parameters
        ----------
        input_data : torch.Tensor
            Input sequence with shape (T, 1).

        Returns
        -------
        predictions : torch.Tensor
            Predicted values with shape (T, 1).
        """
        reservoir_states = self.run_reservoir(input_data)

        predictions = reservoir_states @ self.W_out

        return predictions

def run_esn(train_norm, val_norm, test_norm, config, device, weights_path, load_pretrained=False):
    """
    Run the complete ESN training and inference pipeline.

    The function creates one-step-ahead input-target pairs, initializes
    the Echo State Network, optionally fits the output weights or loads
    pretrained weights, and generates predictions for the training,
    validation, and test sets.

    Parameters
    ----------
    train_norm : pd.Series
        Normalized training time series.
    val_norm : pd.Series
        Normalized validation time series.
    test_norm : pd.Series
        Normalized test time series.
    config : dict
        ESN configuration containing the model hyperparameters.
    device : torch.device
        Device used for model fitting and inference.
    weights_path : str or Path
        Path used to save or load the ESN weights.
    load_pretrained : bool, default=False
        If True, load previously saved weights and skip model fitting.

    Returns
    -------
    results : dict
        Dictionary containing:
        - model : fitted or loaded Echo State Network.
        - train_pred : training predictions in the normalized scale.
        - val_pred : validation predictions in the normalized scale.
        - test_pred : test predictions in the normalized scale.
        - training_time : fitting time in seconds, or None when pretrained
          weights are used.
        - inference_time : test-set inference time in seconds.
    """
    # Create one-step-ahead input-target pairs
    X_train, y_train, X_val, y_val, X_test, y_test = create_esn_data(train_norm, val_norm, test_norm)

    # Model initialization
    model = EchoStateNetwork(
        reservoir_size=config["reservoir_size"],
        spectral_radius=config["spectral_radius"],
        reservoir_connectivity=config["reservoir_connectivity"],
        input_scaling=config["input_scaling"],
        alpha = config["alpha"],
        washout = config["washout"],
        ridge=config["ridge"]
    ).to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    X_test = X_test.to(device)

    if load_pretrained:
        #load presaved weights from our best run:
        model.load_state_dict(torch.load(weights_path))
        training_time = None
    else:
        # Training model
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        model.fit(X_train, y_train)

        if device.type == "cuda":
            torch.cuda.synchronize()
        training_time = time.perf_counter() - start_time

        torch.save(model.state_dict(), weights_path)

    # Inference
    train_pred = model(X_train)
    val_pred = model(X_val)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    test_pred = model(X_test)

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - start_time

    return {
        "model": model,
        "train_pred": train_pred,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "training_time": training_time,
        "inference_time": inference_time
    }