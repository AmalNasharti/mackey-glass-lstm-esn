from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from main.source_code import lstm_model
from main.source_code import esn_model
from main.source_code import utils
import json
import pandas as pd
import random
import numpy as np
from pathlib import Path

def sample_lstm_config(search_space):
    """
    Randomly sample one LSTM hyperparameter configuration.

    Parameters
    ----------
    search_space : dict
        Dictionary containing the search ranges or candidate values
        for the LSTM hyperparameters.

    Returns
    -------
    config : dict
        Randomly sampled LSTM configuration.
    """

    config = {
        "input_size": 1,

        "hidden_size": random.choice(
            search_space["hidden_size"]
        ),

        "output_size": 1,

        "sequence_length": random.choice(
            search_space["sequence_length"]
        ),

        "batch_size": random.choice(
            search_space["batch_size"]
        ),

        "learning_rate": float(10 ** random.uniform(
            np.log10(search_space["learning_rate"][0]),
            np.log10(search_space["learning_rate"][1])
        )),

        "num_epochs": random.choice(
            search_space["num_epochs"]
        ),

        "patience": random.choice(
            search_space["patience"]
        )
    }

    return config

def sample_esn_config(search_space):
    """
    Randomly sample one ESN hyperparameter configuration.

    Parameters
    ----------
    search_space : dict
        Dictionary containing the search ranges or candidate values
        for the ESN hyperparameters.

    Returns
    -------
    config : dict
        Randomly sampled ESN configuration.
    """

    config = {
        "reservoir_size": random.randint(
            search_space["reservoir_size"][0],
            search_space["reservoir_size"][1]
        ),

        "spectral_radius": random.uniform(
            search_space["spectral_radius"][0],
            search_space["spectral_radius"][1]
        ),

        "reservoir_connectivity": random.uniform(
            search_space["reservoir_connectivity"][0],
            search_space["reservoir_connectivity"][1]
        ),

        "input_scaling": random.uniform(
            search_space["input_scaling"][0],
            search_space["input_scaling"][1]
        ),

        "washout": random.choice(
            search_space["washout"]
        ),

        "alpha": random.uniform(
            search_space["alpha"][0],
            search_space["alpha"][1]
        ),

        "ridge": float(
            10 ** random.uniform(
                np.log10(search_space["ridge"][0]),
                np.log10(search_space["ridge"][1])
            )
        )
    }

    return config

def generate_configs(
    model,
    search_space_path,
    n_config,
    configs_path,
    search_seed=42
):
    """
    Generate random hyperparameter configurations and save them
    to a CSV file.

    Parameters
    ----------
    model : str
        Model for which configurations are generated.
        Must be either "lstm" or "esn".
    search_space_path : str or Path
        Path to the JSON file containing the hyperparameter search space.
    n_config : int
        Number of configurations to generate.
    configs_path : str or Path
        Path where the generated configurations are saved as a CSV file.
    search_seed : int, default=42
        Random seed used for reproducible configuration sampling.

    Returns
    -------
    configs_df : pd.DataFrame
        DataFrame containing all generated configurations.
    """

    # Load search space
    with open(search_space_path, "r") as file:
        search_space = json.load(file)

    # Set seed for reproducible hyperparameter sampling
    random.seed(search_seed)

    configs = []

    for trial in range(1, n_config + 1):

        if model == "lstm":
            config = sample_lstm_config(search_space)

        elif model == "esn":
            config = sample_esn_config(search_space)

        else:
            raise ValueError(
                'model must be either "lstm" or "esn".'
            )

        config["trial"] = trial
        configs.append(config)

    configs_df = pd.DataFrame(configs)

    # Put trial as the first column
    cols = ["trial"] + [
        col for col in configs_df.columns
        if col != "trial"
    ]

    configs_df = configs_df[cols]

    # Save configurations
    configs_df.to_csv(
        configs_path,
        index=False
    )

    return configs_df

def val_loss_lstm(train_norm, val_norm, config, device):
    """
    Train an LSTM using a given hyperparameter configuration and
    return the minimum validation loss reached during training.

    Parameters
    ----------
    train_norm : pd.Series
        Normalized training time series.
    val_norm : pd.Series
        Normalized validation time series.
    config : pd.Series
        Row of the hyperparameter configuration DataFrame containing
        the LSTM parameters.
    device : torch.device
        Device used for model training.

    Returns
    -------
    min_loss : float
        Minimum validation loss reached during training.
    """

    # Create sliding-window data
    X_train, y_train = lstm_model.create_sequences(
        train_norm,
        int(config["sequence_length"])
    )

    X_val, y_val = lstm_model.create_sequences(
        val_norm,
        int(config["sequence_length"])
    )

    # Create datasets and loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False
    )

    # Initialize model
    model = lstm_model.TimeSeriesLSTM(
        input_size=int(config["input_size"]),
        hidden_size=int(config["hidden_size"]),
        output_size=int(config["output_size"])
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"])
    )

    _, val_losses = lstm_model.train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        int(config["num_epochs"]),
        int(config["patience"]),
        save_weights=False,
        verbose=False,
    )

    return min(val_losses)

def val_loss_esn(train_norm, val_norm, config, device):
    """
    Train an Echo State Network with a given hyperparameter configuration
    and return its validation loss.

    The ESN is fitted using the normalized training set and evaluated on
    the normalized validation set. The validation performance is measured
    using the mean squared error (MSE).

    Parameters
    ----------
    train_norm : pd.Series
        Normalized training time series.
    val_norm : pd.Series
        Normalized validation time series.
    config : pd.Series
        ESN hyperparameter configuration containing reservoir size,
        spectral radius, reservoir connectivity, input scaling, leaking
        parameter, washout length, and ridge regularization coefficient.
    device : torch.device
        Device used for model fitting and inference.

    Returns
    -------
    val_loss : float
        Mean squared error obtained on the validation set.
    """

    # Create one-step-ahead input-target pairs
    X_train, y_train = esn_model.create_input_target(train_norm)
    X_val, y_val = esn_model.create_input_target(val_norm)

    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    # Initialize model
    model = esn_model.EchoStateNetwork(
        reservoir_size=int(config["reservoir_size"]),
        spectral_radius=float(config["spectral_radius"]),
        reservoir_connectivity=float(config["reservoir_connectivity"]),
        input_scaling=float(config["input_scaling"]),
        alpha=float(config["alpha"]),
        washout=int(config["washout"]),
        ridge=float(config["ridge"])
    ).to(device)

    # Fit model
    model.fit(X_train, y_train)

    # Validation inference
    val_pred = model(X_val)

    # Compute validation loss
    val_loss = utils.mse(
        y_val,
        val_pred
    )

    return val_loss

def evaluate_config(
    model,
    train_norm,
    val_norm,
    config,
    device,
    seeds
):
    """
    Evaluate one configuration over multiple random seeds.

    Parameters
    ----------
    model : str
        Model for which configurations are generated.
        Must be either "lstm" or "esn".
    train_norm : pd.Series
        Normalized training time series.
    val_norm : pd.Series
        Normalized validation time series.
    config : pd.Series
        Row of the hyperparameter configuration
    device : torch.device
        Device used for model training.
    seeds : iterable
        Random seeds used for repeated evaluations.

    Returns
    -------
    mean_val_loss : float
        Mean validation loss across all runs.
    std_val_loss : float
        Standard deviation of the validation losses across all runs.
    """

    losses = []

    for seed in seeds:

        print(f"Calculating validation loss with seed {seed}...")

        utils.set_seed(seed)

        if model == "lstm":
            loss = val_loss_lstm(train_norm, val_norm, config, device)

        elif model == "esn":
            loss = val_loss_esn(train_norm, val_norm, config, device)

        else:
            raise ValueError(
                'model must be either "lstm" or "esn".'
            )    

        losses.append(loss)

    mean_val_loss = np.mean(losses)
    std_val_loss = np.std(losses)

    return mean_val_loss, std_val_loss

def random_search(
    model,
    train_norm,
    val_norm,
    device,
    configs,
    seeds,
    results_path,
    best_results_path,
    restart=False
):
    """
    Perform random hyperparameter search.

    Each configuration is evaluated over multiple random seeds.
    Results are saved immediately after each completed trial so that
    the search can resume after an interruption.

    The best configuration found so far is also saved separately.

    Parameters
    ----------
    model : str
        Model for which configurations are generated.
        Must be either "lstm" or "esn".
    train_norm : pd.Series
        Normalized training time series.
    val_norm : pd.Series
        Normalized validation time series.
    device : torch.device
        Device used for model training.
    configs : pd.DataFrame
        DataFrame containing the hyperparameter configurations.
        Each row corresponds to one trial.
    seeds : iterable
        Random seeds used to evaluate each configuration.
    results_path : str or Path
        CSV file where all completed trials are stored.
    best_results_path : str or Path
        CSV file where the best result found so far is stored.
    restart : bool, default=False
        If True, delete previous results and restart the search
        from the first trial. If False, skip already completed trials.

    Returns
    -------
    best_config : pd.Series
        Hyperparameter configuration with the lowest mean validation loss.
    best_val_loss : float
        Lowest mean validation loss obtained.
    """

    results_path = Path(results_path)
    best_results_path = Path(best_results_path)

    configs = configs.copy()

    # Add trial number if not already present
    if "trial" not in configs.columns:
        configs.insert(
            0,
            "trial",
            range(1, len(configs) + 1)
        )

    n_trials = len(configs)

    # ---------------------------------------------------------
    # Restart search if requested
    # ---------------------------------------------------------

    if restart:

        if results_path.exists():
            results_path.unlink()

        if best_results_path.exists():
            best_results_path.unlink()

        print("Previous results deleted.")
        print("Starting hyperparameter search from scratch.\n")

    # ---------------------------------------------------------
    # Load previous results
    # ---------------------------------------------------------

    if results_path.exists():

        previous_results = pd.read_csv(results_path)

        completed_trials = set(
            previous_results["trial"].astype(int)
        )

        print(
            f"{len(completed_trials)} previously completed "
            f"trial(s) found."
        )

        # Recover current best result
        best_index = previous_results[
            "mean_val_loss"
        ].idxmin()

        best_result = previous_results.loc[best_index]

        best_val_loss = float(
            best_result["mean_val_loss"]
        )

        config_columns = [
            column
            for column in configs.columns
            if column != "trial"
        ]

        best_config = best_result[
            config_columns
        ].copy()

    else:

        completed_trials = set()
        best_val_loss = float("inf")
        best_config = None

    # ---------------------------------------------------------
    # Evaluate configurations
    # ---------------------------------------------------------

    for _, config in configs.iterrows():

        trial = int(config["trial"])

        # Skip completed trials
        if trial in completed_trials:

            print(
                f"Trial {trial:3d}/{n_trials} "
                "already completed. Skipping..."
            )

            continue

        print(f"\nTrial {trial:3d}/{n_trials}")

        # Evaluate configuration over multiple seeds
        mean_val_loss, std_val_loss = evaluate_config(
            model,
            train_norm,
            val_norm,
            config,
            device,
            seeds
        )

        print(
            f"Mean val loss: {mean_val_loss:.8e} | "
            f"Std: {std_val_loss:.2e}"
        )

        # -----------------------------------------------------
        # Prepare result
        # -----------------------------------------------------

        result = config.to_dict()

        result["mean_val_loss"] = mean_val_loss
        result["std_val_loss"] = std_val_loss

        result_df = pd.DataFrame([result])

        # -----------------------------------------------------
        # Save trial immediately
        # -----------------------------------------------------

        file_exists = results_path.exists()

        result_df.to_csv(
            results_path,
            mode="a",
            header=not file_exists,
            index=False
        )

        completed_trials.add(trial)

        print("Trial result saved.")

        # -----------------------------------------------------
        # Update best result
        # -----------------------------------------------------

        if mean_val_loss < best_val_loss:

            best_val_loss = mean_val_loss

            config_columns = [
                column
                for column in configs.columns
                if column != "trial"
            ]

            best_config = config[
                config_columns
            ].copy()

            # Overwrite best-result file
            result_df.to_csv(
                best_results_path,
                index=False
            )

            print("New best configuration found and saved.")

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    return best_config, best_val_loss