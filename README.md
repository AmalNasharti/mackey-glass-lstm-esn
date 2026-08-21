# LSTM and Echo State Network for Mackey-Glass Time-Series Prediction

## Overview

This project implements and compares two recurrent neural network approaches for one-step-ahead prediction of the Mackey-Glass time series:

- Long Short-Term Memory network (LSTM)
- Echo State Network (ESN)

### `source_code/`

Contains the Python source code and experiment configuration.

- `main.py`: main entry point of the project. It loads the data and configuration, performs common preprocessing, runs both models, evaluates their performance, and saves the results.
- `lstm_model.py`: contains the LSTM architecture and LSTM-specific functions for sequence generation, DataLoader creation, training, and inference.
- `esn_model.py`: contains the Echo State Network architecture and ESN-specific functions for sequential input-target generation, reservoir computation, fitting, and inference.
- `utils.py`: contains common utilities shared by both models, including data splitting, normalization, inverse transformation, evaluation metrics, and functions to save plots.
- `config.json`: contains the model hyperparameters, data-splitting parameters, and random seed used for reproducibility.

### `input/`

Contains the Mackey-Glass time-series dataset used by the models.

### `output/`

Contains the results automatically generated when the experiment is executed, including:

- model performance metrics;
- original time-series visualization;
- LSTM training and validation loss;
- LSTM predictions;
- ESN predictions.

## Configuration Parameters

The experiment parameters are defined in `config.json`. The configuration file contains the random seed, dataset split indices, and model-specific hyperparameters for the LSTM and ESN.

### General Parameters

- **`seed`**: Random seed used to improve the reproducibility of the experiments by controlling the random initialization of the models and other stochastic operations.

### Data Parameters

- **`train_end`**: Index marking the end of the training set. All observations before this index are assigned to the training set.
- **`val_end`**: Index marking the end of the validation set. Observations between `train_end` and `val_end` form the validation set, while the remaining observations form the test set.

### LSTM Parameters

- **`input_size`**: Number of input features at each time step. It is set to 1 because the Mackey-Glass time series is univariate.
- **`hidden_size`**: Number of hidden units in the LSTM layer, corresponding to the dimension of the hidden and cell states.
- **`output_size`**: Number of values predicted by the network. It is set to 1 for one-step-ahead prediction of a univariate time series.
- **`sequence_length`**: Number of consecutive past time steps included in each sliding-window input sequence.
- **`batch_size`**: Number of input-target pairs processed together before updating the model parameters.
- **`learning_rate`**: Step size used by the optimizer when updating the trainable parameters of the LSTM.
- **`num_epochs`**: Maximum number of complete passes through the training dataset.
- **`patience`**: Number of consecutive epochs without improvement in the validation loss allowed before early stopping terminates training.

### ESN Parameters

- **`reservoir_size`**: Number of recurrent units in the reservoir and therefore the dimension of the reservoir state.
- **`spectral_radius`**: Target spectral radius of the recurrent reservoir weight matrix, defined as the largest absolute eigenvalue of the matrix after rescaling.
- **`reservoir_connectivity`**: Probability that a connection between two reservoir units is present. Connections not selected according to this probability are assigned a weight of zero.
- **`input_scaling`**: Controls the magnitude of the input weights. Input weights are sampled from the uniform distribution `U(-input_scaling, input_scaling)`.
- **`washout`**: Number of initial reservoir states discarded before fitting the readout weights, allowing the influence of the initial reservoir state to decrease.
- **`alpha`**: Leaky-integration parameter controlling the contribution of the previous reservoir state relative to the newly computed state. The implementation uses:

  `x(t) = alpha * x(t-1) + (1 - alpha) * x_new(t)`

- **`ridge`**: Ridge regularization coefficient used when fitting the output weights. It penalizes large readout weights and improves the numerical stability of the regression.