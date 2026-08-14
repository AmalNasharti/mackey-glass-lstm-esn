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