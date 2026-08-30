# Mackey-Glass Time-Series Prediction with LSTM and ESN

## Overview

This repository implements two recurrent approaches for **one-step-ahead
time-series prediction**:

-   **Long Short-Term Memory (LSTM)**
-   **Echo State Network (ESN)**

The project has currently been tested on the **Mackey-Glass chaotic time
series with delay parameter τ = 17 (MG17)**. Performance on other time
series has not yet been validated and is planned as future work.

The repository provides:

-   a command-line experiment pipeline;
-   random-search hyperparameter tuning;
-   a Gradio graphical user interface;
-   training and loading of pretrained models;
-   saving/loading of model configurations and PyTorch weights;
-   evaluation using train, validation, and test MSE.

## Repository Structure

``` text
.
├── app.py                  # Gradio graphical interface
├── main.py                 # Main LSTM/ESN experiment
├── fine_tuning.py          # Hyperparameter tuning entry point
├── requirements.txt
│
├── main/
│   ├── input/              # Dataset and experiment configuration
│   ├── output/             # Metrics and generated plots
│   ├── weights/            # Saved LSTM and ESN weights
│   └── source_code/
│       ├── lstm_model.py
│       ├── esn_model.py
│       └── utils.py
│
├── fine_tuning/
│   ├── input/
│   │   └── search_spaces/  # LSTM and ESN random-search spaces
│   ├── output/             # Tuning results
│   └── source_code/
│       └── tuning.py
│
└── gui/
    ├── config/             # Configurations used by the GUI
    ├── weights/            # Model weights used by the GUI
    ├── fine_tuning/        # GUI tuning results and best configurations
    └── utils.py
```

## Input Data

The input must be a CSV file containing the time series in a column
named:

``` text
value
```

Do not include a time column. Samples must be ordered in time and
equally spaced (for example, one observation every second, minute, or
hour).

The default MG17 experiment uses a chronological split of:

-   **6000 samples (60%)** for training;
-   **1000 samples (10%)** for validation;
-   **3000 samples (30%)** for testing.

The relatively large test set is used to provide a robust evaluation of
generalization on unseen data.

## Installation

Create and activate a Python virtual environment, then install the
dependencies:

``` bash
python -m pip install -r requirements.txt
```

PyTorch can run on CPU or CUDA when a compatible GPU installation is
available.

## Running the Main Experiment

Model, data-split, and reproducibility settings are defined in:

``` text
main/input/config.json
```

Run the complete experiment with:

``` bash
python main.py
```

The script:

1.  loads and splits the time series;
2.  normalizes all splits using training-set statistics;
3.  runs the LSTM and ESN;
4.  returns predictions to the original scale;
5.  computes train, validation, and test MSE;
6.  saves metrics and plots in `main/output/`.

The flags `LOAD_PRETRAINED_LSTM` and `LOAD_PRETRAINED_ESN` in `main.py`
control whether each model is trained or loaded from its saved `.pt`
weights.

## Graphical User Interface

Start the Gradio interface with:

``` bash
python app.py
```

The GUI contains three main sections:

-   **Data** --- upload and visualize the dataset and define the
    chronological train/validation/test split.
-   **LSTM** --- train a new LSTM or use a pretrained model, inspect
    metrics and predictions, download configuration and weights, and run
    random-search tuning.
-   **ESN** --- equivalent workflow for the Echo State Network.

When training a new model, hyperparameters can be entered manually or
pre-filled from a JSON configuration. Pretrained models require the
corresponding JSON configuration and `.pt` weights.

## Hyperparameter Tuning

Random-search tuning can be launched from the GUI or through:

``` bash
python fine_tuning.py
```

Search spaces are defined in:

``` text
fine_tuning/input/search_spaces/
```

Each configuration can be evaluated over multiple random seeds.
Configurations are ranked using their **mean validation loss**.

The tuning procedure returns the best hyperparameter configuration. To
obtain final model weights for that configuration, train the
corresponding model again using the selected configuration.

## Reproducibility

The project uses a configurable random seed for Python/NumPy/PyTorch
operations. The seed is reset before running each model so that LSTM and
ESN results do not depend on the order in which the models are executed.

## Outputs

The main experiment generates:

-   `results.csv` with train, validation, and test MSE plus execution
    times;
-   the input time-series plot;
-   LSTM training/validation loss plot when training is performed;
-   LSTM test predictions;
-   ESN test predictions;
-   saved `.pt` model weights.

The GUI additionally allows the current model configuration, weights,
and best configurations found by random search to be downloaded.
