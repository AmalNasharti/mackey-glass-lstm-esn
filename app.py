import gradio as gr
from gui import utils

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
                fn=utils.load_dataset,
                inputs=csv_file,
                outputs=[
                    input_series_state,
                    dataset_info
                ]
            )

            # Save uploaded dataset inside gui/input
            save_dataset_button = gr.Button(
                "Save Dataset",
                variant="primary"
            )

            save_dataset_message = gr.Markdown()

            # Save dataset when the button is pressed
            save_dataset_button.click(
                fn=utils.save_dataset,
                inputs=csv_file,
                outputs=save_dataset_message
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
                fn=utils.plot_time_series,
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
                fn=utils.split_and_normalize,
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
                fn=utils.run_lstm_from_gui,
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

            # Hyperparameter Tuning
            with gr.Accordion(
                "Automatic Hyperparameter Tuning",
                open=False
            ):

                gr.Markdown(
                    """
                    Random search evaluates multiple LSTM hyperparameter configurations
                    and selects the one with the lowest mean validation MSE.

                    **Warning:** LSTM hyperparameter tuning may take several hours when
                    running on CPU.

                    For a quick test, it is recommended to use:
                    - `trials = 5`
                    - `seeds = 1`

                    A more extensive search performed with `trials = 30` and `seeds = 5`
                    obtained the following best configuration:

                    - Hidden size: `128`
                    - Sequence length: `100`
                    - Batch size: `16`
                    - Learning rate: `0.0013258369731779726`
                    - Number of epochs: `200`
                    - Patience: `15`
                    - Mean validation MSE: `6.0661e-07`
                    - Standard deviation: `2.0493e-07`
                    """
                )

                with gr.Row():

                    lstm_n_trials = gr.Number(
                        value=5,
                        label="Number of trials",
                        precision=0
                    )

                    lstm_n_seeds = gr.Number(
                        value=1,
                        label="Number of seeds",
                        info="Number of training runs with different random seeds used to evaluate each hyperparameter configuration.",
                        precision=0
                    )

                start_lstm_tuning_button = gr.Button(
                    "Start Random Search",
                    variant="primary"
                )

                best_lstm_config_output = gr.JSON(
                    label="Best LSTM Configuration"
                )

                start_lstm_tuning_button.click(
                    fn=utils.run_lstm_tuning_gui,
                    inputs=[
                        input_series_state,
                        train_norm_state,
                        val_norm_state,
                        lstm_n_trials,
                        lstm_n_seeds
                    ],
                    outputs=[
                        best_lstm_config_output
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
                fn=utils.run_esn_from_gui,
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

            # Hyperparameter Tuning
            with gr.Accordion(
                "Automatic Hyperparameter Tuning",
                open=False
            ):

                gr.Markdown(
                    """
                    Random search evaluates different ESN hyperparameter configurations
                    and returns the configuration with the best validation performance.

                    ESN tuning is generally faster than LSTM tuning.

                    Suggested values:
                    - `trials = 150`
                    - `runs per configuration = 10`
                    """
                )

                with gr.Row():

                    esn_n_trials = gr.Number(
                        value=150,
                        label="Number of trials",
                        precision=0
                    )

                    esn_n_seeds = gr.Number(
                        value=10,
                        label="Runs per configuration",
                        info=(
                            "Number of runs with different random seeds used "
                            "to evaluate each configuration."
                        ),
                        precision=0
                    )

                start_esn_tuning_button = gr.Button(
                    "Start Random Search",
                    variant="primary"
                )

                best_esn_config_output = gr.JSON(
                    label="Best ESN Configuration"
                )

                start_esn_tuning_button.click(
                    fn=utils.run_esn_tuning_gui,
                    inputs=[
                        input_series_state,
                        train_norm_state,
                        val_norm_state,
                        esn_n_trials,
                        esn_n_seeds
                    ],
                    outputs=best_esn_config_output
                )


# ============================================================
# LAUNCH APPLICATION
# ============================================================

demo.launch()