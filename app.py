import gradio as gr
from gui import utils

# Reset config, input, output and weights directories
utils.reset_gui()

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
                **Input format:** Upload a CSV file containing the time series in a column named `value`. 
                Do not include a time column. Values must be equally spaced in time, for example one value 
                every second, minute, or hour. The x-axis of the plots represents these time units.
                            
                **Note:** The application has currently been tested only on the Mackey-Glass 
                chaotic time series with delay parameter τ = 17 (MG17). Model performance 
                is therefore not guaranteed for other time series. Validation and evaluation 
                on different time series are planned as future research work.
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

            gr.Markdown(
                """
                    **Suggested split for MG17:** 6000 samples for training (60%), 1000 for validation (10%), 
                    and 3000 for testing (30%). A larger test set allows for a more robust evaluation 
                    of model generalization on unseen data.
                """
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

            # -------------------------
            # Select LSTM mode
            # -------------------------

            lstm_mode = gr.Radio(
                choices=[
                    "Train New Model",
                    "Use Pretrained Model"
                ],
                value="Train New Model",
                label="LSTM Mode"
            )

            # ====================================================
            # TRAIN NEW MODEL
            # ====================================================

            with gr.Group(visible=True) as lstm_train_section:

                gr.Markdown("### Train New Model")

                gr.Markdown(
                    "Define the hyperparameters manually or optionally "
                    "load them from a JSON configuration file."
                )

                # Optional JSON used only to pre-fill the hyperparameter fields
                lstm_train_config_file = gr.File(
                    label="Optional Configuration (.json)",
                    file_types=[".json"],
                    type="filepath"
                )

                gr.Markdown("### Hyperparameters")

                gr.Markdown(
                    "Fixed training parameters: "
                    "`num_epochs = 100`, `patience = 5`"
                )

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
                    info="Suggested for MG17: 1e-4–1e-2"
                )

                # If a JSON is uploaded, use it to fill the fields
                lstm_train_config_file.change(
                    fn=utils.load_lstm_config,
                    inputs=lstm_train_config_file,
                    outputs=[
                        lstm_hidden_size,
                        lstm_sequence_length,
                        lstm_batch_size,
                        lstm_learning_rate
                    ]
                )

            # ====================================================
            # USE PRETRAINED MODEL
            # ====================================================

            with gr.Group(visible=False) as lstm_pretrained_section:

                gr.Markdown("### Use Pretrained Model")

                gr.Markdown(
                    "Load the configuration and weights of a previously trained LSTM."
                )

                lstm_pretrained_config_file = gr.File(
                    label="Configuration (.json)",
                    file_types=[".json"],
                    type="filepath"
                )

                lstm_pretrained_weights_file = gr.File(
                    label="Weights (.pt)",
                    file_types=[".pt"],
                    type="filepath"
                )

            # ====================================================
            # SWITCH BETWEEN MODES
            # ====================================================

            def update_lstm_mode(mode):
                train_mode = mode == "Train New Model"

                return (
                    gr.update(visible=train_mode),
                    gr.update(visible=not train_mode)
                )

            lstm_mode.change(
                fn=update_lstm_mode,
                inputs=lstm_mode,
                outputs=[
                    lstm_train_section,
                    lstm_pretrained_section
                ]
            )

            # ====================================================
            # RUN LSTM
            # ====================================================

            run_lstm_button = gr.Button(
                "Run LSTM",
                variant="primary"
            )

            gr.Markdown(
                """
                **Note:** LSTM training may take a few minutes when running on CPU.
                """
            )

            # ====================================================
            # RESULTS
            # ====================================================

            gr.Markdown("### Results")

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

            # ====================================================
            # CONNECT BUTTON TO BACKEND FUNCTION
            # ====================================================

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

                    lstm_mode,

                    lstm_hidden_size,
                    lstm_sequence_length,
                    lstm_batch_size,
                    lstm_learning_rate,

                    lstm_pretrained_config_file,
                    lstm_pretrained_weights_file
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
            # DOWNLOAD LSTM FILES
            # ====================================================

            gr.Markdown("### Download Model Files")

            with gr.Row():

                download_lstm_config_button = gr.Button(
                    "Download Configuration",
                    variant="primary"
                )

                download_lstm_weights_button = gr.Button(
                    "Download Weights",
                    variant="primary"
                )

            # Files returned to the user
            lstm_config_download = gr.File(
                label="LSTM Configuration"
            )

            lstm_weights_download = gr.File(
                label="LSTM Weights"
            )

            # Return the current LSTM configuration file
            download_lstm_config_button.click(
                fn=utils.get_lstm_config_file,
                inputs=[],
                outputs=lstm_config_download
            )

            # Return the current LSTM weights file
            download_lstm_weights_button.click(
                fn=utils.get_lstm_weights_file,
                inputs=[],
                outputs=lstm_weights_download
            )

            # ====================================================
            # Hyperparameter Tuning
            # ====================================================
            with gr.Accordion(
                "Automatic Hyperparameter Tuning",
                open=False
            ):

                gr.Markdown(
                    """
                    Random search evaluates multiple LSTM hyperparameter configurations
                    and returns the best configuration found.

                    **Note:** Execution time depends on the number of trials and runs per configuration
                    and may take several hours when running on CPU. The default settings provide a
                    faster search and typically take around 5 minutes on CPU.

                    **Best configuration found:** *TO DO: insert best configuration.*
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
                        label="Runs per configuration",
                        precision=0
                    )

                start_lstm_tuning_button = gr.Button(
                    "Start Random Search",
                    variant="primary"
                )

                best_lstm_config_output = gr.JSON(
                    label="Best LSTM Configuration"
                )

                download_best_lstm_config_button = gr.Button(
                    "Download Best Configuration",
                    variant="primary"
                )

                best_lstm_config_download = gr.File(
                    label="Best LSTM Configuration File"
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
                    outputs=best_lstm_config_output
                )

                download_best_lstm_config_button.click(
                    fn=utils.get_best_lstm_config_file,
                    inputs=[],
                    outputs=best_lstm_config_download
                )
                        

        # ====================================================
        # ESN
        # ====================================================

        with gr.Tab("ESN"):

            gr.Markdown("## ESN")

            # -------------------------
            # Select ESN mode
            # -------------------------

            esn_mode = gr.Radio(
                choices=[
                    "Train New Model",
                    "Use Pretrained Model"
                ],
                value="Train New Model",
                label="ESN Mode"
            )

            # ====================================================
            # TRAIN NEW MODEL
            # ====================================================

            with gr.Group(visible=True) as esn_train_section:

                gr.Markdown("### Train New Model")

                gr.Markdown(
                    "Define the hyperparameters manually or optionally "
                    "load them from a JSON configuration file."
                )

                # Optional JSON used only to pre-fill the hyperparameter fields
                esn_train_config_file = gr.File(
                    label="Optional Configuration (.json)",
                    file_types=[".json"],
                    type="filepath"
                )

                gr.Markdown("### Hyperparameters")

                esn_reservoir_size = gr.Number(
                    value=100,
                    label="Reservoir size",
                    info="Suggested for MG17: 50–200",
                    precision=0
                )

                esn_spectral_radius = gr.Number(
                    value=0.9,
                    label="Spectral radius",
                    info="Suggested for MG17: 0.70–0.99"
                )

                esn_reservoir_connectivity = gr.Number(
                    value=0.05,
                    label="Reservoir connectivity",
                    info="Suggested for MG17: 0.01–0.10"
                )

                esn_input_scaling = gr.Number(
                    value=0.5,
                    label="Input scaling",
                    info="Suggested for MG17: 0.10–1.00"
                )

                esn_washout = gr.Number(
                    value=100,
                    label="Washout",
                    info="Suggested for MG17: 50–500",
                    precision=0
                )

                esn_alpha = gr.Number(
                    value=0.1,
                    label="Alpha",
                    info="Suggested for MG17: 0.05–1.00"
                )

                esn_ridge = gr.Number(
                    value=1e-6,
                    label="Ridge",
                    info="Suggested for MG17: 1e-8–1e-4"
                )

                # If a JSON is uploaded, use it to fill the fields
                esn_train_config_file.change(
                    fn=utils.load_esn_config,
                    inputs=esn_train_config_file,
                    outputs=[
                        esn_reservoir_size,
                        esn_spectral_radius,
                        esn_reservoir_connectivity,
                        esn_input_scaling,
                        esn_washout,
                        esn_alpha,
                        esn_ridge
                    ]
                )

            # ====================================================
            # USE PRETRAINED MODEL
            # ====================================================

            with gr.Group(visible=False) as esn_pretrained_section:

                gr.Markdown("### Use Pretrained Model")

                gr.Markdown(
                    "Load the configuration and weights of a previously trained ESN."
                )

                esn_pretrained_config_file = gr.File(
                    label="Configuration (.json)",
                    file_types=[".json"],
                    type="filepath"
                )

                esn_pretrained_weights_file = gr.File(
                    label="Weights (.pt)",
                    file_types=[".pt"],
                    type="filepath"
                )

            # ====================================================
            # SWITCH BETWEEN MODES
            # ====================================================

            def update_esn_mode(mode):
                train_mode = mode == "Train New Model"

                return (
                    gr.update(visible=train_mode),
                    gr.update(visible=not train_mode)
                )

            esn_mode.change(
                fn=update_esn_mode,
                inputs=esn_mode,
                outputs=[
                    esn_train_section,
                    esn_pretrained_section
                ]
            )

            # ====================================================
            # RUN ESN
            # ====================================================

            run_esn_button = gr.Button(
                "Run ESN",
                variant="primary"
            )

            # ====================================================
            # RESULTS
            # ====================================================

            gr.Markdown("### Results")

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

            # ====================================================
            # CONNECT BUTTON TO BACKEND FUNCTION
            # ====================================================

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

                    esn_mode,

                    esn_reservoir_size,
                    esn_spectral_radius,
                    esn_reservoir_connectivity,
                    esn_input_scaling,
                    esn_washout,
                    esn_alpha,
                    esn_ridge,

                    esn_pretrained_config_file,
                    esn_pretrained_weights_file
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

            # ====================================================
            # DOWNLOAD ESN FILES
            # ====================================================

            gr.Markdown("### Download Model Files")

            with gr.Row():

                download_esn_config_button = gr.Button(
                    "Download Configuration",
                    variant="primary"
                )

                download_esn_weights_button = gr.Button(
                    "Download Weights",
                    variant="primary"
                )

            # Files returned to the user
            esn_config_download = gr.File(
                label="ESN Configuration"
            )

            esn_weights_download = gr.File(
                label="ESN Weights"
            )

            # Return the current ESN configuration file
            download_esn_config_button.click(
                fn=utils.get_esn_config_file,
                inputs=[],
                outputs=esn_config_download
            )

            # Return the current ESN weights file
            download_esn_weights_button.click(
                fn=utils.get_esn_weights_file,
                inputs=[],
                outputs=esn_weights_download
            )
            
            # ====================================================
            # Hyperparameter Tuning
            # ====================================================

            with gr.Accordion(
                "Automatic Hyperparameter Tuning",
                open=False
            ):

                gr.Markdown(
                    """
                    Random search evaluates multiple ESN hyperparameter configurations
                    and returns the best configuration found.

                    **Note:** ESN tuning is generally faster than LSTM tuning, but execution
                    time still depends on the number of trials and runs per configuration.

                    Suggested settings:
                    - `trials = 150`
                    - `runs per configuration = 10`

                    **Best configuration found:** *TO DO: insert best configuration.*
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

                download_best_esn_config_button = gr.Button(
                    "Download Best Configuration",
                    variant="primary"
                )

                best_esn_config_download = gr.File(
                    label="Best ESN Configuration File"
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

                download_best_esn_config_button.click(
                    fn=utils.get_best_esn_config_file,
                    inputs=[],
                    outputs=best_esn_config_download
                )


# ============================================================
# LAUNCH APPLICATION
# ============================================================

demo.launch()