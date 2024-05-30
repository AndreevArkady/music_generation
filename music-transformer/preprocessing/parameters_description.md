# Parameters description:

* **input_dir**: directory with midi files
* **output_dir**: directory to place train, val, test directories with tokens sequence
* **ignore_warnings**: bool whether to ignore warnings
* **n_pools**: number of pool to use for multiprocessed preprocessing
* **random_state**: random_state parameter for train_test_split
* **processor_args**: dict of parameters for processor encoding
    * **drop_pauses**: bool whether to drop pauses
    * **append_silence**: seconds of silens to add
* **local_features_params**: local parameters
* **global_features_params**: global parameters
