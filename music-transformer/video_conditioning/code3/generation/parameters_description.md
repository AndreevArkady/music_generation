# Parameters description:

* **force_cpu**: bool whether use cpu instead of cuda
* **generation_params**: parameters for generating song
    * **additional_features_params**: parameters used as additional features (must  have parameters, that were used in training):
        * **timing**: duration of generating song (in 1/100 of sec)
        * **nnotes**: just use true, if this parameter was used in training
        * **genre**: genre of generating composition (possible genres can be found in *utilities/constants.py*)
        * **author**: author's name. Must be one of the tokenizer vocabulary
    * **n_generate**: number of tokens to generate
    * **window_size**: size of sliding window of tokens that are used for generation
    * **primary_params**: parameters for loading primary sample
        * **primary_path**: path to the primary sample (*.midi or *.tsv)
        * **n_tokens**: number of tokens to take from primary
        * **save_primary**: bool whether to save primary file
    * **take_most_probable**: bool whether to take most probable next token or sample from distribution (better use sampling)
    * **generate_by_time**: bool whether to generate before reaching time limit
* **model_weights**: path to model weights
* **output_dir**: directory to save generated,
* **output_name**: name of folder with generated,
* **random_seed**: random seed,
* **save_params**: bool whether to save parameters, that were used for generation in json

