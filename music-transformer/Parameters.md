# Parameters description:

* **force_cpu**: Forces model to run on a cpu even when gpu is available
* **rpr**: Use a modified Transformer for Relative Position Representations
* **no_tensorboard**: Turns off tensorboard result reporting
* **continue_weights**: Model weights to continue training based on
* **continue_epoch**: Epoch the continue_weights model was at
* **lr**: Constant learn rate. Leave as None for a custom scheduler
* **ce_smoothing**: Smoothing parameter for smoothed cross entropy loss (defaults to no smoothing)
* **input_dir**: Folder of preprocessed and pickled midi files
* **output_dir**: Folder to save model weights. Saves one every epoch
* **sentiment_file**: File with sentiments per file
* **sentiment_per_token_file**: File with sentiments for each token
* **genres_file**: File with genres for each midi file,
* **weight_modulus**: How often to save epoch weights (ex: value of 10 means save every 10 epochs)
* **print_modulus**: How often to print train results for a batch (batch loss, learn rate, etc.)
* **n_workers**: Number of threads for the dataloader
* **batch_size**:  Batch size to use
* **epochs**: Number of epochs to use
* **max_sequence**: Maximum midi sequence to consider
* **n_layers**: Number of decoder layers to use
* **num_heads**: Number of heads to use for multi-head attention
* **d_model**:  Dimension of the model (output dim of embedding layers, etc.)
* **dim_feedforward**: Dimension of the feedforward layer
* **dropout**: Dropout rate
* **seed**: Seed for random
* **wandb_project_name**: Name for Wandb project