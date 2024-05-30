# Parameters description:

* **dataset_root**: path to tokenized (processed) MIDI files
* **max_length**: max length of tokens sequence that we will use to predict next token
* **additional_features_params**: dict of parameters for additional features
    * **global**: global features parameters
        * **genre**: bool whether to add genre feature
        * **author**: bool whether to add author feature
    * **local**: local features parameters
        * **timing**: bool whether to add timing feature (relative token position in composition timeline in the segment [0, 127])
        * **nnotes**: bool whether to add nnotes feature (numer of notes pressed currently)
