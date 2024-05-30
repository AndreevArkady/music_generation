import json
import os
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset import augmentations, features_makers
from utilities.constants import ADDITIONAL_FEATURES_TYPE, GENRES_LIST, MAX_TOKENS_SEQUENCE_LENGTH, TOKEN_END, TOKEN_PAD, TORCH_FLOAT, TORCH_LABEL_TYPE  # noqa: E501
from utilities.device import cpu_device  # noqa: I005

# default start position for extracting input sequence (for model)
# from sample tokens sequence
SEQUENCE_START = 0
N_TIMING_SEGMENTS = 128


def filter_by_length(args):
    """Check if number of tokens in file is less than max_length.

    Due to using this function with imap from Pool
    it is necessary to pass arguments as one object

    Args:
        args: pair of file_path and max_length

    Returns:
        file_path if there is less tokens than max_length
        None elsewise
    """
    file_path, max_length = args
    composition_features = pd.read_csv(file_path, sep='\t')
    length = composition_features.shape[0]
    if length <= max_length:
        return file_path
    return None


class GenerationAdditionalFeatures(object):  # noqa: WPS214
    """Class for additional features at generation stage."""
    def __init__(self, length, parameters, primary):
        """Initialize GenerationAdditionalFeatures.

        Args:
            length: length of features Tensor
                should be equal to the final generated sequence
            parameters: parameters for additional features for generation
            primary: Tensor of primary tokens sequence that will be evaluated
        """
        self.generation_features_funcs = {
            'genre': self.add_genre,
            'author': self.add_author,
            'timing': self.add_timing,
            'nnotes': self.add_nnotes,
        }
        self.tokenizers = {}
        import os
        print('------', os.path.abspath('.'))
        with open('dataset/additional_features_state.json') as fp:
            state = json.load(fp)
            self.local_features_columns_count = state['local_features_columns_count']
            self.global_features_columns_count = state['global_features_columns_count']
            self.local_features_indices = state['local_features_indices']
            self.global_features_indices = state['global_features_indices']
            self.init_tokenizers(state['tokenizers'])

        self.local_features = torch.full(  # noqa: WPS317
            (length, self.local_features_columns_count),
            TOKEN_PAD,
            dtype=ADDITIONAL_FEATURES_TYPE,
            device=cpu_device()
        )
        self.global_features = torch.zeros(  # noqa: WPS317
            self.global_features_columns_count,
            dtype=ADDITIONAL_FEATURES_TYPE,
            device=cpu_device()
        )
        self.length = length
        self.parameters = parameters
        self.updatable_features = []
        self.init_updatable_features(parameters)
        self.primary = primary.detach().clone()
        self.init_tensor(parameters, primary)

    def init_updatable_features(self, parameters):
        """Init list of updatable features.

        Args:
            parameters: parameters for additional features for generation
        """
        order = ['timing', 'nnotes', 'token_type', 'note_type', 'notes_duration']
        # updates should be strictly in this order

        for feature in order:
            if feature in parameters and (
                feature in self.local_features_indices or feature in self.global_features_indices
            ):
                self.updatable_features.append(feature)

    def init_tokenizers(self, tokenizers_paths):
        """Initialize tokenizers.

        Args:
            tokenizers_paths: dict of tokenizers path
        """
        for tokenizer_name, tokenizer_path in tokenizers_paths.items():
            tokenizer = Tokenizer.from_file(tokenizer_path)
            self.tokenizers[tokenizer_name] = tokenizer

    def add_genre(self, genre, *args):
        """Add Genre parameter for generation.

        Args:
            genre: genre value
            args: unneccesary args

        Raises:
            RuntimeError: if Genre was not used on training
            ValueError: trying to initialize with unknown value
        """
        if 'genre' not in self.global_features_indices:
            raise RuntimeError(
                "Feature 'genre' was not used in AdditionalFeatures initialization."  # noqa: E501
            )
        if genre not in GENRES_LIST:
            raise ValueError(f'Unknown genre: {genre}')
        begin_column, end_column = self.global_features_indices['genre']  # noqa: WPS204
        tokenizer = self.tokenizers['genre']
        encoding = tokenizer.encode(genre)
        self.global_features[begin_column:end_column] = encoding.ids[0]  # noqa: WPS362

    def add_author(self, author, *args):
        """Add Author parameter for generation.

        Args:
            author: author value
            args: unneccesary args

        Raises:
            RuntimeError: if Author was mot used on training
        """
        if 'author' not in self.global_features_indices:
            raise RuntimeError(
                "Feature 'author' was not used in AdditionalFeatures initialization."  # noqa: E501
            )
        begin_column, end_column = self.global_features_indices['author']  # noqa: WPS204
        tokenizer = self.tokenizers['author']
        encoding = tokenizer.encode(author)
        self.global_features[begin_column:end_column] = encoding.ids[0]  # noqa: WPS362

    def add_timing(self, total_time, primary):
        """Add Timing feature into additional features tensor for generation.

        Args:
            total_time: duration of generating sequence
            primary: Tensor of primary tokens sequence that will be evaluated

        Raises:
            ValueError: if total_time in not int
        """
        if not isinstance(total_time, int):
            raise ValueError(
                f'timing parameter in not bool, its type: {type(total_time)}'
            )
        begin_column, end_column = self.local_features_indices['timing']
        length = primary.shape[0]
        timings, _ = features_makers.timing_feature(primary)
        self.local_features[
            :length,
            begin_column:end_column
        ] = timings.unsqueeze(dim=-1) * N_TIMING_SEGMENTS // (total_time + 1)

    def add_nnotes(self, parameters, primary):
        """Add nnotes feature for generation.

        Add number of notes pressed at each moment.
        Updates nnotes according to the specified preset.

        Args:
            parameters: 'enable': bool, 'preset': dict of boarders and values
            primary: Tensor of primary tokens sequence that will be evaluated

        Raises:
            ValueError: if parameter in not bool
        """
        if not isinstance(parameters, dict):
            raise ValueError(
                f'nnotes parameter in not dict, its type: {type(parameters)}'
            )

        if not parameters['enable']:
            return

        begin_column, end_column = self.local_features_indices['nnotes']
        length = primary.shape[0]
        nnotes = features_makers.nnotes_feature(primary).unsqueeze(dim=-1)
        self.local_features[:length, begin_column:end_column] = nnotes

        if not parameters['preset'] or parameters['preset'] is None:
            return

        timing_begin, timing_end = self.local_features_indices['timing']
        nnotes_begin, nnotes_end = self.local_features_indices['nnotes']
        timing = self.local_features[:, timing_begin:timing_end]
        nnotes = self.local_features[:, nnotes_begin:nnotes_end]
        for segment_info in parameters['preset']:
            borders, value = segment_info['borders'], segment_info['value']
            begin, end = borders
            nnotes[(timing >= begin) & (timing < end)] = value  # noqa: WPS465

    def init_tensor(self, parameters, primary):
        """Build Additional Features Tensor that will be used for generation.

        Args:
            parameters: parameters for additional features for generation
            primary: Tensor of primary tokens sequence that will be evaluated

        Raises:
            ValueError: if unknown feature name found
        """
        for feature, feature_parameters in parameters.items():
            if feature not in self.generation_features_funcs:
                raise ValueError(f'Unknown feature: {feature}')
            if feature not in self.local_features_indices and feature not in self.global_features_indices:
                continue
            self.generation_features_funcs[feature](
                feature_parameters,
                primary
            )

    def update_additional_features(self, tokens):
        """Build Additional Features Tensor that will be used for generation.

        Args:
            tokens: currently generated tokens
        """
        for feature in self.updatable_features:
            self.generation_features_funcs[feature](
                self.parameters[feature],
                tokens
            )

    def process_new_token(self, new_token):
        """Updates features tensor for O(1).

        Updates features tensor in according to new token

        Args:
            new_token: new token value
        """
        ...  # noqa: WPS428


class AdditionalFeatures(object):  # noqa: WPS214
    """Class for additional features for tokens.

    Attributes:
        files_paths: list of tokens files paths
        features_params: dict of parameter for features:
            <feature_name, parameters>
        current_data: DataFrame of added element
        current_elem_index: index of currently adding element
            (element := sequence of tokens)
        current_elem_length: length of currently adding element
        local_features_columns_count: number of columns of local features
        global_features_columns_count: number of columns of global features
        features_count: number of features that were  parsed from
            parameters and will be added
        local_features_indices: begin and end column indices of local feature:
            <feature_name: (begin, end)>
        global_features_indices: begin and end column indices of global feature:
            <feature_name: (begin, end)>
        global_features: dict of global features params
        tokenizers: tokenizers dict for global featores
        vocab_sizes: vocabulary sizes for tokenizers
        features_processors: dict of methods for processing every
            feature: <feature_name, methods>
        local_features_tensor: tensor for local features
        global_features_tensor: tensor for global features
    """

    def __init__(
        self,
        files_paths=None,
        additional_features_params=None,
        global_features=None
    ):
        """Initialize AdditionalFeatures.

        Args:
            files_paths: list of tokens files paths
            additional_features_params: dict of additional features
                parameters
            global_features: global features  dict of files in dataset
        """
        self.files_paths = files_paths
        self.features_params = {}
        self.current_data = None
        self.current_elem_index = 0
        self.current_elem_length = 0
        self.local_features_columns_count = 0
        self.global_features_columns_count = 0
        self.local_features_indices = {}
        self.global_features_indices = {}
        self.global_features = global_features if global_features else {}
        self.tokenizers = {}

        self.vocab_sizes = {
            'timing': 128,
            'nnotes': 11,
        }

        self.features_processors = {
            'genre': self.add_genre,
            'author': self.add_author,
            'timing': self.add_timing,
            'nnotes': self.add_nnotes,
        }

        self.init_params(additional_features_params)

        self.local_features_tensor = torch.zeros(  # noqa: WPS317
            (
                MAX_TOKENS_SEQUENCE_LENGTH,
                self.local_features_columns_count,
                len(files_paths)
            ),
            dtype=ADDITIONAL_FEATURES_TYPE,
            device=cpu_device()
        )
        for feature, indices in self.local_features_indices.items():
            begin, end = indices
            pad_token = self.vocab_sizes[feature] - 1
            self.local_features_tensor[:, begin:end, :] = pad_token

        self.global_features_tensor = torch.zeros(  # noqa: WPS317
            (
                self.global_features_columns_count,
                len(files_paths)
            ),
            dtype=ADDITIONAL_FEATURES_TYPE,
            device=cpu_device()
        )

    def init_params(self, additional_features_params):
        """Initialize parameters for additional features.

        Args:
            additional_features_params: parameters for additional features

        Raises:
            ValueError: passed unknown parameter
        """
        params_parsers = {
            'global': self.parse_global_params,
            'local': self.parse_local_params,
        }

        if additional_features_params is not None:
            for feature, params in additional_features_params.items():
                if feature not in params_parsers:
                    raise ValueError(f'Unknown parameter: {feature}')
                params_parsers[feature](params)

        tokenizers_paths = self.init_tokenizers(
            additional_features_params['global']
        )

        dataset_type = os.path.basename(os.path.dirname(self.files_paths[0]))
        if dataset_type == 'train':
            state = {
                'additional_features_params': additional_features_params,
                'local_features_columns_count': self.local_features_columns_count,
                'global_features_columns_count': self.global_features_columns_count,
                'local_features_indices': self.local_features_indices,
                'global_features_indices': self.global_features_indices,
                'tokenizers': tokenizers_paths,
                'vocab_sizes': self.vocab_sizes
            }

            with open('dataset/additional_features_state.json', 'w') as fp:
                json.dump(state, fp, indent=4)

    def init_tokenizers(self, global_features):
        """Initialize tokeniezers and return their paths.

        Args:
            global_features: dict of global features

        Returns:
            tokenizers paths
        """
        path_prefix = os.path.dirname(os.path.dirname(self.files_paths[0]))
        tokenizers_paths = {}
        for feature, use_feature in global_features.items():
            if use_feature:
                tokenizer_path = os.path.join(path_prefix, feature + '-tokenizer.json')  # noqa: WPS336
                tokenizer = Tokenizer.from_file(tokenizer_path)
                self.tokenizers[feature] = tokenizer
                tokenizers_paths[feature] = tokenizer_path

        return tokenizers_paths

    def parse_global_params(self, params):  # noqa: WPS231
        """Parse global additional features.

        Args:
            params: parameters for Genre features category

        Raises:
            ValueError: if wrong type of parameter argument
        """
        if 'genre' in params:
            param = params['genre']
            if not isinstance(param, bool):
                raise ValueError(
                    f'Param genre is not a bool, its type: {type(param)}'  # noqa: E501
                )
            if param:
                self.fit_global_feature_params(
                    'genre',
                    parameters=True,
                    columns_count=1
                )
        if 'author' in params:
            param = params['author']
            if not isinstance(param, bool):
                raise ValueError(
                    f'Param author is not a bool, its type: {type(param)}'  # noqa: E501
                )
            if param:
                self.fit_global_feature_params(
                    'author',
                    parameters=True,
                    columns_count=1
                )

    def add_genre(self):
        """Adds tokenized Genre feature."""
        file_path = self.files_paths[self.current_elem_index]
        genre = self.global_features[file_path]['genre']
        tokenizer = self.tokenizers['genre']
        encoding = tokenizer.encode(genre)
        self.fill_global_feature('genre', encoding.ids[0])

    def add_author(self):
        """Adds tokenized Author feature."""
        file_path = self.files_paths[self.current_elem_index]
        author = self.global_features[file_path]['author']
        tokenizer = self.tokenizers['author']
        encoding = tokenizer.encode(author)
        self.fill_global_feature('author', encoding.ids[0])

    def parse_local_params(self, params):  # noqa: WPS231, WPS238
        """Parse local additional features.

        Args:
            params: parameters for Structure features category

        Raises:
            ValueError: if wrong type of parameter argument
        """
        if 'duration' in params:
            param = params['duration']
            if not isinstance(param, bool):
                raise ValueError(
                    f"Param 'duration' is not a bool, its type: {type(param)}"
                )
            if param:
                self.fit_local_feature_params(
                    'duration',
                    parameters=True,
                    columns_count=1
                )
        if 'nnotes' in params:
            param = params['nnotes']
            if not isinstance(param, bool):
                raise ValueError(
                    f"Param 'nnotes' is not a bool, its type: {type(param)}"
                )
            if param:
                self.fit_local_feature_params(
                    'nnotes',
                    parameters=True,
                    columns_count=1
                )
        if 'timing' in params:
            param = params['timing']
            if not isinstance(param, bool):
                raise ValueError(
                    f"Param 'timing' is not a bool, its type: {type(param)}"
                )
            if param:
                self.fit_local_feature_params(
                    'timing',
                    parameters=True,
                    columns_count=1
                )

    def add_timing(self):
        """Adds timing feature."""
        timings = self.current_data['timings'].to_numpy()
        timings = torch.tensor(timings, dtype=ADDITIONAL_FEATURES_TYPE)
        self.fill_local_feature(
            'timing',
            timings.unsqueeze(dim=-1)
        )

    def add_nnotes(self):
        """Adds nnotes feature."""
        nnotes = self.current_data['nnotes'].to_numpy()
        nnotes = torch.tensor(nnotes, dtype=ADDITIONAL_FEATURES_TYPE)
        self.fill_local_feature(
            'nnotes',
            nnotes.unsqueeze(dim=-1)
        )

    def fill_local_feature(self, feature_name, data):
        """Write local feature's data into correct columns.

        Args:
            feature_name: feature name
            data: tensor of feature data
        """
        begin_column, end_column = self.local_features_indices[feature_name]
        self.local_features_tensor[
            :self.current_elem_length,
            begin_column:end_column,
            self.current_elem_index
        ] = data

    def fill_global_feature(self, feature_name, data):
        """Write global feature's data into correct columns.

        Args:
            feature_name: feature name
            data: value of feature data
        """
        begin_column, end_column = self.global_features_indices[feature_name]
        self.global_features_tensor[
            begin_column:end_column,
            self.current_elem_index
        ] = data

    def fit_local_feature_params(self, feature_name, parameters, columns_count):
        """Write feature's parameters and updates other counters.

        Args:
            feature_name: feature name
            parameters: object that will be feature's parameters
            columns_count: number of columns that feature require
        """
        self.features_params[feature_name] = parameters

        begin = self.local_features_columns_count
        end = self.local_features_columns_count + columns_count
        self.local_features_indices[feature_name] = (begin, end)

        self.local_features_columns_count += columns_count

    def fit_global_feature_params(self, feature_name, parameters, columns_count):
        """Write feature's parameters and updates other counters.

        Args:
            feature_name: feature name
            parameters: object that will be feature's parameters
            columns_count: number of columns that feature require
        """
        self.features_params[feature_name] = parameters
        self.global_features_indices[feature_name] = (
            self.global_features_columns_count,
            self.global_features_columns_count + columns_count
        )

        self.global_features_columns_count += columns_count

    def append(self, composition_data):
        """Add additional features for tokens sequence x.

        Sequences should be appended in the same order as filenames in
        files_paths.

        Args:
            composition_data: pandas DataFrame of composition data

        Raises:
            RuntimeError: if trying to add more elements than provided paths
        """
        if self.current_elem_index >= len(self.files_paths):
            raise RuntimeError(
                'Trying to add more elements than provided paths.'
            )
        self.current_data = composition_data
        self.current_elem_length = composition_data.shape[0]
        for feature in self.features_params.keys():
            self.features_processors[feature]()

        self.current_elem_index += 1

    def get_additional_features_tensor(self, idx):
        """Get Tensor of additional features by idx.

        Args:
            idx: idx of element whose feature will be taken

        Returns:
            Tensor of local features, Tensor of global features
        """
        return self.local_features_tensor[:, :, idx], self.global_features_tensor[:, idx]


class EPianoDataset(Dataset):
    """Dataset for tokens sequences and additional features of MIDI files.

    Attributes:
        root: directory with MIDI files
        max_seq: max_seq param
        augment: whether to augment sequences
        random_seq: random_seq param
        data_files_paths: list of tokens files paths
        file_names_indices: dict of file names and indices
            of corresponding files in other lists (like data_files_paths or
            sequences_actual_lengths)
        additional_features_processor: AdditionalFeatures object for
            managing additional features
        additional_features_columns_count: number of columns in each
            additional features Tensor
        sequences_actual_lengths: array of actual tokens sequences
            lengths
        tokens_sequences: Tensor with columns of original
            tokens sequences
    """

    def __init__(  # noqa: WPS211
        self,
        root,
        max_length=None,
        max_seq=2048,
        random_seq=True,
        random_pos=True,
        augment=False,
        additional_features_params=None
    ):
        """Initialize EPianoDataset.

        Args:
            root: directory with MIDI files
            max_length: max number of tokens in compositions to use
            max_seq: length of max token sequence to extract
            augment: whether to augment sequences
            random_seq: whether process_midi use random start point for
                sequence if length of tokens sequence is greater than max_seq
            random_pos: take max_length tokens from random position if
                total length is greater than max_length
            additional_features_params: parameters for AdditionalFeatures
        """
        self.root = root
        self.max_length = max_length
        self.max_seq = max_seq
        self.random_seq = random_seq
        self.random_pos = random_pos
        self.augment = augment

        root_contents = [
            os.path.join(root, content_name)
            for content_name in os.listdir(self.root)
        ]

        self.data_files_paths = [
            file_name
            for file_name in root_contents
            if os.path.isfile(file_name) and file_name.endswith('.tsv')
        ]

        if self.max_length is not None:
            func_args = list(zip(
                self.data_files_paths,
                [max_length] * len(self.data_files_paths),  # noqa: WPS435
            ))

            total = len(self.data_files_paths)
            progress_iters = 20
            max_iter_interval = 400
            n_processes = 20

            with Pool(n_processes) as p:
                filtered_paths = list(tqdm(
                    p.imap(filter_by_length, func_args),
                    miniters=total // progress_iters,
                    maxinterval=max_iter_interval,
                    total=total
                ))

            self.data_files_paths = [
                file_path
                for file_path in filtered_paths
                if file_path is not None
            ]

        self.file_names_indices = {
            os.path.basename(file_path): idx
            for idx, file_path in enumerate(self.data_files_paths)
        }

        if additional_features_params is None:
            additional_features_params = {}

        json_path = os.path.join(root, 'global_features.json')
        with open(json_path, 'r') as global_features_params_file:
            global_features = json.load(global_features_params_file)

        self.additional_features = AdditionalFeatures(
            self.data_files_paths,
            additional_features_params,
            global_features
        )

        self.sequences_actual_lengths = np.zeros(
            len(self.data_files_paths),
            dtype='int32'
        )
        self.tokens_sequences = torch.full(  # noqa: WPS317
            (MAX_TOKENS_SEQUENCE_LENGTH + 1, len(self.data_files_paths)),
            TOKEN_PAD,
            dtype=TORCH_LABEL_TYPE,
            device=cpu_device()
        )

        self.initialize_objects()

    def initialize_objects(self):
        """Initialize class objects.

        Initialize input sequences, additional features and target sequences.
        """
        total = len(self.data_files_paths)
        progress_iters = 20
        max_iter_interval = 400
        progress = tqdm(
            enumerate(self.data_files_paths),
            miniters=total // progress_iters,
            maxinterval=max_iter_interval,
            total=total
        )
        for i, file_path in progress:
            composition_features = pd.read_csv(file_path, sep='\t')
            tokens = composition_features['tokens'].to_numpy(dtype=np.int32)
            start = 0
            if self.random_pos and tokens.shape[0] > MAX_TOKENS_SEQUENCE_LENGTH:
                start = np.random.randint(tokens.shape[0] - MAX_TOKENS_SEQUENCE_LENGTH)
            tokens = tokens[start:start + MAX_TOKENS_SEQUENCE_LENGTH]

            sequence_length = tokens.shape[0]
            self.tokens_sequences[:sequence_length, i] = torch.tensor(
                tokens,
                dtype=TORCH_LABEL_TYPE,
                device=cpu_device()
            )
            self.tokens_sequences[sequence_length, i] = TOKEN_END
            self.sequences_actual_lengths[i] = sequence_length

            self.additional_features.append(
                composition_features.iloc[start:start + sequence_length, :]
            )

    def __len__(self):
        """How many data files exist in the given dataset.

        Returns:
            length of Dataset
        """
        return len(self.data_files_paths)

    def __getitem__(self, idx):
        """Get (x, additional features, y) by index.

        Create random sequence or from start depending on random_seq.

        Args:
            idx: index of the object in dataset

        Returns:
            input, additional features and the target Tensors
        """
        sequence, begin = self.get_item(idx)
        if self.augment:
            sequence = augmentations.apply_augmentations(sequence)

        x = sequence[:self.max_seq]
        y = sequence[1:self.max_seq + 1]

        local_features, global_features = self.additional_features.get_additional_features_tensor(idx)  # noqa: E501
        return (
            x,
            local_features[
                begin:begin + self.max_seq, :
            ],
            global_features,
            y
        )

    def get_by_file_name(self, file_name):
        """Get (x, additional features, y) by file name (with extension).

        Create random sequence or from start depending on random_seq.

        Args:
            file_name: file name (with extension)

        Returns:
            input, additional features and the target Tensors

        Raises:
            KeyError: if no such file in dataset
        """
        if file_name not in self.file_names_indices:
            raise KeyError(f'No such file in dataset: {file_name}')
        return self.__getitem__(self.file_names_indices[file_name])

    def get_item(self, idx):
        """Take in pre-processed raw midi and returns the input and target.

        Can use a random sequence or go from the start based on random_seq.

        Args:
            idx: index of sequence in sequences Tensor to use

        Returns:
            sequence of tokens, actual length of sequences without padding
        """
        tokens_tensor = self.tokens_sequences[:, idx]

        sample_size = self.sequences_actual_lengths[idx]
        full_sequence_size = self.max_seq + 1

        if sample_size < full_sequence_size:
            # if the input sample size is less than size of the sequences we
            # just use the tokens that we have and pair them with the next ones
            tokens_sequence = tokens_tensor[:self.max_seq + 1]
            return tokens_sequence, 0

        # else selecting begin position as SEQUENCE_START or randomly
        if self.random_seq:
            end_range = sample_size - full_sequence_size
            begin = random.randint(SEQUENCE_START, end_range)
        else:
            begin = SEQUENCE_START
        end = begin + full_sequence_size
        tokens_sequence = tokens_tensor[begin:end]
        return tokens_sequence, begin


def create_epiano_datasets(
    dataset_root,
    max_length=None,
    max_seq=2048,
    additional_features_params=None
):
    """Create train, evaluation, and test EPianoDataset objects.

    Uses pre-processed (preprocess_midi.py) MIDI files.
    root contains train, val, and test folders.

    Args:
        dataset_root: path to tokenized (processed) MIDI files
        max_length: max number of tokens in compositions to use
        max_seq: max length of tokens sequence that we will use to
            predict next token
        additional_features_params: dict of parameters for additional
            features

    Returns:
        train, val, test EPianoDataset objects
    """
    train_root = os.path.join(dataset_root, 'train')
    val_root = os.path.join(dataset_root, 'val')
    test_root = os.path.join(dataset_root, 'test')

    train_dataset = EPianoDataset(
        train_root,
        max_length,
        max_seq,
        random_seq=True,
        random_pos=True,
        augment=False,
        additional_features_params=additional_features_params
    )

    val_dataset = EPianoDataset(
        val_root,
        max_length,
        max_seq,
        random_seq=False,
        random_pos=False,
        augment=False,
        additional_features_params=additional_features_params
    )

    test_dataset = EPianoDataset(
        test_root,
        max_length,
        max_seq,
        random_seq=False,
        random_pos=False,
        augment=False,
        additional_features_params=additional_features_params
    )

    return train_dataset, val_dataset, test_dataset


def compute_epiano_accuracy(preds, target):
    """Compute the average accuracy for the given input and output batches.

    Args:
        preds: result of model prediction. Tensor of shape
            (batch_size, sequence_size, preds_count)
        target: true values of the target. Tensor of shape
            (batch_size, sequence_size)

    Returns:
        accuracy
    """
    preds = torch.argmax(preds, dim=-1)

    preds = preds.flatten()
    target = target.flatten()

    mask = (target != TOKEN_PAD)

    preds = preds[mask]
    target = target[mask]

    if target.nelement() == 0:
        return 1.0

    num_right = (preds == target)
    num_right = torch.sum(num_right).type(TORCH_FLOAT)

    return num_right / len(target)
