import json
import os
import shutil
import warnings
from multiprocessing import Pool

import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch import tensor
from tqdm import tqdm

from dataset import features_makers
from dataset.e_piano import N_TIMING_SEGMENTS
from preprocessing.processor import encode_midi
from tension_calculation import key_seq_per_file

def prep_midi(  # noqa: WPS210, WPS211
    input_dir,
    output_dir,
    ignore_warnings,
    n_pools,
    random_state,
    processor_args,
    local_features_params,
    global_features_params
):  # noqa: WPS210
    """Pre-processes midi files.

    Pre-processes midi files in directory and its subdirectories.
    Splits them into train, eval, test directories.

    Args:
        input_dir: directory with midi files.
        output_dir: directory to place train test and val datasets
        n_pools: number of Pools to use for multiprocessing
        random_state: random_state parameter for train_test_split
        processor_args: dict of parameters for processor encoding
        local_features_params: parameter for local features
        global_features_params: parameters for global features
    """
    if ignore_warnings:
        warnings.filterwarnings('ignore')
    create_dirs(output_dir)

    supported_formats = {'.midi', '.mid'}
    file_paths = []

    for root, dirs, files in os.walk(input_dir, topdown=False, followlinks=True):  # noqa: B007
        for file_name in files:
            name, ext = os.path.splitext(file_name)
            if ext.lower() in supported_formats:
                file_paths.append(os.path.join(root, file_name))

    train, test = train_test_split(
        file_paths,
        test_size=0.1,
        random_state=random_state
    )
    train, val = train_test_split(  # noqa: WPS110
        train,
        test_size=0.1,
        random_state=random_state
    )

    files_final_names = {}
    for train_file_path in train:
        files_final_names[train_file_path] = os.path.join(
            output_dir,
            'train',
            tsv_basename(train_file_path, standardize=True)
        )
    for val_file_path in val:
        files_final_names[val_file_path] = os.path.join(
            output_dir,
            'val',
            tsv_basename(val_file_path, standardize=True)
        )
    for test_file_path in test:
        files_final_names[test_file_path] = os.path.join(
            output_dir,
            'test',
            tsv_basename(test_file_path, standardize=True)
        )

    local_features_args = [
        [
            file_name,
            file_type,
            processor_args,
            local_features_params,
        ]
        for file_name, file_type in files_final_names.items()
    ]

    with Pool(n_pools) as p:
        list(tqdm(
            p.imap(save_local_features, local_features_args),
            total=len(files_final_names)
        ))
    # without list wrapping interpreter skips pool execution

    global_features_args = [
        [
            file_name,
            file_type,
            global_features_params,
        ]
        for file_name, file_type in files_final_names.items()
    ]

    with Pool(n_pools) as p:  # noqa: WPS440
        global_features_pairs = list(tqdm(
            p.imap(get_global_features, global_features_args),
            total=len(files_final_names)
        ))

    save_global_features(
        global_features_pairs,
        files_final_names,
        train,
        val,
        test,
        output_dir,
        global_features_params
    )
    
    count_files = lambda x: len([
            name 
            for name in os.listdir(os.path.join(output_dir, x)) 
            if name.endswith('.tsv')
        ])

    print(
        'Train size:',
        count_files('train')
    )
    print(
        'Val size:',
        count_files('val')
    )
    print(
        'Test size:',
        count_files('test'),
        '\n'
    )


def save_local_features(args):  # noqa: WPS210
    """Creates and saves tokens sequence with features.

    Args:
        args: file_name, file_type, processor_args, local_features_params
    """
    file_name, file_type, processor_args, local_features_params = args
    if local_features_params['harmony_labeling_path'] is not None:
        with open(local_features_params['harmony_labeling_path'], 'r') as harmony_labeling_file:
            harmony_labeling = json.load(harmony_labeling_file)
    encoded = encode_midi(file_name, **processor_args)
    if encoded is None:
        return
    tokens = tensor(encoded)
    data = {'tokens': tokens}
    if local_features_params.get('timings'):
        timings, total_time = features_makers.timing_feature(tokens)
        timings = timings * N_TIMING_SEGMENTS // (total_time + 1)
        data['timings'] = timings
    if local_features_params.get('nnotes'):
        data['nnotes'] = features_makers.nnotes_feature(tokens)
    if local_features_params.get('rel_pos'):
        data['rel_pos'] = features_makers.rel_pos_feature(tokens)
    if local_features_params.get('token_type'):
        data['token_type'] = features_makers.token_type(tokens)
    if local_features_params.get('note_type'):
        data['note_type'] = features_makers.note_type(tokens)
    if local_features_params.get('notes_duration'):
        data['notes_duration'] = features_makers.notes_duration(tokens)

    if local_features_params.get('harmony'):
        labeling = harmony_labeling[os.path.basename(file_type)]
        harmony = features_makers.harmony(tokens, labeling)
        data['harmony'] = harmony
    if local_features_params.get('sentiments'):
        labeling = harmony_labeling[os.path.basename(tokens, file_type)]
        sentiments = features_makers.sentiments(labeling)
        data['sentiments'] = sentiments
    if local_features_params.get('tonality'):
        labeling = harmony_labeling[os.path.basename(file_type)]
        tonality = features_makers.tonality(tokens, labeling)
        data['tonality'] = tonality
    processed_midi = pd.DataFrame(data)
    processed_midi.to_csv(
        file_type,
        sep='\t',
        index=False
    )


def get_global_features(args):  # noqa: WPS210
    """Creates json with global features.

    Args:
        args: file_name, file_type, processor_args, global_features_params

    Returns:
        file name, dict of global features for file
    """
    file_name, file_type, global_features_params = args
    global_features = {}
    global_features['midi_file'] = file_name
    if 'genre' in global_features_params and global_features_params['genre']:
        global_features['genre'] = features_makers.genre_feature(file_name)
    if 'author' in global_features_params:
        tokenizer_path = 'tokenizers/authors-tokenizer.json'
        tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab = tokenizer.get_vocab().keys()
        global_features['author'] = features_makers.author_feature(file_name, vocab)
    return file_name, global_features


def save_global_features(  # noqa: WPS210, WPS211
    global_features_pairs,
    files_final_names,
    train,
    val,  # noqa: WPS110
    test,
    output_dir,
    global_features_params
):
    """Saves global features.

    Args:
        global_features_pairs: pairs of file name and its features
        files_final_names: dict of file original path and its final path in dataset
        train: train file names
        val: val file names
        test: test file names
        output_dir: output directory path
        global_features_params: parameters for global features
    """
    global_features = dict(global_features_pairs)

    types_dict = {
        'train': train,
        'val': val,
        'test': test
    }

    for type_name, files_names in types_dict.items():
        type_global_features = {
            standart_file_name(files_final_names[file_name]): global_features[file_name]
            for file_name in files_names
            if os.path.isfile(standart_file_name(files_final_names[file_name]))
        }

        if type_name == 'train':
            create_tokenizers(type_global_features, global_features_params, output_dir)

        json_path = os.path.join(output_dir, type_name, 'global_features.json')
        with open(json_path, 'w') as fp:
            json.dump(type_global_features, fp, indent=4)


def create_tokenizers(global_features, global_features_params, output_dir):
    """Creates tokenizers for global features.

    Args:
        global_features: dict of global features for files
        global_features_params: parameters for global features (from json)
        output_dir: directory to save tokenizers
    """
    for feature in global_features_params:
        if feature == 'midi_file':
            continue
        feature_values = [
            object_features[feature]
            for object_features in global_features.values()
        ]
        tk = Tokenizer(WordLevel(unk_token='[UNK]'))  # noqa S106
        tr = WordLevelTrainer(special_tokens=['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]'])
        tk.pre_tokenizer = Whitespace()

        tk.train_from_iterator(iter(feature_values), tr)
        save_path = os.path.join(output_dir, feature) + '-tokenizer.json'  # noqa: WPS336
        tk.save(save_path)


def tsv_basename(file_path: str, standardize: bool = False):
    """Extracts file name and changes it to tsv.

    Args:
        file_path: file path
        standardize: whether standardize file name

    Returns:
        file name with tsv extension
    """
    file_name = os.path.basename(file_path)
    name, ext = os.path.splitext(file_name)
    if standardize:
        name = standart_file_name(name)
    return name + '.tsv'  # noqa: WPS336


def standart_file_name(file_path: str):
    """Standardizes file name in path.

    Args:
        file_path: file path

    Returns:
        file path with standardized name
    """
    parent_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    file_name = '_'.join(file_name.strip().split())
    return os.path.join(parent_dir, file_name)


def create_dirs(root_dir: str):
    """Creates empty 'train', 'val', 'test' folders.

    Args:
        root_dir: directory to put 'train', 'val', 'test' directories in
    """
    dirs = ['train', 'val', 'test']
    for dir_name in dirs:
        target_dir = os.path.join(root_dir, dir_name)
        os.makedirs(target_dir, exist_ok=True)
        with os.scandir(target_dir) as entries:
            for entry in entries:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry.path)
                else:
                    os.remove(entry.path)


def main():
    """Preprocess files and save tokens to specified output folder.

    Creates train, val, test directories.
    """
    json_path = 'preprocessing/preprocessing_parameters.json'
    with open(json_path, 'r') as model_params_file:
        args = json.load(model_params_file)
    prep_midi(**args)


if __name__ == '__main__':
    main()
