import json
import os
import random
import sys
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from dataset.e_piano import N_TIMING_SEGMENTS, GenerationAdditionalFeatures
from dataset.features_makers import timing_feature
from midi_to_audio.converter import midi_to_audio
from model.music_transformer import MusicTransformer
from preprocessing.processor import START_IDX, decode_midi, encode_midi
from utilities.constants import TOKEN_END, TOKEN_PAD, TORCH_LABEL_TYPE
from utilities.device import get_device, use_cuda

min_note_on = START_IDX['note_on']
max_note_on = START_IDX['note_off'] - 1
min_note_off = START_IDX['note_off']
max_note_off = START_IDX['time_shift'] - 1
min_time_value = START_IDX['time_shift']
max_time_value = START_IDX['velocity'] - 1

time_token_bias = min_time_value - 1

min_start_velocity = 367
max_start_velocity = 377

min_first_note = 41
max_first_note = 57


class Sampler:
    def __init__(self, take_most_probable, control_temperature, std_window) -> None:
        """Inits Sampler class.

        Class is used for handling sampling process.

        Args:
            take_most_probable: bool, whether to take most probable next token
            control_temperature: bool, whether to control temperature
            std_window: window size of controlling temperature via tokens' probas std
        """
        self.softmax = torch.nn.Softmax(dim=-1)
        self.take_most_probable = take_most_probable
        self.control_temperature = control_temperature
        self.std_window = std_window
        self.softmax_std = []

    def sample(self, logits):
        """Samples next token from logits.

        Args:
            logits: tensor of next token's logits

        Returns:
            next token
        """
        token_probas = self.softmax(logits)[:TOKEN_END]
        self.softmax_std.append(token_probas.cpu().std().item())

        # TODO: Move option to config
        #t = 0.7
        #new_logits = torch.nn.functional.gumbel_softmax(logits, t)[:TOKEN_END]
        #distribution = torch.distributions.categorical.Categorical(
        #    probs=new_logits,
        #)
        #return distribution.sample()

        if self.take_most_probable:
            return torch.argmax(token_probas).item()

        if not self.control_temperature or len(self.softmax_std) < self.std_window:
            distribution = torch.distributions.categorical.Categorical(
                probs=token_probas,
            )
            return distribution.sample()

        distribution = torch.distributions.categorical.Categorical(
            logits=logits / (2 - np.log(np.mean(self.softmax_std[-self.std_window:]) + np.e) / 0.9)
        )
        return distribution.sample()

    def plot_probas_std(self, save_path):
        """Plots and saves std distribution of previous samplings.

        Args:
            save_path: path to save the plot
        """
        sns.set(rc={'figure.figsize': (20, 8)})
        sns.lineplot(x=np.arange(len(self.softmax_std)), y=self.softmax_std)
        plt.xticks(rotation=90, fontsize=5)
        plt.savefig(save_path, dpi=300)


def is_time_token(token):
    """Checks if token represents time shift.

    Args:
        token: int token value

    Returns:
        bool whether it is time shift token
    """
    return min_time_value <= token <= max_time_value


def main(args=None):  # noqa: WPS213, WPS210
    """Generates music with parameters from json.

    Args:
        args: idaf what's that
    """
    with open('model/model_params.json', 'r') as model_params_file:
        model_params = json.load(model_params_file)  # noqa: WPS110

    if args is not None and len(args) > 3:
        # if we use gen_script.py and specify special generation parameters
        output_dir = args[1]
        output_name = args[2]
        with open(
            '{0}/generate_sample_{1}_parameters.json'.format(
                output_dir,
                output_name,
            ),
            'r',
        ) as json_file:
            params = json.load(json_file)  # noqa: WPS110
    else:
        with open(  # noqa: WPS440
            'generation/generate_sample_parameters.json',
            'r',
        ) as json_file:
            params = json.load(json_file)  # noqa: WPS110

    random_seed = params['random_seed']
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    if params['force_cpu']:
        use_cuda(False)
        print('WARNING: Forced CPU usage, expect model to perform slower')

    model = MusicTransformer(
        **model_params,
    )
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(params['model_weights']))
    model = model.to(get_device())
    model.eval()

    with torch.set_grad_enabled(False):
        generated_sequence, primary = generate(
            model,
            **params['generation_params'],
        )

    save_generated(
        generated_sequence,
        primary,
        **params['saving_options']
    )


def generate(  # noqa: WPS210, WPS231, WPS213, WPS211
    model,
    additional_features_params=None,
    n_generate=512,
    window_size=None,
    primary_params=None,
    sampler_params=None,
    generate_by_time=False,
    presets=None,
):
    """Generates tokens sequence.

    Generation starts from random token.

    Args:
        model: MusicTransformer model object
        additional_features_params: dict of parameters to use for
            AdditionalFeatures in generation
        n_generate: length of generating sample
        window_size: size of the window to use as input seq for the model
        primary_params: parameters for using primary sample
        sampler_params: parameters for sampling
        generate_by_time: bool whether to
            generate before reaching time limit
        presets: dict of features to use presets for

    Returns:
        generated tokens sequence
    """
    generating_sequence = torch.full(  # noqa: WPS317
        (n_generate,),
        TOKEN_PAD,
        dtype=TORCH_LABEL_TYPE,
    )

    primary = None
    if primary_params['primary_path']:
        primary = torch.tensor(process_primary(**primary_params))
        generating_sequence[:primary.shape[0]] = primary  # noqa: WPS362
        n_primary_tokens = primary_params['n_tokens']
    else:
        # setting velocity
        generating_sequence[0] = np.random.randint(min_start_velocity, max_start_velocity)
        # choosing first note
        generating_sequence[1] = np.random.randint(min_first_note, max_first_note)
        n_primary_tokens = 2
    generating_sequence.to(get_device())

    afg = GenerationAdditionalFeatures(
        n_generate,
        additional_features_params,
        generating_sequence[:n_primary_tokens],
    )
    afg.local_features = afg.local_features.to(get_device())
    local_features = afg.local_features
    afg.global_features = afg.global_features.to(get_device())
    global_features = afg.global_features

    i = 2
    if primary_params['primary_path']:
        i = primary.shape[0]

    desc = 'timing_segments' if generate_by_time else 'tokens'
    total = N_TIMING_SEGMENTS - 1 if generate_by_time else n_generate - i
    pbar = tqdm(total=total, desc=desc)

    if generate_by_time:
        if 'timing' in afg.local_features_indices:
            timing_column, _ = afg.local_features_indices['timing']
            timing_value = local_features[i, timing_column].item()
        else:
            timings, duration = timing_feature(generating_sequence[:i])
            timings = timings * N_TIMING_SEGMENTS // (additional_features_params['timing'] + 1)
            timing_value = timings[-1].item()
        pbar.update(timing_value)

    sampler = Sampler(**sampler_params)

    while i < n_generate:
        begin = 0
        if window_size is not None:
            begin = max(0, i - window_size)
        pred = model(
            generating_sequence[begin:i].unsqueeze(dim=0),
            local_features[begin:i, :].unsqueeze(dim=0),
            global_features.unsqueeze(dim=0),
        )

        next_token_pred = pred[0, i - 1 - begin]
        next_token = sampler.sample(next_token_pred)
        generating_sequence[i] = next_token

        afg.update_additional_features(generating_sequence[:i + 1])

        if next_token == TOKEN_END:
            pbar.close()
            print('Model called end of sequence.')
            break

        if generate_by_time:
            if 'timing' in afg.local_features_indices:
                timing_column, _ = afg.local_features_indices['timing']
                timing_value = local_features[i, timing_column].item()
                prev_timing_value = local_features[i - 1, timing_column].item()
            else:
                timings, duration = timing_feature(generating_sequence[:i])
                timings = timings * N_TIMING_SEGMENTS // (additional_features_params['timing'] + 1)
                timing_value = timings[-1].item()
                prev_timing_value = timings[-2].item()
            if timing_value >= N_TIMING_SEGMENTS:
                pbar.update(N_TIMING_SEGMENTS - prev_timing_value)
                break
            if timing_value > prev_timing_value:
                pbar.update(timing_value - prev_timing_value)
        else:
            pbar.update(1)
        i += 1
        set_features(afg, presets)
    pbar.close()
    return generating_sequence[:i], primary


def set_features(afg: GenerationAdditionalFeatures, presets: Optional[Dict[str, bool]]):
    """Sets features according to timings.

    Args:
        afg: GenerationAdditionalFeatures for additional features
        presets: dict of features to use presets for
    """
    features_setters = {
        'nnotes': set_nnotes,
    }
    for feature, status in presets.items():
        if status:
            features_setters[feature](afg)


def set_nnotes(afg: GenerationAdditionalFeatures):
    """Sets nnotes feature according to timings.

    Args:
        afg: GenerationAdditionalFeatures for additional features
    """
    begin, end = afg.local_features_indices['timing']
    timings = afg.local_features[:, begin:end]

    begin, end = afg.local_features_indices['nnotes']
    nnotes = afg.local_features[:, begin:end]

    genre = afg.parameters['genre']

    with open('presets/genre_timings.json', 'r') as fp:
        genre_nnotes = json.load(fp)[genre]
    for timing in range(N_TIMING_SEGMENTS):
        nnotes[timings == timing] = genre_nnotes[timing]


def process_primary(
    primary_path: str,
    n_tokens: int,
    duration: float
) -> np.array:
    """Loads primary tokens. Saves them to 'output_dir'.

    Args:
        primary_path: path to primary file
        n_tokens: number of tokens to take
        duration: length to take from primary in seconds

    Returns:
        primary tokens
    """
    name, ext = os.path.splitext(primary_path)
    if ext == '.tsv':
        data = pd.read_csv(primary_path, sep='\t')
        tokens = data['tokens'].to_numpy()
    if ext in {'.midi', '.mid'}:
        tokens = encode_midi(primary_path)

    if n_tokens is not None and n_tokens != -1:
        tokens = tokens[:n_tokens]
    if duration is not None and duration != -1:
        timings, total_duration = timing_feature(torch.tensor(tokens))
        idx = (timings > duration * 100).type(torch.int32).argmax().item()
        if idx != 0:
            tokens = tokens[:idx]
    return tokens


def save_generated(
    tokens: torch.Tensor,
    primary: Optional[torch.Tensor],
    output_dir: str,
    output_folder_name: str,
    save_params: bool,
    save_primary: bool,
    save_mp3: bool
):
    """Saves generated sample and additional parameters.

    Args:
        tokens: tensor of generated tokens,
        primary: tensor of primary tokens,
        output_dir: path to directory for saving,
        output_folder_name: name of the folder to use for save,
        save_params: bool whether to save used params as json,
        save_primary: bool whether to save primary as midi,
        save_mp3: bool whether to save generated as mp3
    """
    os.makedirs(output_dir, exist_ok=True)
    folder = os.path.join(output_dir, output_folder_name)
    os.makedirs(folder, exist_ok=True)

    if save_params:
        with open(os.path.join(folder, f'{output_folder_name}_params.json'), 'w') as json_output:
            with open(  # noqa: WPS440
                'generation/generate_sample_parameters.json',
                'r',
            ) as json_file:
                params = json.load(json_file)  # noqa: WPS110
                json.dump(params, json_output, indent=4)

    if save_primary and primary is not None:
        decode_midi(primary.cpu().numpy(), os.path.join(folder, 'primary.midi'))

    midi_path = os.path.join(folder, f'{output_folder_name}.midi')
    print(midi_path)
    decode_midi(
        tokens.cpu().numpy(),
        file_name=midi_path,
    )
    if save_mp3:
        mp3_path = os.path.join(folder, f'{output_folder_name}.mp3')
        midi_to_audio(
            tokens=tokens.cpu().numpy(),
            file_name=mp3_path,
        )


if __name__ == '__main__':
    main(sys.argv)
