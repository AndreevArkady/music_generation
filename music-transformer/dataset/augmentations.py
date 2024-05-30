import json

import numpy as np
import torch

from preprocessing.processor import START_IDX

min_time_value = START_IDX['time_shift']
max_time_value = START_IDX['velocity'] - 1
min_note_on = START_IDX['note_on']
max_note_on = START_IDX['note_off'] - 1
min_note_off = START_IDX['note_off']
max_note_off = START_IDX['time_shift'] - 1
min_velocity = START_IDX['velocity']
max_velocity = START_IDX['end_of_scope'] - 1


with open('dataset/augmentations_parameters.json', 'r') as params_file:
    params = json.load(params_file)


def apply_augmentations(tokens):
    """Applies augmentations for tokens sequence.

    Args:
        tokens: 1d tensor of tokens sequence

    Returns:
        augmented tokens
    """
    if params['shift_volume']['enabled']:
        tokens, _ = shift_volume(tokens)

    if params['move_octave']['enabled']:
        tokens, _ = move_octave(tokens)

    if params['change_tempo']['enabled']:
        tokens, _ = change_tempo(tokens)

    if params['shift_velocity']['enabled']:
        tokens, _ = shift_velocity(tokens)

    return tokens


def shift_volume(tokens):
    """Slightly shifts volume.

    Args:
        tokens: 1d tensor of tokens sequence

    Returns:
        shifted tokens
        shift value
    """
    min_shift, max_shift = params['shift_volume']['shift_range']
    shift_val = np.random.choice(np.arange(min_shift, max_shift + 1))
    if shift_val == 0:
        return tokens, 0

    res = tokens.detach().clone()

    if shift_val < 0:
        res[(min_note_on - shift_val <= res) & (res <= max_note_on)] += shift_val
        res[(min_note_off - shift_val <= res) & (res <= max_note_off)] += shift_val
    else:
        res[(min_note_on <= res) & (res <= max_note_on - shift_val)] += shift_val
        res[(min_note_off <= res) & (res <= max_note_off - shift_val)] += shift_val

    return res, shift_val

def move_octave(tokens):
    """Moves octave.

    Args:
        tokens: 1d tensor of tokens sequence

    Returns:
        shifted tokens
        shift value
    """
    octaves_shifts = np.array(params['move_octave']['octaves_shifts'])
    shifts_probabilities = params['move_octave']['shifts_probabilities']
    tones_in_octave = 12
    shift_val = np.random.choice(
        octaves_shifts * tones_in_octave,
        p=shifts_probabilities,
    )
    if shift_val == 0:
        return tokens, 0

    res = tokens.detach().clone()

    if shift_val < 0:
        res[(min_note_on - shift_val <= res) & (res <= max_note_on)] += shift_val
        res[(min_note_off - shift_val <= res) & (res <= max_note_off)] += shift_val
    else:
        res[(min_note_on <= res) & (res <= max_note_on - shift_val)] += shift_val
        res[(min_note_off <= res) & (res <= max_note_off - shift_val)] += shift_val

    return res, shift_val


def change_tempo(tokens):  # noqa: WPS210
    """Slightly changes tempo.

    Args:
        tokens: 1d tensor of tokens sequence

    Returns:
        tokens with changed tempo
        coefficient of speeding up
    """
    res = tokens.detach().clone().to(torch.float)
    time_tokens_idx = (min_time_value <= tokens) & (tokens <= max_time_value)

    max_time_token = torch.max(tokens).item()
    min_time_token = torch.min(tokens).item()

    max_coef = (max_time_value - min_time_value + 1) / (max_time_token - min_time_value + 1)
    min_coef = 1 / (min_time_token - min_time_value + 1)

    mean = params['change_tempo']['mean']
    scale = params['change_tempo']['scale']
    coef = np.random.normal(mean, scale)
    coef = min(max_coef, coef)
    coef = max(min_coef, coef)

    res[time_tokens_idx] -= min_time_value - 1
    res[time_tokens_idx] *= coef
    res[time_tokens_idx] += min_time_value - 1

    if coef < 1:
        res = torch.ceil(res).long()
    else:
        res = torch.floor(res).long()

    return res, coef


def shift_velocity(tokens):  # noqa: WPS210
    """Slightly changes velocity.

    Args:
        tokens: 1d tensor of tokens sequence

    Returns:
        tokens with changed velocity
        coefficient of changing velocity
    """
    res = tokens.detach().clone().to(torch.float)
    min_shift, max_shift = params['shift_velocity']['shift_range']
    shift_val = np.random.uniform(min_shift, max_shift)
    idx = (min_velocity <= res) & (res <= max_velocity)
    res[idx] *= shift_val
    res[idx] = res[idx].clip(min_velocity + 1, max_velocity)

    return res.long(), shift_val
