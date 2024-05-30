from collections import deque

from torch import arange, cumsum, zeros_like

from preprocessing.processor import START_IDX
from utilities.constants import GENRES_LIST


min_time_value = START_IDX['time_shift']
max_time_value = START_IDX['velocity'] - 1
min_note_on = START_IDX['note_on']
max_note_on = START_IDX['note_off'] - 1
min_note_off = START_IDX['note_off']
max_note_off = START_IDX['time_shift'] - 1
min_velocity = START_IDX['velocity']
max_velocity = START_IDX['end_of_scope'] - 1

n_tokens_segments = 128
tones_in_octave = 12


def timing_feature(tokens):
    """Get timing feature from token sequences.

    Get timing in range [0, total_duration] for every token.

    Args:
        tokens: 1d Tensor or np.ndarray of sequence of tokens

    Returns:
        sequence of timings and total_duration
    """
    times = zeros_like(tokens)
    times[1:] = tokens[:-1].detach().clone()  # noqa: WPS362
    not_time_tokens_idx = (min_time_value > times) | (times > max_time_value)
    times -= min_time_value - 1
    times[not_time_tokens_idx] = 0
    res = cumsum(times, dim=0)
    return res, res[-1].item()


def nnotes_feature(tokens):
    """Get number of notes feature from token sequences.

    Represents how many notes are pressed with current token

    Args:
        tokens: 1d Tensor or np.ndarray of sequence of tokens

    Returns:
        sequence of number of notes
    """
    nnotes = zeros_like(tokens)
    note_on_tokens_idx = (min_note_on < tokens) & (tokens < max_note_on)
    nnotes[note_on_tokens_idx] = 1
    note_off_tokens_idx = (min_note_off < tokens) & (tokens < max_note_off)
    nnotes[note_off_tokens_idx] = -1
    res = cumsum(nnotes, dim=0)
    return res.clip(0, 10)


def rel_pos_feature(tokens):
    """Get relative position feature from token sequences.

    Get relative position as integer in range [0, 128] for every token.

    Args:
        tokens: 1d Tensor or np.ndarray of sequence of tokens

    Returns:
        sequence of relative position and total number of tokens
    """
    rel_pos = arange(tokens.shape[0])
    return rel_pos * n_tokens_segments // tokens.shape[0]


def genre_feature(path):
    """Creates genre feature.

    Args:
        path: path to midi file

    Returns:
        genre
    """
    file_name_lower = path.lower()
    for genre in GENRES_LIST:
        if genre in file_name_lower:
            return genre
    return '[UNK]'


def author_feature(path, vocab):
    """Extracts author from MIDI file path.

    Args:
        path: MIDI file path
        vocab: authors vocabulary

    Returns:
        author string
    """
    path = path.replace(', ', '_').replace(' ', '_').lower()
    for author in vocab:
        if author in path:
            return author
    return '[UNK]'


def token_type(tokens):
    """Creates token type feature.

    Args:
        tokens: 1d Tensor of sequence of tokens

    Returns:
        tokens' types Tensor
            0: time_shift
            1: note_on
            2: note_off
            3: velocity
    """
    types = zeros_like(tokens)
    types[(tokens >= min_note_on) & (tokens <= max_note_on)] = 1
    types[(tokens >= min_note_off) & (tokens <= max_note_off)] = 2
    types[(tokens >= min_velocity) & (tokens <= max_velocity)] = 3
    return types


def note_type(tokens):
    """Creates note type feature.

    Type is a name of token in octave.

    Args:
        tokens: 1d Tensor of sequence of tokens

    Returns:
        notes' types Tensor
    """
    types = zeros_like(tokens)
    on_ids = (tokens >= min_note_on) & (tokens <= max_note_on)
    off_ids = (tokens >= min_note_off) & (tokens <= max_note_off)
    types[on_ids] = (tokens[on_ids] - min_note_on) % tones_in_octave + 1
    types[off_ids] = (tokens[off_ids] - min_note_off) % tones_in_octave + 1
    return types


def notes_duration(tokens, window_size=5):
    """Creates last notes duration feature.

    Feature is a mean value of n last notes durations.
    n = 5 by default.

    Args:
        tokens: 1d Tensor of sequence of tokens
        window_size: number of last notes durations to use

    Returns:
        notes' types Tensor
    """
    time = 0
    notes_starts = {}
    mean_durations = zeros_like(tokens)
    durations_window = deque()
    for idx, token in enumerate(tokens):
        token = token.item()
        if min_note_on <= token <= max_note_on:
            token -= min_note_on
            if token not in notes_starts:
                notes_starts[token] = time
        elif min_note_off <= token <= max_note_off:
            token -= min_note_off
            if token in notes_starts:
                duration = time - notes_starts[token]
                del notes_starts[token]
                durations_window.append(duration)
                if len(durations_window) > window_size:
                    durations_window.popleft()
        elif min_time_value <= token <= max_time_value:
            time += token - min_time_value + 1
        mean_duration = sum(durations_window) / len(durations_window) if durations_window else 0
        mean_durations[idx] = round(mean_duration)
    mean_durations[mean_durations > 200] = 200
    return mean_durations


def harmony(tokens, harmony_labeling):
    """Creates harmony feature from harmony labelling.

    Args:
        tokens: tokenized composition
        harmony_labeling: harmony labelling

    Returns:
        harmony feature
    """
    harmony = zeros_like(tokens)
    for segment in harmony_labeling:
        begin, end, tonality = segment
        harmony[begin:end] = tonality
    return harmony


def sentiments(tokens, harmony_labeling):
    """Creates sentiments feature from harmony labelling.

    Args:
        tokens: tokenized composition
        harmony_labeling: harmony labelling

    Returns:
        sentiments feature
    """
    sentiments = zeros_like(tokens)
    for segment in harmony_labeling:
        begin, end, tonality = segment
        sentiments[begin:end] = 1 if tonality < tones_in_octave else 0
    return sentiments


def tonality(tokens, harmony_labeling):
    """Creates harmony feature from harmony labelling.

    Args:
        tokens: tokenized composition
        harmony_labeling: harmony labelling

    Returns:
        harmony feature
    """
    tonality = zeros_like(tokens)
    for segment in harmony_labeling:
        begin, end, tonality_value = segment
        tonic = tonality_value
        if tonality_value >= tones_in_octave:
            tonic -= tones_in_octave
        tonality[begin:end] = tonic
    return tonality
