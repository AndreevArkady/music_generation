import argparse
import copy
import itertools
import json
import logging
import math
import os
import sys
from multiprocessing import Pool
from typing import List, Tuple

import coloredlogs
import numpy as np
import tqdm
from pretty_midi import PrettyMIDI

# Source with more advanced musical features:
# https://github.com/ruiguo-bio/midi-miner/blob/master/tension_calculation.py

PianoRoll = np.ndarray

major_enharmonics = {'C#': 'D-', 'D#': 'E-', 'F#': 'G-', 'G#': 'A-', 'A#': 'B-'}

minor_enharmonics = {'D-': 'C#', 'D#': 'E-', 'G-': 'F#', 'A-': 'G#', 'A#': 'B-'}

octave = 12

pitch_index_to_sharp_names = np.array(
    ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
)


pitch_index_to_flat_names = np.array(
    ['C', 'D-', 'D', 'E-', 'E', 'F', 'G-', 'G', 'A-', 'A', 'B-', 'B'],
)


pitch_name_to_pitch_index = {
    'G-': -6,
    'D-': -5,
    'A-': -4,
    'E-': -3,
    'B-': -2,
    'F': -1,
    'C': 0,
    'G': 1,
    'D': 2,
    'A': 3,
    'E': 4,
    'B': 5,
    'F#': 6,
    'C#': 7,
    'G#': 8,
    'D#': 9,
    'A#': 10,
}

pitch_index_to_pitch_name = {v: k for k, v in pitch_name_to_pitch_index.items()}

valid_major = ['G-', 'D-', 'A-', 'E-', 'B-', 'F', 'C', 'G', 'D', 'A', 'E', 'B']

valid_minor = ['E-', 'B-', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#']

enharmonic_dict = {'F#': 'G-', 'C#': 'D-', 'G#': 'A-', 'D#': 'E-', 'A#': 'B-'}
enharmonic_reverse_dict = {v: k for k, v in enharmonic_dict.items()}

all_key_names = [
    'C major',
    'G major',
    'D major',
    'A major',
    'E major',
    'B major',
    'F major',
    'B- major',
    'E- major',
    'A- major',
    'D- major',
    'G- major',
    'A minor',
    'E minor',
    'B minor',
    'F# minor',
    'C# minor',
    'G# minor',
    'D minor',
    'G minor',
    'C minor',
    'F minor',
    'B- minor',
    'E- minor',
]


# use ['C','D-','D','E-','E','F','F#','G','A-','A','B-','B'] to map the
# midi to pitch name
note_index_to_pitch_index = [0, -5, 2, -3, 4, -1, -6, 1, -4, 3, -2, 5]

weight = np.array([0.536, 0.274, 0.19])
alpha = 0.75
beta = 0.75
vertical_step = 0.4
radius = 1.0


def cal_diameter(
    piano_roll: PianoRoll,
    key_index: int,
    key_change_beat=-1,
    changed_key_index=-1,
) -> List[int]:

    diameters = []

    for i in range(0, piano_roll.shape[1]):
        indices = []
        for index, j in enumerate(piano_roll[:, i]):
            if j > 0:
                if i / 4 > key_change_beat and key_change_beat != -1:
                    shifted_index = index % octave - changed_key_index
                    if shifted_index < 0:
                        shifted_index += octave
                else:
                    shifted_index = index % octave - key_index
                    if shifted_index < 0:
                        shifted_index += octave

                indices.append(note_index_to_pitch_index[shifted_index])
        diameters.append(largest_distance(indices))

    return diameters


def largest_distance(pitches: List[int]) -> int:
    if len(pitches) < 2:
        return 0
    diameter = 0
    pitch_pairs = itertools.combinations(pitches, 2)
    for pitch_pair in pitch_pairs:
        distance = np.linalg.norm(
            pitch_index_to_position(
                pitch_pair[0],
            ) - pitch_index_to_position(pitch_pair[1]),
        )
        if distance > diameter:
            diameter = distance
    return diameter


def piano_roll_to_ce(piano_roll: PianoRoll, shift: int) -> np.ndarray:

    pitch_index = []
    for i in range(0, piano_roll.shape[1]):
        indices = []
        for index, j in enumerate(piano_roll[:, i]):
            if j > 0:
                shifted_index = index % octave - shift
                if shifted_index < 0:
                    shifted_index += octave

                indices.append(note_index_to_pitch_index[shifted_index])

        pitch_index.append(indices)

    return ce_sum(pitch_index)


def notes_to_ce(notes: List[int], shift: int) -> np.ndarray:
    indices = []

    for index, j in enumerate(notes):
        if j > 0:

            shifted_index = index % octave - shift
            if shifted_index < 0:
                shifted_index += octave

            indices.append(note_index_to_pitch_index[shifted_index])

    total = np.zeros(3)
    count = 0
    for ind in indices:
        total += pitch_index_to_position(ind)
        count += 1

    if count != 0:
        total /= count
    return total


def pitch_index_to_position(pitch_index: int) -> np.ndarray:

    c = pitch_index - (4 * (pitch_index // 4))

    pos = np.zeros((3,))

    if c == 0:
        pos[1] = radius
    if c == 1:
        pos[0] = radius
    if c == 2:
        pos[1] = -1 * radius
    if c == 3:
        pos[0] = -1 * radius

    pos[2] = pitch_index * vertical_step
    return np.array(pos)


def ce_sum(indices: List[int], start=None, end=None) -> np.ndarray:
    if not start:
        start = 0
    if not end:
        end = len(indices)

    indices = indices[start:end]
    total = np.zeros(3)
    count = 0
    for data in indices:
        for pitch in data:
            total += pitch_index_to_position(pitch)
            count += 1
    return total / count


def major_triad_position(root_index: int) -> np.ndarray:
    root_pos = pitch_index_to_position(root_index)

    fifth_index = root_index + 1
    third_index = root_index + 4

    fifth_pos = pitch_index_to_position(fifth_index)
    third_pos = pitch_index_to_position(third_index)

    return weight[0] * root_pos + weight[1] * fifth_pos + weight[2] * third_pos


def minor_triad_position(root_index: int) -> np.ndarray:
    root_pos = pitch_index_to_position(root_index)

    fifth_index = root_index + 1
    third_index = root_index - 3

    fifth_pos = pitch_index_to_position(fifth_index)
    third_pos = pitch_index_to_position(third_index)

    return weight[0] * root_pos + weight[1] * fifth_pos + weight[2] * third_pos


def major_key_position(key_index: int) -> np.ndarray:
    root_triad_pos = major_triad_position(key_index)
    fifth_index = key_index + 1

    fourth_index = key_index - 1

    fifth_triad_pos = major_triad_position(fifth_index)
    fourth_triad_pos = major_triad_position(fourth_index)

    return (
        weight[0] * root_triad_pos +
        weight[1] * fifth_triad_pos +
        weight[2] * fourth_triad_pos
    )


def minor_key_position(key_index: int) -> np.ndarray:

    root_triad_pos = minor_triad_position(key_index)
    fifth_index = key_index + 1
    fourth_index = key_index - 1
    major_fourth_triad_pos = major_triad_position(fourth_index)
    minor_fourth_triad_pos = minor_triad_position(fourth_index)

    major_fifth_triad_pos = major_triad_position(fifth_index)
    minor_fifth_triad_pos = minor_triad_position(fifth_index)

    fifth_pos = (
        alpha * major_fifth_triad_pos +
        (1 - alpha) * minor_fifth_triad_pos
    )
    forth_pos = (
        beta * minor_fourth_triad_pos +
        (1 - beta) * major_fourth_triad_pos
    )

    return (
        weight[0] * root_triad_pos +
        weight[1] * fifth_pos +
        weight[2] * forth_pos
    )


def cal_key(
    piano_roll: PianoRoll,
    key_names: str,
    end_ratio=0.5,
) -> Tuple[str, int, int]:
    # use the song to the place of end_ratio to find the key
    # for classical it should be less than 0.2
    end = int(piano_roll.shape[1] * end_ratio)
    distances = []
    key_positions = []
    key_indices = []
    key_shifts = []
    for name in key_names:
        key = name.split()[0].upper()
        mode = name.split()[1]

        if mode == 'minor':
            if key not in valid_minor:
                if key in enharmonic_dict:
                    key = enharmonic_dict.get(key)
                elif key in enharmonic_reverse_dict:
                    key = enharmonic_reverse_dict.get(key)
                else:
                    logger.info('no such key')
            if key not in valid_minor:
                logger.info('no such key')
                return None

        else:
            if key not in valid_major:
                if key in enharmonic_dict:
                    key = enharmonic_dict.get(key)
                elif key in enharmonic_reverse_dict:
                    key = enharmonic_reverse_dict.get(key)
                else:
                    logger.info('no such key')
            if key not in valid_major:
                logger.info('no such key')
                return None
        key_index = pitch_name_to_pitch_index.get(key)

        if mode == 'minor':
            # all the minor key_pos is a minor
            key_pos = minor_key_position(3)
        else:
            # all the major key_pos is C major
            key_pos = major_key_position(0)
        key_positions.append(key_pos)

        if mode == 'minor':
            key_index -= 3
        key_shift_name = pitch_index_to_pitch_name[key_index]

        if key_shift_name in pitch_index_to_sharp_names:
            key_shift_for_ce = np.argwhere(
                pitch_index_to_sharp_names == key_shift_name,
            )[0][0]
        else:
            key_shift_for_ce = np.argwhere(
                pitch_index_to_flat_names == key_shift_name,
            )[0][0]
        key_shifts.append(key_shift_for_ce)
        ce = piano_roll_to_ce(piano_roll[:, :end], key_shift_for_ce)
        distance = np.linalg.norm(ce - key_pos)
        distances.append(distance)
        key_indices.append(key_index)

    index = np.argmin(np.array(distances))
    key_name = key_names[index]
    key_pos = key_positions[index]
    key_shift_for_ce = key_shifts[index]
    return key_name, key_pos, key_shift_for_ce


def pianoroll_to_pitch(pianoroll: PianoRoll) -> np.ndarray:
    pitch_roll = np.zeros((octave, pianoroll.shape[1]))
    for i in range(0, pianoroll.shape[0] - octave + 1, octave):
        pitch_roll = np.add(pitch_roll, pianoroll[i:i + octave])
    return np.transpose(pitch_roll)


def note_to_index(pianoroll: PianoRoll) -> np.ndarray:
    note_ind = np.zeros((128, pianoroll.shape[1]))
    for i in range(0, pianoroll.shape[1]):
        step = []
        for j, note in enumerate(pianoroll[:, i]):
            if note != 0:
                step.append(j)
        if step:
            note_ind[step[-1], i] = 1
    return np.transpose(note_ind)


def merge_tension(
    metric: List[float],
    beat_indices: List[int],
    down_beat_indices: List[int],
    window_size=-1,
) -> np.ndarray:

    # every bar window
    if window_size == -1:

        new_metric = []

        for i in range(len(down_beat_indices) - 1):
            new_metric.append(
                np.mean(
                    metric[down_beat_indices[i]:down_beat_indices[i + 1]],
                    axis=0,
                ),
            )

    else:
        new_metric = []

        for j in range(0, len(beat_indices) - window_size, window_size):
            new_metric.append(
                np.mean(
                    metric[beat_indices[j]:beat_indices[j + window_size]],
                    axis=0,
                ),
            )

    return np.array(new_metric)


def moving_average(tension: np.ndarray, window=4) -> np.ndarray:

    # size moving window, the output size is the same
    outputs = []
    zeros = np.zeros((window,), dtype=tension.dtype)

    tension = np.concatenate([tension, zeros], axis=0)
    for i in range(0, tension.shape[0] - window + 1):
        outputs.append(np.mean(tension[i:i + window]))
    return np.array(outputs)


def cal_tension(
    file_name: str,
    piano_roll: PianoRoll,
    sixteenth_time: np.ndarray,
    beat_time: np.ndarray,
    beat_indices: List[int],
    down_beat_time: np.ndarray,
    down_beat_indices: List[int],
    output_folder: str,
    window_size=1,
    key_name='',
):
    try:

        key_name, key_pos, note_shift = cal_key(
            piano_roll, key_name, end_ratio=args.end_ratio,
        )
        return key_name

    except (
        ValueError,
        EOFError,
        IndexError,
        OSError,
        KeyError,
        ZeroDivisionError,
    ) as exc:
        exception_str = 'Unexpected error in {0}:\n{1} {2}'.format(
            file_name,
            exc,
            sys.exc_info()[0],
        )
        logger.info(exception_str)


def get_key_index_change(
    pm: PianoRoll,
    start_time: float,
    sixteenth_time: np.ndarray,
):
    new_pm = copy.deepcopy(pm)
    for instrument in new_pm.instruments:
        for i, note in enumerate(instrument.notes):
            if note.start > start_time:
                instrument.notes = instrument.notes[i:]
                break

    piano_roll = get_piano_roll(new_pm, sixteenth_time)
    key_name = all_key_names

    # WPS331 Found variables that are only used for `return`: piano_roll

    return cal_key(piano_roll, key_name, end_ratio=1)


def note_pitch(melody_track: np.ndarray) -> List[float]:
    pitch_sum = []
    for i in range(0, melody_track.shape[1]):
        indices = []
        for index, j in enumerate(melody_track[:, i]):
            if j > 0:
                indices.append(index - 24)

        pitch_sum.append(np.mean(indices))
    return pitch_sum


def get_piano_roll(pm: PrettyMIDI, beat_times: np.ndarray) -> PianoRoll:
    piano_roll = pm.get_piano_roll(times=beat_times)
    np.nan_to_num(piano_roll, copy=False)
    piano_roll = piano_roll > 0
    return piano_roll.astype(int)


def cal_centroid(
    piano_roll: PianoRoll,
    key_index: int,
    key_change_beat=-1,
    changed_key_index=-1,
):
    centroids = []
    for time_step in range(0, piano_roll.shape[1]):

        roll = piano_roll[:, time_step]

        if key_change_beat == -1:
            centroids.append(notes_to_ce(roll, key_index))
        else:
            if time_step / 4 > key_change_beat:
                centroids.append(notes_to_ce(roll, changed_key_index))
            else:
                centroids.append(notes_to_ce(roll, key_index))
    return centroids


def detect_key_change(
    key_diff: np.ndarray,
    diameter: np.ndarray,
    start_ratio=0.5,
) -> int:
    # 8 bar window
    key_diff_ratios = []
    fill_one = False
    steps = 4
    for i in range(8, key_diff.shape[0] - 8):
        if fill_one and steps > 0:
            key_diff_ratios.append(1)
            steps -= 1
            if steps == 0:
                fill_one = False
            continue

        if np.any(key_diff[i - 4:i]) and np.any(key_diff[i:i + 4]):
            previous = np.mean(key_diff[i - 4:i])
            current = np.mean(key_diff[i:i + 4])
            key_diff_ratios.append(current / previous)
        else:
            fill_one = True
            steps = 4

    len_range = int(len(key_diff_ratios) * start_ratio)
    step_range = len(key_diff_ratios) - 2
    for j in range(len_range, step_range):
        if np.mean(key_diff_ratios[j:j + 4]) > 2:
            key_diff_change_bar = j
            break
    else:
        key_diff_change_bar = -1

    if key_diff_change_bar != -1:
        return key_diff_change_bar + octave
    return key_diff_change_bar


def remove_drum_track(pm: PrettyMIDI) -> PrettyMIDI:
    instrument_idx = []
    for idx, intrument in enumerate(pm.instruments):
        if intrument.is_drum:
            instrument_idx.append(idx)
    for inv_idx in instrument_idx[::-1]:
        del pm.instruments[inv_idx]  # noqa: WPS420
    return pm


def get_beat_time(
    pm: PrettyMIDI,
    beat_division=4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
    beats = pm.get_beats()

    beats = np.unique(beats, axis=0)

    divided_beats = []
    for i in range(len(beats) - 1):
        for j in range(beat_division):
            divided_beats.append(
                (beats[i + 1] - beats[i]) / beat_division * j + beats[i],
            )
    divided_beats.append(beats[-1])
    divided_beats = np.unique(divided_beats, axis=0)

    beat_indices = []
    for beat in beats:
        beat_indices.append(np.argwhere(divided_beats == beat)[0][0])

    down_beats = pm.get_downbeats()
    if divided_beats[-1] > down_beats[-1]:
        down_beats = np.append(
            down_beats, down_beats[-1] - down_beats[-2] + down_beats[-1],
        )

    down_beats = np.unique(down_beats, axis=0)

    down_beat_indices = []
    for down_beat in down_beats:

        down_beat_indices.append(np.argmin(np.abs(down_beat - divided_beats)))

    return (
        np.array(divided_beats),
        np.array(beats),
        np.array(down_beats),
        beat_indices,
        down_beat_indices,
    )


def extract_notes(
    file_name: str,
    track_num: int,
) -> Tuple[PrettyMIDI, PianoRoll, np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
    try:
        pm = PrettyMIDI(file_name)
        pm = remove_drum_track(pm)

        if track_num != 0:
            if len(pm.instruments) < track_num:
                logger.warning(' '.join([
                    'the file {0} has {1} tracks,'.format(
                        file_name,
                        len(pm.instruments),
                    ),
                    f'less than the required track num {track_num}.',
                    'Use all the tracks',
                ]))
            pm.instruments = pm.instruments[:track_num]

        (
            sixteenth_time,
            beat_time,
            down_beat_time,
            beat_indices,
            down_beat_indices,
        ) = get_beat_time(pm, beat_division=4)

        piano_roll = get_piano_roll(pm, sixteenth_time)

    except (
        ValueError,
        EOFError,
        IndexError,
        OSError,
        KeyError,
        ZeroDivisionError,
    ) as exc:
        exception_str = 'Unexpected error in {0}:\n{1} {2}'.format(
            file_name,
            exc,
            sys.exc_info()[0],
        )
        logger.info(exception_str)
        return None

    return (
        pm,
        piano_roll,
        sixteenth_time,
        beat_time,
        down_beat_time,
        beat_indices,
        down_beat_indices,
    )


def walk(folder_name: str) -> List[str]:
    files = []
    for path, _, all_files in os.walk(folder_name):
        for file_name in sorted(all_files):
            endname = file_name.split('.')[-1].lower()
            if 'mid' in endname:
                files.append(os.path.join(path, file_name))
    return files


def get_args(default='.') -> argparse.Namespace:
    default_pool_num = 25
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_folder',
        default=default,
        type=str,
        help='MIDI file input folder',
    )

    parser.add_argument(
        '-f',
        '--file_name',
        default='',
        type=str,
        help='input MIDI file name',
    )

    parser.add_argument(
        '-o',
        '--output_folder',
        default=default,
        type=str,
        help='MIDI file output folder',
    )

    parser.add_argument(
        '-p',
        '--pool_num',
        default=default_pool_num,
        type=int,
        help='number of processes for tension calculation',
    )

    parser.add_argument(
        '-w',
        '--window_size',
        default=-1,
        type=int,
        help=' '.join([
            'Tension calculation window size,',
            '1 for a beat, 2 for 2 beat etc., -1 for a downbeat',
        ]),
    )

    parser.add_argument(
        '-n',
        '--key_name',
        default='',
        type=str,
        help='key name of the song, e.g. B- major, C# minor',
    )

    parser.add_argument(
        '-t',
        '--track_num',
        default=0,
        type=int,
        help=' '.join([
            'number of tracks used to calculate tension,',
            'e.g. 3 means use first 3 tracks,',
            'default 0 means use all',
        ]),
    )

    parser.add_argument(
        '-r',
        '--end_ratio',
        default=0.5,
        type=float,
        help=' '.join([
            'the place to find the first key',
            'of the song, 0.5 means the first key',
            'is calculate by the first half the song',
        ]),
    )

    parser.add_argument(
        '-k',
        '--key_changed',
        default=False,
        type=bool,
        help='try to find key change, default false',
    )

    parser.add_argument(
        '-v',
        '--vertical_step',
        default=vertical_step,
        type=float,
        help=' '.join([
            'the vertical step parameter in the spiral array,',
            'which should be set between sqrt(2/15) and sqrt(0.2)',
        ]),
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose log output',
    )

    return parser.parse_args()


def note_to_key_pos(note_indices: List[int], key_pos: int) -> np.ndarray:
    note_positions = []
    for note_index in note_indices:
        note_positions.append(
            pitch_index_to_position(note_index_to_pitch_index[note_index]),
        )
    # diffs = np.linalg.norm(np.array(note_positions) - key_pos, axis=1)
    return np.linalg.norm(np.array(note_positions) - key_pos, axis=1)


def note_to_note_pos(note_indices: List[int], note_pos: int) -> np.ndarray:
    note_positions = []
    for note_index in note_indices:
        note_positions.append(
            pitch_index_to_position(note_index_to_pitch_index[note_index]),
        )
    # diffs = np.linalg.norm(np.array(note_positions) - note_pos, axis=1)
    return np.linalg.norm(np.array(note_positions) - note_pos, axis=1)


def chord_to_key_pos(chord_indices: List[int], key_pos: int) -> np.ndarray:
    chord_positions = []
    for chord_index_maj in chord_indices:
        chord_positions.append(
            major_triad_position(note_index_to_pitch_index[chord_index_maj]),
        )

    for chord_index_min in chord_indices:
        chord_positions.append(
            minor_triad_position(note_index_to_pitch_index[chord_index_min]),
        )
    # diffs = np.linalg.norm(np.array(chord_positions) - key_pos, axis=1)
    return np.linalg.norm(np.array(chord_positions) - key_pos, axis=1)


def key_to_key_pos(key_indices: List[int], key_pos: int) -> np.ndarray:
    key_positions = []
    for key_index_maj in key_indices:
        key_positions.append(
            major_key_position(note_index_to_pitch_index[key_index_maj]),
        )

    for key_index_min in key_indices:
        key_positions.append(
            minor_key_position(note_index_to_pitch_index[key_index_min]),
        )

    return np.linalg.norm(np.array(key_positions) - key_pos, axis=1)


def keys_per_token(task):

    file_name, args = task

    base_name = os.path.basename(file_name)

    try:
        res = extract_notes(file_name, args.track_num)
    except Exception:
        res = None

    if res is None:
        return file_name, None, base_name, None
    else:
        (
            pm,
            piano_roll,
            sixteenth_time,
            beat_time,
            down_beat_time,
            beat_indices,
            down_beat_indices,
        ) = res

    win_size = 10

    d = {}

    for i in range(piano_roll.shape[1]):
        try:
            left = max(i - win_size, 0)
            right = min(i + win_size, piano_roll.shape[1])
            piano_roll_seg = piano_roll[:, left:right]
            sixteenth_time_seg = sixteenth_time[left:right]
            beat_indices_seg = [
                i - left for i in beat_indices if left <= i <= right
            ]
            beat_indices_indices = [
                i for i, j in enumerate(beat_indices) if left <= j <= right
            ]
            beat_time_seg = beat_time[
                beat_indices_indices[0]:beat_indices_indices[-1] + 1
            ]
            down_beat_indices_seg = [
                i - left for i in down_beat_indices if left <= i <= right
            ]
            down_beat_indices_indices = [
                i for i, j in enumerate(down_beat_indices) if left <= j <= right
            ]
            down_beat_time_seg = down_beat_time[
                down_beat_indices_indices[0]:down_beat_indices_indices[-1] + 1
            ]

            if args.key_name == '':
                key_name = all_key_names

                res = cal_tension(
                    file_name,
                    piano_roll_seg,
                    sixteenth_time_seg,
                    beat_time_seg,
                    beat_indices_seg,
                    down_beat_time_seg,
                    down_beat_indices_seg,
                    args.output_folder,
                    args.window_size,
                    key_name,
                )

            else:
                res = cal_tension(
                    file_name,
                    piano_roll_seg,
                    sixteenth_time_seg,
                    beat_time_seg,
                    beat_indices_seg,
                    down_beat_time_seg,
                    down_beat_indices_seg,
                    args.output_folder,
                    args.window_size,
                    [args.key_name],
                )

            key_name = res

            if args.key_name == '':
                d[i] = key_name
        except Exception:
            d[i] = 'A minor'

    return file_name, os.path.dirname(base_name), base_name, d


def key_seq_per_file(file_name, win_size=10, end_ratio=0.5) -> List[str]:
    """Extract sequence of keys from midi file.

    Args:
        file_name (str) : path to midi file
        win_size (int) : size of window on piano roll to calculate key on
        end_ratio(float) : the place to find the first key of the song,
            0.5 means the first key is calculated by the first half of the song

    Returns:
        List[str]: list of keys for each window

    """
    track_num = 0  # default 0 means use all tracks
    try:
        res = extract_notes(file_name, track_num)
    except Exception:
        res = None

    if res is None:
        return None
    else:
        piano_roll = res[1]

    len_roll = piano_roll.shape[1]

    sentiments_list = []

    for i in range(len_roll):
        try:
            left = max(i - win_size, 0)
            right = min(i + win_size, len_roll)
            piano_roll_seg = piano_roll[:, left:right]

            key_names = all_key_names

            key_name, _, _ = cal_key(piano_roll_seg, key_names, end_ratio=end_ratio)

            sentiments_list.append(key_name)
        except Exception:
            sentiments_list.append('A minor')

    return sentiments_list


if __name__ == '__main__':
    args = get_args()

    args.output_folder = os.path.abspath(args.output_folder)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    logger = logging.getLogger(__name__)

    logger.handlers = []
    logfile = '{0}/tension_calculate.log'.format(args.output_folder)
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=logfile,
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(
        '%(asctime)s : %(levelname)s : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    console.setFormatter(formatter)
    logger.addHandler(console)

    coloredlogs.install(level='INFO', logger=logger, isatty=True)

    low_bound = 15
    upp_bound = 0.2
    if math.sqrt(2 / low_bound) <= args.vertical_step <= math.sqrt(upp_bound):
        vertical_step = args.vertical_step
    else:
        logger.info('invalid vertical step, use 0.4 instead')
        vertical_step = 0.4

    output_json_name = os.path.join(args.output_folder, 'files_result.json')

    files_result = {}

    if args.file_name:
        all_names = [args.file_name]
        args.input_folder = os.path.dirname(args.file_name)
    else:
        all_names = walk(args.input_folder)

    tasks = [(file_name, args) for file_name in all_names]
    print(len(tasks), len(all_names))

    num_processes = args.pool_num
    with Pool(num_processes) as p:
        res = list(tqdm.tqdm(p.imap(keys_per_token, tasks), total=len(all_names)))

    for r in res:

        file_name, new_output_folder, base_name, d = r

        if d is not None:
            files_result['{0}/{1}'.format(new_output_folder, base_name)] = []
            files_result[
                '{0}/{1}'.format(new_output_folder, base_name)
            ].append(d)
        else:
            logger.info(
                'cannot find key of song {0}, skip this file'.format(file_name),
            )

    logger.info(len(files_result))
    with open(os.path.join(args.output_folder, 'files_result.json'), 'w') as fp:
        json.dump(files_result, fp)
