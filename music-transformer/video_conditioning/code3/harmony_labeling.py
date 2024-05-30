import argparse
import json
import logging
import os
import sys
from multiprocessing import Pool, context
from typing import List

import coloredlogs
import tqdm
from music21 import chord, converter

sys.modules['__main__'].__file__ = 'ipython'

all_tones_names = [
    'C',
    'G',
    'D',
    'A',
    'E',
    'B',
    'F',
    'C#',
    'G#',
    'D#',
    'A#',
    'E#',
    'B#',
    'F#',
    'Cb',
    'Gb',
    'Db',
    'Ab',
    'Eb',
    'Bb',
    'Fb',
]


def extract_pitch(chord_common_name):
    """Extract pitch from music21 format of chord.pitchedCommonName.

    Args:
        chord_common_name (str): obtained from chord.pitchedCommonName()

    Returns:
        str: chord name from all_tones_names list

    """
    try:
        if chord_common_name.split()[-1] in all_tones_names:
            pitch_name = chord_common_name.split()[-1]
        elif chord_common_name.split('-')[0] in all_tones_names:
            pitch_name = chord_common_name.split('-')[0]
        else:
            return None
        return add_chord_spec(chord_common_name, pitch_name)
    except Exception:
        return None


def add_chord_spec(chord_common_name, pitch_name):
    """Add specification to extracted pitch_name.

    Args:
        chord_common_name (str): obtained from chord.pitchedCommonName()
        pitch_name(str): extracted with extract_pitch function

    Returns:
        str: chord name from all_tones_names list with specification

    """
    if 'minor' in chord_common_name.lower():
        return '{0}m'.format(pitch_name)
    if 'sixth' in chord_common_name.lower():
        return '{0}6'.format(pitch_name)
    if 'seventh' in chord_common_name.lower():
        return '{0}7'.format(pitch_name)


def harmony_labeling_tokens(task):
    """Parse midi file and extract harmonies into dict.

    Args:
        task (tuple): (file_name, args)

    Returns:
        string, string, string, dict (int : str)

    """
    file_name, _ = task
    base_name = os.path.basename(file_name)
    try:
        midi = converter.parse(file_name)  # parse midi
    except Exception:
        return file_name, os.path.dirname(base_name), base_name, None
    midi_flat = midi.flat  # flatten midi
    harmony_dict = harmony_dict_from_midi(midi_flat)
    return file_name, os.path.dirname(base_name), base_name, harmony_dict


def harmony_dict_from_midi(midi_flat):
    """Write harmonies to dict.

    Args:
        midi_flat (music21.stream.Score): flatten Stream from midi file

    Returns:
        dict (int : str): dict of number of token : harmony name

    """
    harmony_dict = {}
    token_ind = 0
    for el in midi_flat:
        if isinstance(el, chord.Chord):
            if extract_pitch(el.pitchedCommonName) is not None:
                harmony_dict[token_ind] = extract_pitch(el.pitchedCommonName)
                token_ind += 1
    return harmony_dict


def get_args(default='.') -> argparse.Namespace:
    """Get arguments.

    Args:
        default (str) : default path for input_folder and output_folder

    Returns:
        argparse.Namespace : parsed args
    """
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
        help='number of processes for harmony labeling',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose log output',
    )
    return parser.parse_args()


def walk(folder_name: str) -> List[str]:
    """Walks through files in folder.

    Args:
        folder_name (str) : name of the folder

    Returns:
        list : sorted list of file names in the folder
    """
    files = []
    for path, _, all_files in os.walk(folder_name):
        for file_name in sorted(all_files):
            endname = file_name.split('.')[-1].lower()
            if 'mid' in endname:
                files.append(os.path.join(path, file_name))
    return files


if __name__ == '__main__':
    args = get_args()
    args.output_folder = os.path.abspath(args.output_folder)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    logger = logging.getLogger(__name__)

    logger.handlers = []
    logfile = '{0}/tokenize.log'.format(args.output_folder)
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

    output_json_name = os.path.join(args.output_folder, 'files_result.json')

    files_result = {}

    if args.file_name:
        all_names = [args.file_name]
        args.input_folder = os.path.dirname(args.file_name)
    else:
        all_names = walk(args.input_folder)

    tasks = [(file_name, args) for file_name in all_names]
    print(len(tasks), len(all_names))

    res = []

    num_processes = args.pool_num
    max_time = 1200
    pbar = tqdm.tqdm(total=len(tasks))
    while tasks:
        with Pool(num_processes) as pool:
            futures_res = pool.imap(harmony_labeling_tokens, tasks.copy())
            while tasks:
                task = tasks.pop(0)
                pbar.update(1)
                try:
                    future_res = futures_res.next(timeout=max_time)
                    res.append(future_res)
                except context.TimeoutError:
                    logger.info(
                        'stuck on file {0}, timeout err, skip'.format(task[0]),
                    )
                    break
    pbar.close()

    for rs in res:
        file_name, new_output_foler, base_name, harmony_dict = rs

        if harmony_dict is not None:
            files_result['{0}/{1}'.format(new_output_foler, base_name)] = []
            files_result[
                '{0}/{1}'.format(new_output_foler, base_name)
            ].append(harmony_dict)

        else:
            logger.info(
                'cannot parse song {0}, skip this file'.format(file_name),
            )

    logger.info(len(files_result))
    with open(os.path.join(args.output_folder, 'files_result.json'), 'w') as fp:
        json.dump(files_result, fp)
