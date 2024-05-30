from typing import List

import numpy as np

from dataset import add_piece_to_dataset_roots, add_piece_to_dataset_modes


def str_time_to_int(timestamp):
    return int(timestamp.replace('.', ''))


def prepare_dataset_roots():
    with open('/data/datasets/kp-corpus-files/kp-chord-list', 'r') as file:
        lines = file.read().split('\n')

    song_seq_from_file_dict = dict()

    curr_song_name = ''

    for line in lines:
        if line[0] == '%':
            curr_song_name = line.replace('% ', '').strip('.')
            song_seq_from_file_dict[curr_song_name] = []
        elif line[0] != '-':
            elems = line.strip().replace('  ', ' ').split(' ')
            song_seq_from_file_dict[curr_song_name].append((str_time_to_int(elems[0]), str_time_to_int(elems[1]), int(elems[-1])))

    file_notes_dict = dict() # from Melisma notes analyzer

    for filename in song_seq_from_file_dict.keys():
        file_notes_dict[filename] = []
        with open(f'/data/datasets/kp-corpus-files/kp-nbck/{filename}.nc', 'r') as file:
            lines = file.read().split('\n')
            for line in lines:
                if line.split(' ')[0] == 'Note':
                    file_notes_dict[filename].append((int(line.split(' ')[1]), int(line.split(' ')[2]), int(line.split(' ')[3])))

    pieces_with_chords = [] # [..., (note, dur), ... ], piece_len, chord_label
    for filename, chords_seq in song_seq_from_file_dict.items():
        notes_seq = file_notes_dict[filename]
        for start_time, end_time, chord in chords_seq:
            curr_notes = []
            for note in notes_seq:
                if note[0] >= start_time and note[1] <= end_time:
                    curr_notes.append((note[2], note[1] - note[0]))
            pieces_with_chords.append((curr_notes, end_time - start_time, chord))

    X = []
    y = []
    for piece in pieces_with_chords:
        add_piece_to_dataset_roots(piece, X, y)

    X = np.array(X)
    y = np.array(y)
    return X, y


def prepare_dataset_mode():
    with open('/data/datasets/kp-corpus-files/kp-chord-list-modes', 'r') as file:
        lines = file.read().split('\n')

    song_seq_roots_modes_dict = dict() # read file with modes and chord roots

    curr_song_name = ''

    for line in lines:
        if line[0] == '%':
            curr_song_name = line.replace('% ', '').strip('.')
            song_seq_roots_modes_dict[curr_song_name] = []
        elif line[0] != '-':
            elems = line.strip().replace('  ', ' ').split(' ')
            song_seq_roots_modes_dict[curr_song_name].append((str_time_to_int(elems[0]), str_time_to_int(elems[1]), int(elems[-2]),  int(elems[-1])))

    file_notes_dict = dict() # from Melisma notes analyzer

    for filename in song_seq_roots_modes_dict.keys():
        file_notes_dict[filename] = []
        with open(f'/data/datasets/kp-corpus-files/kp-nbck/{filename}.nc', 'r') as file:
            lines = file.read().split('\n')
            for line in lines:
                if line.split(' ')[0] == 'Note':
                    file_notes_dict[filename].append((int(line.split(' ')[1]), int(line.split(' ')[2]), int(line.split(' ')[3])))

    pieces_with_roots_modes= [] # [..., (note, dur), ... ], piece_len, root (one of 12), mode (1 -- major, 2 -- minor)
    for filename, modes_seq in song_seq_roots_modes_dict.items():
        notes_seq = file_notes_dict[filename]
        for start_time, end_time, chord_root, mode in modes_seq:
            curr_notes = []
            for note in notes_seq:
                if note[0] >= start_time and note[1] <= end_time:
                    curr_notes.append((note[2], note[1] - note[0]))
            pieces_with_roots_modes.append((curr_notes, end_time - start_time, chord_root, mode))

    X = []
    y = []

    for piece in pieces_with_roots_modes:
        add_piece_to_dataset_modes(piece, X, y)

    X = np.array(X)
    y = np.array(y)
    return X, y


def add_note_on(tokens: List[int], note_pitch: int, filename: str):
    if note_pitch >= 128:
        raise AssertionError(f'pitch must be less 128, but got pitch = {note_pitch} in {filename}')
    tokens.append(note_pitch)


def add_note_off(tokens: List[int], note_pitch: int, filename: str):
    if note_pitch >= 128:
        raise AssertionError(f'pitch must be less 128, but got pitch = {note_pitch} in {filename}')
    tokens.append(note_pitch + 128)


def add_timeshift(tokens: List[int], timeshift: int):
    while timeshift - 1000 > 0:
        tokens.append(255 + 100)
        timeshift -= 1000

    tokens.append(255 + (timeshift // 10))


def convert_nc_to_tokens(filename):
    with open(filename, 'r') as file:
        lines = file.read().split('\n')

    events_time = dict()

    for line in lines:
        if line.split(' ')[0] == 'Note':
            note_on_time = (int(line.split(' ')[1]) // 10) * 10
            note_off_time = (int(line.split(' ')[2]) // 10) * 10
            note_pitch = int(line.split(' ')[3])

            if note_on_time not in events_time:
                events_time[note_on_time] = [[], []]
            events_time[note_on_time][0].append(note_pitch)

            if note_off_time not in events_time:
                events_time[note_off_time] = [[], []]
            events_time[note_off_time][1].append(note_pitch)

    events_time = dict(sorted(events_time.items()))
    tokens = []

    prev_event_time = 0
    for event_time, (note_on_events, note_off_events) in events_time.items():
        if event_time > prev_event_time:
            add_timeshift(tokens, event_time - prev_event_time)

        for note_off_pitch in note_off_events:
            add_note_off(tokens, note_off_pitch, filename)

        for note_on_pitch in note_on_events:
            add_note_on(tokens, note_on_pitch, filename)

        prev_event_time = event_time

    return tokens
