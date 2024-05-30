import numpy as np


def add_piece_to_dataset_roots(piece, X=None, y=None):
    notes_dur_seq = piece[0]
    piece_len = piece[1]
    
    if y is not None:
        harmony_num = piece[2]
        if harmony_num == -1:
            return
          
    if X is None:
        X = []
    for pitch in range(12):
        notes_norm_lens = np.zeros(12)
        for note, dur in notes_dur_seq:
            shift = (note - pitch + 240) % 12
            notes_norm_lens[shift] = max(dur / piece_len, notes_norm_lens[shift])

        if y is not None:
            if pitch == harmony_num:
                y.append(1)
            else:
                y.append(0)

        features = np.zeros(13) # piece_len + 12 features
        features[0] = piece_len
        features[1:] = notes_norm_lens
        X.append(features)
    return X


def add_piece_to_dataset_modes(piece, X=None, y=None):
    notes_dur_seq = piece[0]
    piece_len = piece[1]
    root = piece[2]
    if y is not None:
        mode = piece[3]
        
    notes_norm_lens = np.zeros(12)

    for note, dur in notes_dur_seq:
        note_shifted = (note - root + 240) % 12 # относительно правильного корня
        notes_norm_lens[note_shifted] = max(dur / piece_len, notes_norm_lens[note_shifted])

    if y is not None:
        y.append(mode)
        
    features = np.zeros(13) # piece_len + 12 features
    features[0] = piece_len
    features[1:] = notes_norm_lens
    if X is None:
        X = []
    X.append(features)
    return X
