import numpy as np
import pandas as pd

from dataset import add_piece_to_dataset_roots, add_piece_to_dataset_modes

# size of values range of each event type
RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_TIME_SHIFT = 100
RANGE_VEL = 32

MAX_TOKEN_SEQUENCE = 4000

# each event we want to convert to int
# So, different events has different values range
# 'note_on': [0; RANGE_NOTE_ON)
# 'note_off': [RANGE_NOTE_ON; RANGE_NOTE_ON + RANGE_NOTE_OFF)
# 'time_shift': [RANGE_NOTE_ON + RANGE_NOTE_OFF; RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT)
# 'velocity': [RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT; 'end_of_scope')

win_min = 50
win_max = 300

def parse_token(token):
    if token < RANGE_NOTE_ON:
        return 'note_on', token
    if token < RANGE_NOTE_ON + RANGE_NOTE_OFF:
        return 'note_off', token - RANGE_NOTE_ON
    if token < RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT:
        return 'time_shift', token - (RANGE_NOTE_ON + RANGE_NOTE_OFF)
    return 'velocity', token - (RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT)

def time_shifts_array(tokens):
    time_shifts = [] # храним (индекс токена в исходном списке, текущее время) (сумма предыдущих time_shifts)
    curr_time = 0
    for i, token in enumerate(tokens):
        name, value = parse_token(token)
        if name == 'time_shift':
            curr_time += value
            time_shifts.append((i, curr_time))
    return time_shifts

# last time shift == sum of all timeshifts
# at token <timeshift> time **after** this event occurs


def make_piece_from_tokens(time_shifts, tokens, j, i, root_note=None, mode=None):
    shifted_notes = set()
    notes_time_segments = dict() # note : [begin_time, end_time], used as weights in voting
    curr_time = 0 # sum of time_shifts
    
    if j == -1:
        l = 0
        l_time = 0
    else:
        l = time_shifts[j][0]
        l_time = time_shifts[j][1]
    if i == len(time_shifts) - 1:
        r = -1
        r_time = time_shifts[-1][1]
    else:
        r = time_shifts[i][0]
        r_time = time_shifts[i][1]

    piece_len = r_time - l_time
    for token in tokens[l+1:r+1]:
        name, value = parse_token(token)
        note = value

        if name == 'note_on':
            shifted_notes.add(note)
            notes_time_segments[note] = [curr_time, None]
        elif name == 'note_off':
            if note not in shifted_notes: # значит нота играет где-то в срезе, но началась в предыдущем, поэтому добавим сюда
                shifted_notes.add(note)
                notes_time_segments[note] = [0, curr_time]
            else:
                notes_time_segments[note][1] =  curr_time
        elif name == 'time_shift':
            curr_time += value

    notes_dur_list = []
    for key, value in notes_time_segments.items():
        if value[1] is None:
            value[1] = curr_time
        notes_dur_list.append((key, (value[1] - value[0]) * 10))
 
    if root_note is not None and mode is not None:

        return (notes_dur_list, piece_len * 10, root_note, mode)
    if root_note is not None:
        return (notes_dur_list, piece_len * 10, root_note)
    else:
        return (notes_dur_list, piece_len * 10)


def root_probs(model_roots, piece):
    piece_12_shifts = add_piece_to_dataset_roots(piece)
    y_proba = model_roots.predict_proba(np.array(piece_12_shifts))
    probs = y_proba[:,1]
    return probs


def root_mode_probs(model_mode, piece, root_probs_vec):
    probs = np.zeros(24)
    pieces_with_root = []
    for root in range(12):
        piece_with_root = (piece[0], piece[1], root)
        piece_with_root_mode = add_piece_to_dataset_modes(piece_with_root, pieces_with_root)
        
    major_prob = model_mode.predict_proba(np.array(piece_with_root_mode))[:, 0]
    minor_prob = model_mode.predict_proba(np.array(piece_with_root_mode))[:, 1]
    
    for root in range(12):
        probs[root] = major_prob[root] * root_probs_vec[root]
        probs[root + 12] = minor_prob[root] * root_probs_vec[root]
    return probs


def prob_root_notes(model_roots, model_mode, time_shifts, tokens, i, j):
    piece = make_piece_from_tokens(time_shifts, tokens, i, j)
    root_probs_vec = root_probs(model_roots, piece)
    probs = root_mode_probs(model_mode, piece, root_probs_vec)
    return np.log(np.max(probs)), np.argwhere(probs == np.max(probs)).flatten(), probs


def fill_dp(model_roots, model_mode, tokens, time_shifts):
    n = len(time_shifts)
    dp = np.array([None] * (n + 1) ** 2).reshape(n + 1, n + 1)

    for i, (_, ts_time) in enumerate(time_shifts):
        if ts_time < win_min:
            continue
        if ts_time >= win_max:
            break
        dp[0][i + 1] = prob_root_notes(model_roots, model_mode, time_shifts, tokens, -1, i)

    for i, (_, ts_time) in enumerate(time_shifts):
        next_token_i = i
        next_ts_time = time_shifts[next_token_i][1]
        while next_ts_time - ts_time <= win_max:
            if next_ts_time - ts_time < win_min:
                next_token_i += 1
                if next_token_i >= len(time_shifts):
                    break
                next_ts_time = time_shifts[next_token_i][1]
                continue
            else:
                dp[i + 1][next_token_i + 1] = prob_root_notes(model_roots, model_mode, time_shifts, tokens, i, next_token_i)
                next_token_i += 1
                if next_token_i >= len(time_shifts):
                    break
                next_ts_time = time_shifts[next_token_i][1]

    return dp


def fill_pref(time_shifts, dp):
    n = len(time_shifts)
    pref_0 = [-100] * (n + 1)  # sum log probs
    pref_1 = [-1] * (n + 1)  # ind of prev best harmony piece
    for j, (_, ts_time) in enumerate(time_shifts):
        if ts_time >= win_min and dp[0][j + 1]:
            pref_0[j + 1] = dp[0][j + 1][0]
            pref_1[j + 1] = 0
            break

    for j, (_, ts_time) in enumerate(time_shifts):
        prev_token_i = j - 1
        prev_ts_time = time_shifts[prev_token_i][1]

        while ts_time - prev_ts_time <= win_max and prev_token_i >= 0:
            if dp[prev_token_i + 1][j + 1] and pref_0[prev_token_i + 1] + dp[prev_token_i + 1][j + 1][0] > pref_0[j + 1]:
                pref_0[j + 1] = pref_0[prev_token_i + 1] + dp[prev_token_i + 1][j + 1][0]
                pref_1[j + 1] = prev_token_i + 1
            prev_token_i -= 1
            prev_ts_time = time_shifts[prev_token_i][1]
    return pref_0, pref_1


def optimal_pieces_inds_array(pref_1):
    optimal_pieces_inds = []
    ind = len(pref_1) - 1
    while pref_1[ind] == -1:
        ind -= 1
    while ind >= 0:
        optimal_pieces_inds.append(ind)
        ind = pref_1[ind]

    return optimal_pieces_inds[::-1]


def harmony_seq(optimal_pieces_inds, time_shifts, dp):
    harmonies = []
    times = []
    curr_step = 0
    l = 0
    r = optimal_pieces_inds[curr_step + 1]
    res = []


    while curr_step < len(optimal_pieces_inds) - 1:
        harmony = dp[l][r][1][0]
        harmonies.append(harmony)
        time = (time_shifts[l][0], time_shifts[r][0])
        times.append(time)
        res.append([int(time[0]), int(time[1]), int(harmony)])

        l = optimal_pieces_inds[curr_step]
        r = optimal_pieces_inds[curr_step + 1]
        curr_step += 1
    return res


def harmony_seq_pipeline(file, model_roots, model_mode):
    tokens_df = pd.read_csv(file)
    tokens = tokens_df['tokens'].tolist()

    # use code below to convert .nc file from Kostka-Payne dataset to MusicTransformer tokens

    # from kp_corpus_process import convert_nc_to_tokens
    # tokens = convert_nc_to_tokens(file)
    #
    # import os
    # name = os.path.splitext(os.path.basename(file))[0]
    #
    # pd.DataFrame({'tokens': tokens}).to_csv(f'/home/azatvaleev/harmony_labeling/kp-labeling/{name}.tsv', sep='\t', index=False)

    time_shifts = time_shifts_array(tokens)
    dp = fill_dp(model_roots, model_mode, tokens, time_shifts)

    pref_0, pref_1 = fill_pref(time_shifts, dp)
    optimal_pieces_inds = optimal_pieces_inds_array(pref_1)

    res = harmony_seq(optimal_pieces_inds, time_shifts, dp)
    return res
