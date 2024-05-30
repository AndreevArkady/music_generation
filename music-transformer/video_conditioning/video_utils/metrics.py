from music21 import converter, midi, stream, note, chord
import music21

def open_midi(midi_path, remove_drums):
    mf = music21.midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if remove_drums:
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]
    return music21.midi.translate.midiFileToStream(mf)

def get_key_midi_moments(
        path_midi,
        method: str = 'velocity',  # ('velocity' or 'density')
        key_moments_count: int = 10,
        return_all_scores=False,
        duration=None
):
    """
    Extracts key moments from the MIDI stream based on the specified method ('velocity' or 'density').

    Parameters:
        path_midi (str): Path to the MIDI file.
        method (str): The method to use for extracting key moments ('velocity' or 'density').
        key_moments_count (int): The number of key moments to extract.
        return_all_scores (bool): Whether to return all measure scores or not.
        duration (float): Duration in seconds to consider for extracting key moments.

    Returns:
        list: A list of measures identified as key moments.
    """
    key_moments_count = round(key_moments_count)
    base_midi = open_midi(path_midi, False)

    measures = base_midi.parts[0].getElementsByClass(music21.stream.Measure)
    measure_scores = []

    if duration is not None:
        # Convert duration from seconds to music21 quarter lengths
        tempo = base_midi.metronomeMarkBoundaries()[0][-1].number
        quarter_length_per_second = tempo / 60
        max_quarter_length = duration * quarter_length_per_second

    if method == 'velocity':
        for measure in measures:
            if duration is not None and measure.offset > max_quarter_length:
                break
            if len(measure.flat.notes) > 0:
                measure_velocity = sum(note.volume.velocity for note in measure.flat.notes if note.volume.velocity is not None) / len(measure.flat.notes)
            else:
                measure_velocity = 0
            measure_scores.append((measure.offset, measure_velocity))

    elif method == 'density':
        for measure in measures:
            if duration is not None and measure.offset > max_quarter_length:
                break
            density = len(measure.flat.notes)
            measure_scores.append((measure.offset, density))


    # Sort measures by score in descending order and select the top key_moments_count
    # measure_scores.sort(key=lambda x: x[1], reverse=True)
    measure_scores_final_s = sorted(measure_scores, key=lambda x: x[1], reverse=True)

    key_midi_moments = sorted([ms[0] for ms in measure_scores_final_s[:key_moments_count]])

    if return_all_scores:
        return key_midi_moments, measure_scores
    return key_midi_moments

#####################


# import mido

# def open_midi_2(midi_path, remove_drums=False):
#     mid = mido.MidiFile(midi_path)
#     if remove_drums:
#         for track in mid.tracks:
#             track.events = [msg for msg in track if not (msg.type == 'note_on' and msg.channel == 9)]
#     return mid

# def get_key_midi_moments_2(
#         path_midi,
#         method: str = 'velocity',  # ('velocity' or 'density')
#         key_moments_count: int = 10,
#         return_all_scores=False
# ):
#     key_moments_count = round(key_moments_count)
#     mid = open_midi_2(path_midi, False)

#     ticks_per_beat = mid.ticks_per_beat
#     measure_ticks = ticks_per_beat * 4  # Assuming 4/4 time signature

#     measure_scores = []
#     current_measure = []
#     current_measure_start = 0
#     for track in mid.tracks:
#         time = 0
#         for msg in track:
#             time += msg.time
#             if time >= current_measure_start + measure_ticks:
#                 if current_measure:
#                     measure_scores.append((current_measure_start, current_measure))
#                 current_measure = []
#                 current_measure_start += measure_ticks

#             if msg.type == 'note_on' and msg.velocity > 0:
#                 current_measure.append(msg)

#     if current_measure:
#         measure_scores.append((current_measure_start, current_measure))

#     measure_scores_final = []

#     if method == 'velocity':
#         for measure_start, measure in measure_scores:
#             if measure:
#                 measure_velocity = sum(msg.velocity for msg in measure) / len(measure)
#             else:
#                 measure_velocity = 0
#             measure_scores_final.append((measure_start, measure_velocity))

#     elif method == 'density':
#         for measure_start, measure in measure_scores:
#             density = len(measure)
#             measure_scores_final.append((measure_start, density))

#     # Sort measures by score in descending order and select the top key_moments_count
#     measure_scores_final.sort(key=lambda x: x[1], reverse=True)
#     key_midi_moments = sorted([ms[0] for ms in measure_scores_final[:key_moments_count]])

#     if return_all_scores:
#         return key_midi_moments, measure_scores_final
#     return key_midi_moments


import mido

def open_midi_2(midi_path, remove_drums=False):
    mid = mido.MidiFile(midi_path)
    if remove_drums:
        for track in mid.tracks:
            track.events = [msg for msg in track if not (msg.type == 'note_on' and msg.channel == 9)]
    return mid

def get_key_midi_moments_2(
        path_midi,
        method: str = 'velocity',  # ('velocity' or 'density')
        key_moments_count: int = 10,
        return_all_scores=False,
        duration=None
):
    key_moments_count = round(key_moments_count)
    mid = open_midi_2(path_midi, False)

    ticks_per_beat = mid.ticks_per_beat
    measure_ticks = ticks_per_beat * 4  # Assuming 4/4 time signature

    measure_scores = []
    current_measure = []
    current_measure_start = 0
    total_time = 0  # Time in ticks

    for track in mid.tracks:
        time = 0
        for msg in track:
            time += msg.time
            total_time += msg.time
            
            if duration is not None:
                # Convert duration from seconds to ticks
                seconds_per_beat = 60 / 120  # Assuming 120 BPM
                ticks_per_second = ticks_per_beat / seconds_per_beat
                max_ticks = duration * ticks_per_second
                if total_time > max_ticks:
                    break

            if time >= current_measure_start + measure_ticks:
                if current_measure:
                    measure_scores.append((current_measure_start, current_measure))
                current_measure = []
                current_measure_start += measure_ticks

            if msg.type == 'note_on' and msg.velocity > 0:
                current_measure.append(msg)

    if current_measure:
        measure_scores.append((current_measure_start, current_measure))

    measure_scores_final = []

    if method == 'velocity':
        for measure_start, measure in measure_scores:
            if measure:
                measure_velocity = sum(msg.velocity for msg in measure) / len(measure)
            else:
                measure_velocity = 0
            measure_scores_final.append((measure_start, measure_velocity))

    elif method == 'density':
        for measure_start, measure in measure_scores:
            density = len(measure)
            measure_scores_final.append((measure_start, density))

    # Sort measures by score in descending order and select the top key_moments_count
    # measure_scores_final.sort(key=lambda x: x[1], reverse=True)
    measure_scores_final_s = sorted(measure_scores_final, key=lambda x: x[1], reverse=True)
    key_midi_moments = sorted([ms[0] for ms in measure_scores_final_s[:key_moments_count]])

    if return_all_scores:
        return key_midi_moments, measure_scores_final
    return key_midi_moments

# Example usage
# path_midi = 'path/to/your/midi/file.mid'
# key_moments = get_key_midi_moments_2(path_midi, method='velocity', key_moments_count=10, duration=30)
# print(key_moments)

##############


def concordance_score(
        video_moments, 
        music_moments, 
        alpha=1.0, 
        offset=0.2, 
        eps=0.001, 
        version='MAX', 
        symmetric=False
    ):
    VandM = 0
    VnotM = 0
    MnotV = 0

    i, j = 0, 0
    video_moment_matches = [0] * len(video_moments)
    while i < len(video_moments) and j < len(music_moments):
        if video_moments[i] - offset <= music_moments[j] <= video_moments[i] + eps:
            j += 1
            VandM += 1
            video_moment_matches[i] = 1
            if symmetric:
                i += 1
        elif video_moments[i] - offset > music_moments[j]:
            MnotV += 1
            j += 1
        else:
            if video_moment_matches[i] != 1:
                VnotM += 1
            i += 1

    MnotV += max(0, len(music_moments)-j-1)
    VnotM += max(0, len(video_moments)-i-1)
    # print(VnotM, MnotV, VandM)
    if version == 'SUM':
        res = VandM / (alpha * VnotM + (1/alpha) * MnotV + VandM)
    elif version == 'MAX':
        res = VandM / (max(alpha * VnotM, (1/alpha) * MnotV) + VandM)
    else:
        return 'not correct version'
    return res