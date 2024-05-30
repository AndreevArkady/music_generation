import argparse
import os
import types
import warnings

import numpy as np
import pretty_midi
from mido.midifiles import meta

# size of values range of each event type  https://arxiv.org/pdf/1808.03715v1
RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_TIME_SHIFT = 100
RANGE_VEL = 32

# each event we want to convert to int
# So, different events has different values range
# 'note_on': [0; RANGE_NOTE_ON);
# 'note_off': [RANGE_NOTE_ON; RANGE_NOTE_ON + RANGE_NOTE_OFF);
# 'time_shift': [RANGE_NOTE_ON + RANGE_NOTE_OFF; RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT);  # noqa: E501
# 'velocity': [RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT; 'end_of_scope');  # noqa: E501

START_IDX = types.MappingProxyType({
    'note_on': 0,
    'note_off': RANGE_NOTE_ON,  # 128
    'time_shift': RANGE_NOTE_ON + RANGE_NOTE_OFF,  # 256
    'velocity': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT,  # 356
    'end_of_scope': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT + RANGE_VEL  # 388 noqa: E501
})


class SustainDownManager(object):
    """Processes Pedal on/off events.

    Attributes:
        start: pedal start
        end: pedal end
        managed_notes: notes in sustain
    """

    def __init__(self, start: int = 0, end: int = 0):
        """Initialize SustainDownManager.

        Args:
            start: pedal start
            end: pedal end
        """
        self.start: int = start
        self.end: int = end
        self.managed_notes = []

    def add_managed_note(self, note: pretty_midi.Note):
        """Add notes during Pedal on.

        Args:
            note: to add
        """
        self.managed_notes.append(note)

    def transposition_notes(self):
        """Transform notes with sustain.

        Change note's end to start of the next same note if exists,
            else to the Pedal off time (self.end)

        Returns:
            processed pretty_midi.Note notes according to Pedal On and Pedal Off
        """
        note_last_start = {}  # key: pitch, value: note.start
        for note in reversed(self.managed_notes):
            note.end = note_last_start.get(note.pitch, max(self.end, note.end))
            note_last_start[note.pitch] = note.start
        return self.managed_notes


class SplitNote(object):
    """Stores note as multiple parameters.

    Used to store 'note_on' and 'note_off' event separately

    Attributes:
        note_type: note type ('note_on', 'note_off')
        time: time from start in seconds
        value: pitch value
        velocity: velocity value
    """

    def __init__(self, note_type: str, time: float, value: int, velocity: int):
        """Initialize SplitNote.

        Args:
            note_type: note type ('note_on', 'note_off')
            time: time from start in seconds
            value: pitch value
            velocity: velocity value
        """
        self.note_type: str = note_type
        self.time: float = time
        self.value: int = value
        self.velocity: int = velocity  # -1 when 'note_off'

    def __repr__(self):
        """Get unambiguous representation of class object.

        Returns:
            representaion
        """
        return f'<[SNote] time: {self.time} type: {self.note_type}, value: {self.value}, velocity: {self.velocity}>'  # noqa: E501


class Event(object):
    """Events for midi events.

    Each event is 1 of 4 types: note_on, note_off, velocity, time_shift

    Attributes:
        event_type: event type ('note_on', 'note_off', 'velocity', 'time_shift')
        value: event value (pitch / velocity / duration)
    """

    def __init__(self, event_type: str, value: int):
        """Initialize Event.

        Args:
            event_type: type of event
            value: value of event
        """
        self.event_type = event_type
        self.value = value

    def __repr__(self):
        """Get unambiguous representation of class object.

        Returns:
            representaion
        """
        return f'<Event type: {self.event_type}, value: {self.value}>'

    def __int__(self):
        """Converts event to token (single integer).

        Returns:
            int representation
        """
        return START_IDX[self.event_type] + self.value


def event_from_int(value):
    """Creates Event object from int value.

    Args:
        value: token int

    Returns:
        Event object created from token

    Raises:
        ValueError: incorrect token value
    """
    if value < 0:
        raise ValueError(
            f'Cannot intialize Event object from negative value: {value}'
        )
    if value >= START_IDX['end_of_scope']:
        raise ValueError(
            f'Cannot intialize Event object from out of scope value: {value}'
        )
    res_event_type = ''
    res_segment_start = 0
    segments = sorted(START_IDX.items(), key=lambda x: x[1])
    for segment_event_type, segment_start in segments:
        if value < segment_start:
            break
        res_event_type = segment_event_type
        res_segment_start = segment_start
    return Event(res_event_type, value - res_segment_start)


def split_notes(notes):
    """Splits midi notes into Event objects by start and end.

    Args:
        notes: list of pretty_midi.Note

    Returns:
        list of SplitNote objects
    """
    splitted_notes = []
    notes.sort(key=lambda x: x.start)
    for note in notes:
        splitted_notes.append(SplitNote(
            'note_on',
            note.start,
            note.pitch,
            note.velocity
        ))
        splitted_notes.append(SplitNote(
            'note_off',
            note.end,
            note.pitch,
            -1
        ))
    return splitted_notes


def merge_note(snote_sequence):  # noqa: WPS231
    """Restores MIDI notes from SplitNote list.

    For each SplitNote object puts note_on into dict and tries to find
    corresponding note_on for each note_off

    Args:
        snote_sequence: list of SplitNote objects

    Returns:
        list of pretty_midi.Note notes
    """
    note_on_dict = {}
    result_array = []

    for snote in snote_sequence:
        if snote.note_type == 'note_on':
            note_on_dict[snote.value] = snote
        elif snote.note_type == 'note_off':
            if snote.value in note_on_dict:
                on = note_on_dict[snote.value]  # noqa: WPS529
                off = snote
                if off.time - on.time == 0:
                    continue
                result = pretty_midi.Note(  # noqa: WPS110
                    on.velocity,
                    snote.value,
                    on.time,
                    off.time
                )
                result_array.append(result)
            else:
                warnings.warn(
                    f"Unable to find note\'s start with pitch: {snote.value}"
                )
    return result_array


def events_to_snote(event_sequence):
    """Translates sequence of events of 4 types to SplitNotes objects sequence.

    Args:
        event_sequence: list of Event objects

    Returns:
        list of SplitNote objects
    """
    current_time = 0
    current_velocity = 0
    snote_sequence = []

    for event in event_sequence:
        if event.event_type == 'time_shift':
            current_time += (event.value + 1) / 100
        if event.event_type == 'velocity':
            current_velocity = event.value * 4
        else:
            snote = SplitNote(
                event.event_type,
                current_time,
                event.value,
                current_velocity
            )
            snote_sequence.append(snote)
    return snote_sequence


def make_time_shift_events(prev_time, post_time):
    """Creates range of Event 'time_shift' objects.

    Args:
        prev_time: begin time
        post_time: end time

    Returns:
        time_shift objects with step 100 (1 sec)
            from prev_time to post_time
    """
    time_interval = round((post_time - prev_time) * 100)
    results = []
    while time_interval >= RANGE_TIME_SHIFT:
        results.append(Event('time_shift', RANGE_TIME_SHIFT - 1))
        time_interval -= RANGE_TIME_SHIFT
    if time_interval == 0:
        return results

    results.append(Event('time_shift', time_interval - 1))
    return results


def control_preprocess(ctrl_changes):
    """Creates sequence of SustainDownManager for each control change.

    Args:
        ctrl_changes: List of pretty_midi.ControlChange objects

    Returns:
        list of SustainDownManager objects for each control change
    """
    sustains = []
    manager = None
    control_change = 64
    for ctrl in ctrl_changes:
        if ctrl.value >= control_change and manager is None:
            # Pedal On
            manager = SustainDownManager(ctrl.time)
        elif ctrl.value < control_change:
            if manager is not None:
                # Pedal Off
                manager.end = ctrl.time
                sustains.append(manager)
                manager = None
            # uncomment if you want to process the last pedal as the true one
            # elif len(sustains) > 0:  # noqa: E800
            #     sustains[-1].end = ctrl.time  # noqa: E800
    return sustains


def note_preprocess(sustains, notes):  # noqa: WPS231
    """Process notes according to sustains.

    Notes and sustains must be sorted by end

    Args:
        sustains: list of SustainDownManager objects
        notes:  list of pretty_midi.Note objects

    Returns:
        sequence
    """
    note_stream = []
    if sustains:  # if the midi file has sustain controls
        # go through each sustain, process every note in sustain time segment
        for sustain in sustains:
            for note_idx, note in enumerate(notes):  # go through notes
                # if note ended before sustain,
                # it is not affected, so just append
                if note.end <= sustain.start:
                    note_stream.append(note)
                elif note.start >= sustain.end:
                    # if notes started after sustain its end and next notes
                    # ends must be after sustain, so they won't be affected.
                    # Also, previous notes won't be affected by next sustains
                    # because they end after the
                    # node.end > note.start > sustain.end
                    notes = notes[note_idx:]
                    sustain.transposition_notes()
                    break
                else:
                    sustain.add_managed_note(note)

        for sustain in sustains:  # noqa: WPS519, WPS440
            note_stream += sustain.managed_notes

    for note in notes:  # noqa: WPS440
        note_stream.append(note)

    note_stream.sort(key=lambda x: x.start)
    return note_stream


def shrink_gaps(event_sequence):
    """Shrinks big gaps to 1 sec.

    Args:
        event_sequence: list of Event objects

    Returns:
        list Event objects
    """
    current_gap = 0
    return_sequence = []
    max_gap = 1.5

    for event in event_sequence:
        if event.event_type == 'time_shift':
            current_gap += (event.value + 1) / 100
        else:
            current_gap = 0
        if current_gap > max_gap:
            continue
        return_sequence.append(event)
    return return_sequence


def strip_pauses(tokens, copy=True):
    """Remove useless pauses before beginning and after ending.

    Args:
        tokens: array of tokenized composition
        copy: whether to return a copy of tokens

    Returns:
        array of stripped tokenized composition
    """
    min_time_value = START_IDX['time_shift']
    max_time_value = START_IDX['velocity'] - 1
    left = 0  # noqa: E741
    right = tokens.shape[0]
    while min_time_value <= tokens[left] <= max_time_value:
        left += 1  # noqa: E741
    while min_time_value <= tokens[right - 1] <= max_time_value:
        right -= 1
    if copy:
        return tokens[left:right].copy()

    return tokens[left:right]


def add_silence(tokens, n_sec):
    """Adds silence to the end of tokens.

    Args:
        tokens: tokens sequence
        n_sec: duration of silence in seconds

    Returns:
        tokens with silence at the end
    """
    n_time = int(n_sec * 100)
    n_time_tokens = int(np.ceil(n_sec))
    new_tokens = np.zeros(tokens.shape[0] + n_time_tokens, dtype=np.int32)
    new_tokens[:tokens.shape[0]] = tokens  # noqa: WPS362
    silence = np.zeros(n_time_tokens, dtype=np.int32)
    for i in range(n_time_tokens):
        silence[i] = START_IDX['time_shift'] + min(100, n_time)
        n_time -= min(100, n_time)
    new_tokens[tokens.shape[0]:] = silence  # noqa: WPS362
    return new_tokens


def encode_midi(file_path, drop_pauses=False, append_silence=0):  # noqa: WPS231
    """Translates MIDI audio to events (tokens).

    Args:
        file_path: MIDI file path
        drop_pauses: whether remove pauses before beginning and after ending
        append_silence: seconds of silence to add to the end

    Returns:
        pandas Dataframe with columns of tokens and other features
        or np.array[int]
    """
    events = []
    notes = []
    try:
        mid = pretty_midi.PrettyMIDI(midi_file=file_path)
    except (meta.KeySignatureError, EOFError, OSError):
        warnings.warn(
            f'PrettyMIDI was unable to encode file: {file_path}'
        )
        return None

    for inst in mid.instruments:
        inst_notes = inst.notes
        # ctrl.number is the number of sustain control.
        # If you want to know abour the number type of control, see
        # https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2
        # value of control change 64 equals sustain pedal
        control_change = 64
        ctrls = control_preprocess([
            ctrl
            for ctrl in inst.control_changes
            if ctrl.number == control_change
        ])
        notes += note_preprocess(ctrls, inst_notes)

    snotes = split_notes(notes)

    snotes.sort(key=lambda x: x.time)

    cur_time = 0
    cur_velocity = 0
    for snote in snotes:
        events += make_time_shift_events(
            prev_time=cur_time,
            post_time=snote.time
        )

        if snote.note_type != 'note_off':
            modified_velocity = snote.velocity // 4
            if cur_velocity != modified_velocity:
                events.append(Event('velocity', modified_velocity))
        events.append(Event(snote.note_type, snote.value))

        cur_time = snote.time
        cur_velocity = snote.velocity

    res = np.array([int(e) for e in events])
    if res.shape[0] == 0:
        warnings.warn(
            f'Unable to encode file: {file_path}'
        )
        return None
    if drop_pauses:
        res = strip_pauses(res, copy=False)
        if res.shape[0] == 0:
            warnings.warn(
                f'Unable to encode file: {file_path}'
            )
            return None

    if append_silence != 0:
        res = add_silence(res, append_silence)

    return res


def decode_midi(tokens, file_name=None, do_shrink_gaps=False):
    """Translates events (tokens) sequence to MIDI audio.

    Args:
        tokens: array of tokens
        file_name: export MIDI file path file_name
        do_shrink_gaps: bool whether to compress long pauses

    Returns:
        pretty_midi.PrettyMIDI object
    """
    event_sequence = [event_from_int(token) for token in tokens]
    if do_shrink_gaps:
        event_sequence = shrink_gaps(event_sequence)
    snote_seq = events_to_snote(event_sequence)
    note_seq = merge_note(snote_seq)
    note_seq.sort(key=lambda x: x.start)

    mid = pretty_midi.PrettyMIDI()
    # if you want to change the instrument, check
    # https://www.midi.org/specifications/item/gm-level-1-sound-set
    instrument = pretty_midi.Instrument(
        0,
        is_drum=False,
        name='Composed by GlinkaMusic'
    )
    instrument.notes = note_seq

    mid.instruments.append(instrument)
    if file_name is not None:
        mid.write(file_name)
    return mid


def parse_args():
    """Arguments parser.

    Returns:
        argspace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'midi_file_path',
        nargs=1,
        default='',
        type=str,
        help='Path to the MIDI file'
    )
    return parser.parse_args()


def main():
    """Main functoin."""
    args = parse_args()
    file_path = args.midi_file_path[0]
    if file_path == '':
        print('Input the .midi file path to apply encode-decode')
        return
    if not os.path.isfile(file_path):
        print('ERROR: Could not find file:', file_path)
        return
    directory = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    name, ext = os.path.splitext(file_name)
    if ext != '.midi':
        print('Not a .midi format')
        return
    encoded = encode_midi(file_path)
    decoded_file_path = f'{directory}/{name}_encoded-decoded{ext}'
    decode_midi(encoded['tokens'].to_numpy(), decoded_file_path)


if __name__ == '__main__':
    main()
