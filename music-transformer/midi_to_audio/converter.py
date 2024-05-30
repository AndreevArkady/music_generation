import os
from typing import List, Optional

from midi2audio import FluidSynth

from preprocessing.processor import decode_midi


def midi_to_audio(
    tokens: Optional[List] = None,
    midi_path: Optional[str] = None,
    file_name: str = '',
):
    """Converts midi to mp3.

    Args:
        tokens: list of tokens
        midi_path: path to midi if tokens are not provided
        file_name: file to save output
    """
    fs = FluidSynth('midi_to_audio/Full_Grand.sf2')
    if midi_path is None:
        os.makedirs('midi_to_audio/tmp_midis', exist_ok=True)
        name, ext = os.path.splitext(os.path.basename(file_name))
        midi_path = f'midi_to_audio/tmp_midis/{name}.midi'
        decode_midi(tokens, midi_path)
    fs.midi_to_audio(midi_path, file_name)
    os.remove(midi_path)
