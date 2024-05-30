BASEPATH = '/storage/arkady/Glinka/music_generation'

from PIL import Image, ImageTk
import moviepy
import moviepy.editor
import os
import json

import torch
# Regular Python libraries
import sys
from collections import deque, defaultdict
import io
import requests
import os
from pathlib import Path
from time import sleep, time
from threading import Thread
from IPython.display import Video

# Third party tools
import decord #
import IPython.display #
# from ipywebrtc import CameraStream, ImageRecorder
from ipywidgets import HBox, HTML, Layout, VBox, Widget, Label
import numpy as np
from PIL import Image
import torch
import torch.cuda as cuda
import torch.nn as nn
from torchvision.transforms import Compose

# utils_cv
sys.path.append("/storage/arkady/Glinka/PrVideoMusic/content/computervision-recipes")

from utils_cv.action_recognition.data import KINETICS, Urls
from utils_cv.action_recognition.dataset import get_transforms
from utils_cv.action_recognition.model import VideoLearner
from utils_cv.action_recognition.references import transforms_video as transforms
from utils_cv.common.gpu import system_info, torch_device
from utils_cv.common.data import data_path

import pickle
import os
from pathlib import Path
import moviepy.editor
import subprocess

def video_to_midi(
    filepath=None, 
    video_dir=Path(BASEPATH) / 'content' / 'video_data', 
    tempdir=Path(BASEPATH) / 'tmp'
):
    """
    Convert a video filename to MIDI by extracting audio, removing vocals, and processing pitch.
    TODO: переписать в 3 функции и вызывать их поочереди
    """
    
    filepath = str(filepath)
    assert Path(filepath).is_file(), f'File {filepath} does not exists'
    tempdir.mkdir(parents=True, exist_ok=True)
    
    # mp4 to mp3
    filename = Path(filepath).stem
    mp3_abspath = tempdir / f'{filename}.mp3'
    video = moviepy.editor.VideoFileClip(str(filepath))
    video.audio.write_audiofile(str(mp3_abspath))  # saving mp3
    print(f'=====Successfully written new mp3 to {mp3_abspath=}')
    
    # Vocal remover: mp3 -> Instruments and Vocals WAV files
    vocal_remover_dir = Path(BASEPATH) / 'content' / 'vocal-remover'
    os.chdir(vocal_remover_dir)
    subprocess.run(['python', 'inference.py', '--input', str(mp3_abspath), '--output_dir', str(tempdir)])
    
    # Basic pitch processing: .wav -> .midi
    path_to_instruments_wav = tempdir / f'{filename}_Instruments.wav'  # files from above !python inference
    path_to_vocals_wav = tempdir / f'{filename}_Vocals.wav'
    print(f"=====Instrumentals at: {path_to_instruments_wav}\n=====Vocals at: {path_to_vocals_wav}")
    subprocess.run(['basic-pitch',
                    str(Path(BASEPATH) / 'content' / 'video_to_midi'),
                    str(path_to_instruments_wav)])

    old_name = Path(BASEPATH) / 'content' / 'video_to_midi' / f'{filename}_Instruments_basic_pitch.mid'
    old_name.rename(old_name.with_name(f'music_orig_{filename}.mid'))
    
    path_to_instruments_wav.rename(Path(BASEPATH) / 'content' / 'video_to_wav' / f'music_orig_{filename}.wav')
    
    # Clean up: remove temporary files
    # mp3_abspath.unlink(missing_ok=True)
    path_to_vocals_wav.unlink(missing_ok=True)


# Example usage
# video_to_midi(filepath=Path(BASEPATH) / "content/video_data/video_ad74.mp4")

if __name__ == '__main__':
	# test
# %cd {BASEPATH}

    for num in range(1, 85):
        print(num)
        video_to_midi(filepath=Path(BASEPATH) / f"content/video_data/video_ad{num}.mp4")