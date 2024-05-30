import os
from pathlib import Path
import moviepy.editor
import subprocess

# BASEPATH = '/storage/arkady/Glinka/music-transformer/video_conditioning'

def mp4_to_midi(
    path_to_mp4=None,
	path_no_midi=None,
    *,
    tempdir=Path('/storage/arkady/Glinka/music-transformer/video_conditioning') / 'tmp',
    BASEPATH='/storage/arkady/Glinka/music-transformer/video_conditioning'
):
    """
    Convert a video basename to MIDI by extracting audio, removing vocals, and processing pitch.
    TODO: переписать в 3 функции и вызывать их поочереди
    """
    
    path_to_mp4 = str(path_to_mp4)
    assert Path(path_to_mp4).is_file(), f'File {path_to_mp4} does not exists'

    if os.path.isabs(tempdir):
        tempdir = Path(tempdir)
    else:
        tempdir = Path(BASEPATH) / tempdir
    tempdir.mkdir(parents=True, exist_ok=True)
    
    # mp4 to mp3
    basename = Path(path_to_mp4).stem
    path_mp3 = tempdir / f'{basename}.mp3'
    video = moviepy.editor.VideoFileClip(str(path_to_mp4))
    video.audio.write_audiofile(str(path_mp3))  # saving mp3
    print(f'=====Successfully written new mp3 to {path_mp3=}')  # or content/video_to_mp3
    
    # Vocal remover: mp3 -> Instruments and Vocals, 2 WAV files
    vocal_remover_dir = Path(BASEPATH) / 'content' / 'vocal-remover'
    # os.chdir(vocal_remover_dir)
    print(os.getcwd())
    subprocess.run([
        '/storage/arkady/miniconda3/envs/VideoMusic/bin/python3',
        # 'inference.py', 
        '/storage/arkady/Glinka/music-transformer/video_conditioning/content/vocal-remover/inference.py', 
        '--input', 
        str(path_mp3),
        '--output_dir', 
        str(tempdir)
    ])
    
    # Basic pitch processing: .wav -> .midi
    path_to_instruments_wav = tempdir / f'{basename}_Instruments.wav'  # files from above !python inference
    path_to_vocals_wav = tempdir / f'{basename}_Vocals.wav'
    print(f"=====Instrumentals at: {path_to_instruments_wav}\n=====Vocals at: {path_to_vocals_wav}")

    os.environ['PATH'] += os.pathsep + '/storage/arkady/miniconda3/envs/VideoMusic/bin'  # instead of conda activate VideoMusic
    os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libffi.so.7'  # for !basic-pitch
    subprocess.run(['basic-pitch',
                    # str(Path(BASEPATH) / 'content' / 'video_to_midi'),
                    str(tempdir),
                    str(path_to_instruments_wav)])

    # file_mid = Path(BASEPATH) / 'content' / 'video_to_midi' / f'{basename}_Instruments_basic_pitch.mid'
    file_mid = tempdir / f'{basename}_Instruments_basic_pitch.mid'
    file_mid.rename(file_mid.with_name(f'bapi_{basename}.mid'))
    
    # path_to_instruments_wav.rename(Path(BASEPATH) / 'content' / 'video_to_wav' / f'music_orig_{basename}.wav')
    
    # Clean up: remove temporary files
    # path_mp3.unlink(missing_ok=True)
    # path_to_vocals_wav.unlink(missing_ok=True)

    return file_mid
