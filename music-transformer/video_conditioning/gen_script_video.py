import os
import sys
import json
from pathlib import Path
import subprocess
import typing as tp

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# from code3.utilities.constants import GENRES_LIST
GENRES_LIST = ( # noqa WPS317
    'pop', 'jazz', 'rock', 'blues', 'classical', 'country',
    'soul', 'rap', 'latin', 'folk', 'electro', '[UNK]',
)
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_audioclips

from CONSTANTS import PYTHON_TRANSFORMER_PRODUCTION, PATH_video_conditioning, PYTHON_VIDEO_MUSIC
from video_utils.mp4_to_midi import mp4_to_midi
from video_utils.kvm import kvm_to_nnotes, draw_kvm_on_video, extract_kvm3
from video_utils.video_preproces import add_time_bar_with_markers_any


def log(obj):
    print(f'{obj=}')

def merge_video_audio(video_path, audio_path, output_path):
    # Load the video file
    video_clip = VideoFileClip(video_path)
    
    # Load the audio file
    audio_clip = AudioFileClip(audio_path)
    
    # The duration of the video
    video_duration = video_clip.duration

    # Check if the audio clip is shorter than the video clip
    if audio_clip.duration < video_duration:
        # Calculate the number of times the audio needs to be repeated
        repeat_count = int(video_duration // audio_clip.duration) + 1
        # Create a new audio clip by concatenating the audio clip with itself
        audio_clip = concatenate_audioclips([audio_clip] * repeat_count)
    
    # Set the duration of the audio clip to match the video clip's duration
    audio_clip = audio_clip.set_duration(video_duration)
    
    # Set the audio of the video clip to the audio clip
    video_clip = video_clip.set_audio(audio_clip)
    
    # Write the result to a file
    # I hardcode commented fps decorator in write_videofile, this shit wasn't working
    video_clip.write_videofile(output_path, codec='libx264', audio_codec='libmp3lame', fps=video_clip.fps)

# def generation_generate_sample(
#     output_dir: str,
#     output_name: str
# ):
#     pr = subprocess.run(  # running according to Instruction.md
#         f"""cd code3;
#         {PYTHON_TRANSFORMER_PRODUCTION} -m \
#         generation.generate_sample \
#         {output_dir} \
#         {output_name} \
#         """, 
#         shell=True, capture_output=True, text=True
#         )
#     print(f"ggs {pr.args}")
#     print(f"ggs pr.stdout")
#     print(f"ggs {pr.stdout}")
#     print(f"ggs pr.stderr")
#     print(f"ggs {pr.stderr}")

def stream_subprocess_output(command):  # for some reason don't work as expected, not intermediate logs
    # Start the subprocess using bash explicitly
    process = subprocess.Popen(
        command,
        shell=True,
        executable="/bin/bash",  # Specify the Bash executable
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Monitor both stdout and stderr
    while True:
        # Poll process for new output: both stdout and stderr
        output = process.stdout.readline()
        error_output = process.stderr.readline()
        # Check if subprocess has terminated and stdout/stderr are empty
        if process.poll() is not None and output == '' and error_output == '':
            break
        # Print any outputs collected to stdout
        if output:
            print(output.strip())
        # Print any errors collected to stderr, where tqdm typically writes
        if error_output:
            print(error_output.strip(), end='')  # Use end='' for tqdm to properly update the line
    # Ensure all output is flushed after subprocess termination
    stdout, stderr = process.communicate()
    if stdout:
        print(stdout.strip())
    if stderr:
        print('*', stderr.strip())

def generation_generate_sample(
    # output_dir: str,
    # output_name: str
    path_parameters: str,
    gpu: int = 0
):
    command = f"""cd code3;
    CUDA_VISIBLE_DEVICES={gpu} \
    {PYTHON_TRANSFORMER_PRODUCTION} -m \
    generation.generate_sample \
    {path_parameters} \
    """
    print(f"{command=}")
    
    # stream_subprocess_output(command)

    pr = subprocess.run(command, 
        shell=True, capture_output=True, text=True)
    print(f"ggs {pr.args}")
    print(f"ggs pr.stdout")
    print(f"ggs {pr.stdout}")
    print(f"ggs pr.stderr")
    print(f"ggs {pr.stderr}")

def create_config(
        args, 
        path_nnotes, 
        length=None,
        genre=None,
    ) -> str:
    
    model_weights = args[1]
    time_ms = length or args[2]  # args[2] not used currently, time extracted from video
    sentiment = args[3]
    print(args)
    genre = genre or GENRES_LIST[int(args[4])]    
    output_dir = str(Path(PATH_video_conditioning) / args[5]) # first part is redundant
    output_name = args[6]
    path_mp4 = args[7]
    random_seed = 0
    if len(args) > 8:
        random_seed = int(args[8])

    with open(
    Path(PATH_video_conditioning) / 'code3' \
                / 'generation' / 'generate_sample_parameters.json', 'r'
              ) as default_config: # loading base parameters
        parameters = json.load(default_config)  # noqa: WPS110

    parameters['model_weights'] = model_weights
    # parameters['generation_params']['additional_features_params']['timing'] = int(time_ms)
    #parameters['generation_params']['additional_features_params']['sentiment_per_token'] = int(sentiment)
    parameters['generation_params']['additional_features_params']['genre'] = genre
    parameters['saving_options']['output_dir'] = output_dir
    parameters['saving_options']['output_folder_name'] = output_name
    parameters['random_seed'] = random_seed

    parameters['generation_params']['additional_features_params']['nnotes']['enable'] = True
    parameters['generation_params']['additional_features_params']['nnotes']['preset'] = \
        json.load(open(path_nnotes, 'r'))
    
    path_nnotes = Path(path_nnotes)
    # generation/generate_sample reads config from this, aka '{0}/generate_sample_{1}_parameters.json'
    cfg_filename = path_nnotes.parent / f"generate_sample_{path_nnotes.parent.stem}_parameters.json"
    with open(cfg_filename, 'w') as generation_cfg:
        json.dump(parameters, generation_cfg, indent=4)
        
    return str(cfg_filename)

def generate_music_for_video(  # can be imported and used outside of the script
    args: tp.Sequence,
    path_mp4: str,
    output_dir: str = './tmp_e2e',
    *,
    genre: str = 0,
    kvm2_threshold:float = 0.8,
    gpu: int = 0
):
    # end_to_end.ipynb code
    log('generate_music_for_video')
    path_mp4 = Path(path_mp4)
     
    os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libffi.so.7'
    
    # 1 mp4 -> mp3, midi, ...
    log('#1')
    path_midi = mp4_to_midi( # UNCOMMENT
        path_to_mp4=path_mp4,
        output_dir=output_dir,
    )

    # # 2.2 kvm   # это было со старым алгоритмом kvm
    # # generating kvm2.csv
    # # не получилось переписать одним запуском без sys.path.append
    # log('#2')
    # sys.path.append('/storage/arkady/Glinka/music-transformer/video_conditioning/content/video-keyframe-detector')
    # pr_kvm = subprocess.run(f'''
    #     {PYTHON_VIDEO_MUSIC} \
    #     content/video_keyframe_detector/cli.py \
    #     -s {str(path_mp4)} \
    #     -d {output_dir} \
    #     -t {kvm2_threshold} \
    #     ''',
    #     shell=True, capture_output=True, text=True
    #     )
    # print(f"{pr_kvm.stdout=}")
    # print(f"{pr_kvm.stderr=}")
    # path_kvm = Path(output_dir) / path_mp4.stem / 'kvm2.csv'

    # 2.2
    # log('#2')
    # kvm3, fps3, *kvm4 = extract_kvm3(path_mp4, threshold=70.0)
    kvm_cfg = json.load(open('kvm_threshold.json', 'r'))
    largest_ind = 0
    for ind, part in enumerate(kvm_cfg[path_mp4.stem]):
        if part['len'] > kvm_cfg[path_mp4.stem][largest_ind]['len']:
            largest_ind = ind
    print(f'{largest_ind=}')

    
    threshold = kvm_cfg[path_mp4.stem][largest_ind]['threshold']

    args_kvm = extract_kvm3(path_mp4, threshold=threshold)

    df = pd.DataFrame({
    	'frame_ind': list(range(args_kvm[3])),
    	'key_moment': np.zeros_like(args_kvm[1]),
    	'fps': float(args_kvm[1])
    })
    kvm4_t = args_kvm[2]
    kvm4 = [k[0].get_frames() for k in kvm4_t] + [kvm4_t[-1][1].get_seconds()]
    df['key_moment'] = df['frame_ind'].isin(kvm4).astype(float)
    path_kvm = Path(output_dir) / path_mp4.stem / 'kvm4.csv'
    df.to_csv(str(path_kvm), index=False)

    # 3
    log('#3')
    path_nnotes = Path(output_dir) / path_mp4.stem / 'nnotes4.json'
    video_length_s = kvm_to_nnotes(
        path_kvm=path_kvm,
        output_filename=path_nnotes,
        max_nnotes=3,
    )


    # 3.5 config generation
    log('#3.5')
    path_parameters = create_config(
        args=args, 
        path_nnotes=path_nnotes, 
        length=video_length_s * 100)
    log('create_config')


    # 4. music-transformer generation
    log('#4')
    generation_generate_sample(
        path_parameters=Path(PATH_video_conditioning) / path_parameters,
        gpu=gpu
    )

    # 5. glue video and music
    log('#5')
    merge_video_audio( # я уже не соображаю как нормально достать пути
        video_path=str(path_mp4), 
        audio_path=str(Path(args[5]) / args[6] / f"{args[6]}.mp3"), 
        output_path=str(Path(args[5]) / args[6] / f"{args[6]}.mp4")
    )

    # # 6. fraw kvm on video with '_kvm' suf
    # log('#6')
    # draw_kvm_on_video(
    #     str(Path(args[5]) / args[6] / f"{args[6]}.mp4"),
    #     str(Path(args[5]) / args[6] / f"{args[6]}_kvm.mp4"),
    #     path_kvm=str(Path(args[5]) / path_mp4.stem / f"kvm2.csv"),  # kvm is the same for the same vidoe
    # )

    # 6. 
    add_time_bar_with_markers_any(
        video_path=str(Path(args[5]) / args[6] / f"{args[6]}.mp4"),
        output_path=str(Path(args[5]) / args[6] / f"{args[6]}_kvm.mp4"),
        bar_thickness=6, 
        bar_opacity=0.5, 
        marker_size=6, 
    	args=[kvm_cfg[path_mp4.stem][largest_ind]['kvm4']]
    )
    

def main(args):
    '''
    Arguments:
    1 model_weights.pickle path
    2 time_ms - length of generated music  *10ms
    3 sentiment - not used
    4 genre - from GENRES_LIST = ('pop', 'jazz', 'rock', 'blues', ...)
    5 output_dir
    6 output_name
    7 path_mp4
    8 (OPTIONAL) random seed 
    9 gpu index
    

Returns:
    saves config - modified from generate_sample_parameters.json
    runs code3/generate_sample.py

    creates output_dir/output_name.wav
    creates output_dir/output_name.mp3
    '''
    output_dir = args[5]
    path_mp4 = args[7]
    gpu = int(args[9])
    assert 0 <= gpu <= 5, 'gpu index is wrong'
    generate_music_for_video(
        args=args,
        path_mp4=path_mp4,
        output_dir=output_dir,
        gpu=gpu
    )

if __name__ == '__main__':
    main(sys.argv)
