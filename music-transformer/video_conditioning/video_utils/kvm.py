import json
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

def smooth_sequence(seq, max_dist=10):
    n = len(seq)
    # Initialize a list to store distances to the nearest '1'
    distances = [float('inf')] * n

    # First pass: find distances from left to right
    for i in range(n):
        if seq[i] == 1:
            distances[i] = 0
        elif i > 0 and distances[i-1] != float('inf'):
            distances[i] = distances[i-1] + 1

    # Second pass: find distances from right to left
    for i in range(n-2, -1, -1):
        if distances[i+1] != float('inf'):
            distances[i] = min(distances[i], distances[i+1] + 1)

    # Determine the maximum distance to scale values
    max_distance = max(distances)

    # Assign values based on proximity to '1'
    transformed = []
    for dist in distances:
        if dist == 0:
            transformed.append(max_dist)  # '1' corresponds to 10
        else:
            # Scale each zero to be between 0 and 9 based on its distance
            scaled_value = round((1 - (dist / max_distance)) ** 2.2 * (max_dist - 1))
            transformed.append(max(1, scaled_value))

    return transformed

def resize_sequence(seq, target_length=128):
    # Convert sequence to numpy array for easier handling
    original_length = len(seq)
    original_indices = np.linspace(0, original_length - 1, num=original_length)
    target_indices = np.linspace(0, original_length - 1, num=target_length)
    
    # Create interpolation function
    interpolation_function = interp1d(original_indices, seq, kind='linear')
    
    # Generate new sequence of the target length
    new_sequence = interpolation_function(target_indices)
    # Round values to maintain integer type consistency
    new_sequence = np.round(new_sequence).astype(int)
    
    return new_sequence.tolist()

def kvm_to_nnotes(path_kvm: str, output_filename: str, max_nnotes=10):
    df = pd.read_csv(path_kvm)
    kvm = df['key_moment']
    smoothed_kvm = smooth_sequence(kvm, max_dist=max_nnotes)
    smoothed_128_kvm = resize_sequence(smoothed_kvm, target_length=128)
    l, r = 0, 0
    nnotes_cfg = []
    while r < len(smoothed_128_kvm):  # always *l = *r
        if (r < len(smoothed_128_kvm) - 1) and smoothed_128_kvm[l] == smoothed_128_kvm[r + 1]:
            r += 1
        else:
            nnotes_cfg.append({'borders': [l, r + 1], 'value': smoothed_128_kvm[l]})
            l = r + 1
            r = r + 1
    assert nnotes_cfg[-1]['borders'][1] == 128

    json.dump(nnotes_cfg, open(output_filename, 'w'), indent=4)
    return len(df) / df.iloc[-1, -1]  # amount of frames / fps
    

import cv2
from moviepy.editor import VideoFileClip
import numpy as np
import os

def draw_kvm_on_video (  # DEPR, use video_process.add_time_bar_with_markers_any instead
        input_video_path, 
        output_video_path, 
        path_kvm, 
        diameter=50, 
        color=(0, 0, 255)
        ):
    """
    Draws a red circle at the top right corner of specified frames in a video 
    and preserves audio. 
    For each kvm frame with index i draws a circle at frames i+-2 

    Args:
    input_video_path (str): Path to the input video file.
    output_video_path (str): Path where the output video will be saved.
    path_kvm (str): path to kvm timings file.csv with List of frame indices where the circle should be drawn.
    diameter (int, optional): Size of the circle side in pixels. Default is 50.
    color (tuple, optional): Color of the circle in BGR format (Blue, Green, Red). Default is red.
    """
    # Create a temporary video path for the mute video
    temp_video_path = f'temp_output_{np.random.randint(999)}.mp4'

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec for saving video
    
    # Create a VideoWriter object to save the mute video
    out = cv2.VideoWriter(temp_video_path, codec, fps, (width, height))

    # Process the video
    current_frame = 0
    # frames_to_mark = np.array(frames_to_mark)
    frames_to_mark = pd.read_csv(path_kvm).query('key_moment == 1')['frame_ind'].values
    colored_frames = np.concatenate([frames_to_mark + shift for shift in range(-2, 3)])
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the current frame or its neighbors need the square
        if current_frame in colored_frames:
            # Coordinates for the square (bottom left corner)

            # start_point = (width - square_size, 0)
            # end_point = (width, square_size)
            # # Draw the rectangle on the frame
            # frame = cv2.rectangle(frame, start_point, end_point, color, -1)
            
            # Coordinates and radius for the circle (top right corner)
            center = (width - diameter // 2, diameter // 2)  # Center of the circle
            radius = diameter // 2  # Radius of the circle
            # Create a copy of the frame to overlay the circle
            overlay = frame.copy()
            # Draw a filled circle on the overlay
            cv2.circle(overlay, center, radius, color, -1)
            cv2.circle(overlay, center, radius // 2, (128, 128, 255), -1)
            # Alpha factor for transparency (between 0 and 1, where 1 is completely opaque)
            alpha = 0.7
            # Apply the overlay with transparency
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Write the frame to the output video
        out.write(frame)
        current_frame += 1

    # Release everything when job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Load the original video and temporary mute video
    original_clip = VideoFileClip(input_video_path)
    video_clip = VideoFileClip(temp_video_path)

    # Set the audio of the original video to the video clip
    video_clip = video_clip.set_audio(original_clip.audio)

    # Write the final video with audio
    video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='libmp3lame', fps=video_clip.fps)

    # Cleanup
    video_clip.close()
    original_clip.close()
    os.remove(temp_video_path)
    print("Video processing complete and saved to:", output_video_path)


from tqdm import tqdm
import peakutils
import time


def scale(img, xScale, yScale):
    res = cv2.resize(img, None, fx=xScale, fy=yScale, interpolation=cv2.INTER_AREA)
    return res

def convert_frame_to_grayscale(frame):
    grayframe = None
    gray = None
    if frame is not None:
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = scale(gray, 1, 1)
        grayframe = scale(gray, 1, 1)
        gray = cv2.GaussianBlur(gray, (9, 9), 0.0)
    return grayframe, gray

def extract_kvm2(source, thres=0.6, min_dist=10) -> tuple:
    '''
    Taken from key-frame-detection module
    
    returns:
        list[float] of frame changes
        fps (flaot)
    '''
    source = str(source)
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)  # ARKADY
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
    if (cap.isOpened()== False):
        print("Error opening video file")

    lstfrm = []
    lstdiffMag = []
    timeSpans = []
    images = []
    full_color = []
    lastFrame = None
    Start_time = time.process_time()
    
    # Read until video is completed
    for i in tqdm(range(length)):
        ret, frame = cap.read()
        grayframe, blur_gray = convert_frame_to_grayscale(frame)

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        lstfrm.append(frame_number)
        images.append(grayframe)
        full_color.append(frame)
        if frame_number == 0:
            lastFrame = blur_gray

        diff = cv2.subtract(blur_gray, lastFrame)
        diffMag = cv2.countNonZero(diff)
        lstdiffMag.append(diffMag)
        stop_time = time.process_time()
        time_Span = stop_time-Start_time
        # timeSpans.append(time_Span)
        timeSpans.append(i/fps)  # ARKADY, i/fps?
        lastFrame = blur_gray

    cap.release()
    y = np.array(lstdiffMag)
    base = peakutils.baseline(y, 2)
    indices = peakutils.indexes(y-base, thres, min_dist=min_dist)
    
    return indices, fps


def get_video_duration(path_mp4):
    '''
    returns:
        duration of video in seconds
        fps
        total_frames
    '''
    path_mp4 = str(path_mp4)
    assert os.path.exists(path_mp4), 'file does not exists'

    cap = cv2.VideoCapture(path_mp4)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    return duration, fps, total_frames

import cv2
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def extract_kvm3(video_path, threshold=30.0):
    """
    Written with GPT-4o
    Extracts key moments from a given video based on scene changes.

    This function uses the `opencv-python` and `pyscenedetect` libraries to detect scene changes
    in the video. It identifies key moments by finding the middle frame of each detected scene.

    Parameters:
    video_path (str): The path to the input video file.
    threshold (float): The threshold value for the `ContentDetector` to detect scene changes.
                       Higher values make the scene detection more sensitive.

    Returns:
    tuple: A tuple containing:
        - key_moments_scores (list[int]): A list with frame numbers.
        - fps (float): The frames per second (FPS) of the input video.

    Example:
    >>> key_moments_scores, fps = extract_key_moments('path/to/video.mp4', threshold=30.0)
    """
    video_path = str(video_path)
    assert os.path.exists(video_path), 'file does not exists'


    # Initialize VideoCapture object
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Initialize lists to store key frames and scores
    key_frames = []
    scores = []

    # Initialize SceneDetect VideoManager and SceneManager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Start video_manager and perform scene detection
    video_manager.set_duration()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # List of scenes with start and end frames
    scene_list = scene_manager.get_scene_list()

    # Iterate over scenes to identify key moments
    for i, scene in enumerate(scene_list):
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()

        # For simplicity, we can consider the middle frame of each scene as a key moment
        key_frame = int((start_frame + end_frame) / 2)
        key_frames.append(key_frame)

    # Release VideoCapture and VideoManager
    cap.release()
    video_manager.release()

    return key_frames, fps, scene_list, total_frames


def search_thresholds_for_kvm3(
        path_mp4, 
        min_length=4, 
        max_length=12, 
        max_searches=200,
        config_file='kvm_treshold.json'
    ):
    """
    Searches for threshold values that result in kvm3 and kvm4 lists of a desired length
    and saves the results to a JSON config file.

    Arguments:
        path_mp4 (str or Path): Path to the input video file.
        min_length (int): Minimum acceptable length of kvm3. Default is 4.
        max_length (int): Maximum acceptable length of kvm3. Default is 12.
        max_searches (int): Maximum number of binary search steps. Default is 200.

    Returns:
        None
    """
    path_mp4 = Path(path_mp4)
    if not os.path.exists(path_mp4):
        return

    # config_file = 
    
    # Initialize config dictionary
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {}
        json.dump({}, open(config_file, 'w'))
    
    if path_mp4.stem not in config:
        config[path_mp4.stem] = []

    found_thresholds = set(threshold for threshold, _, __ in config[path_mp4.stem])

    total_iter = -1
    low, high = 1, max_searches
    while low <= high and len(found_thresholds) < 3:
        total_iter += 1
        if total_iter > max_searches:
            break
        mid = (low + high) // 2
        kvm3, fps3, kvm4 = extract_kvm3(path_mp4, threshold=mid)
        # print(f'{kvm4=}')
        kvm4 = [] if (not kvm4) else [k[0].get_seconds() for k in kvm4] + [kvm4[-1][1].get_seconds()]
        len_kvm3 = len(kvm3)
        print(f'{mid=}, {len_kvm3=}')

        if min_length <= len_kvm3 <= max_length and mid not in found_thresholds:
            print(f'writing {mid}')
            found_thresholds.add(mid)
            # config[path_mp4.stem].append([mid, len_kvm3, kvm4])
            config[path_mp4.stem].append(
                {
                    'threshold': mid,
                    'len': len_kvm3,
                    'kvm4': kvm4,
                }
                )
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
        
        if len_kvm3 < min_length:
            high = mid - 1
        elif len_kvm3 > max_length:
            low = mid + 1
        else:
            # If we found a valid threshold, continue searching around it
            # low, high = mid - 1, mid + 1
            if len_kvm3 > (min_length + max_length) / 2:
                low = mid + 1
            else:
                high = mid - 1