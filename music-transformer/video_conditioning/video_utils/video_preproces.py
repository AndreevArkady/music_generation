import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip

def add_time_bar(video_path, output_path, bar_thickness=6, bar_opacity=0.5):
    video_path = str(video_path)
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize VideoWriter to save the video with the time bar
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = 'temp_with_time_bar_123182739zxc.mp4'
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate the position of the time bar
        bar_length = int((frame_number / total_frames) * width)
        bar_y_position = height - bar_thickness - 5

        # Create a transparent overlay
        overlay = frame.copy()
        cv2.rectangle(img=overlay, 
                      pt1=(0, bar_y_position), 
                      pt2=(bar_length, bar_y_position + bar_thickness), 
                      color=(0, 255, 0), 
                      thickness=-1)

        # Blend the overlay with the frame
        cv2.addWeighted(overlay, bar_opacity, frame, 1 - bar_opacity, 0, frame)

        # Write the frame with the time bar
        out.write(frame)

        frame_number += 1

    # Release everything if job is finished
    cap.release()
    out.release()

    # Use moviepy to add audio back to the new video
    original_clip = VideoFileClip(video_path)
    new_clip = VideoFileClip(temp_video_path)
    final_clip = new_clip.set_audio(original_clip.audio)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='libmp3lame', fps=final_clip.fps)

    # Delete the temporary video file
    os.remove(temp_video_path)

def add_time_bar_with_markers(
        video_path, 
        output_path, 
        key_moments, 
        key_moments_2=None, 
        bar_thickness=6, 
        bar_opacity=0.5, 
        marker_size=5
    ):
    video_path = str(video_path)
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize VideoWriter to save the video with the time bar
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = 'temp_with_time_bar_123432zxc.mp4'
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    frame_number = 0

    # Convert key moments from seconds to frame indices
    key_frame_indices = [int(t * fps) for t in key_moments]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate the position of the time bar
        bar_length = int((frame_number / total_frames) * width)
        bar_y_position = height - bar_thickness - 5

        # Create a transparent overlay
        overlay = frame.copy()
        cv2.rectangle(img=overlay, 
                      pt1=(0, bar_y_position), 
                      pt2=(bar_length, bar_y_position + bar_thickness), 
                      color=(0, 255, 0), 
                      thickness=-1)

        # Draw markers for key moments
        for key_frame in key_frame_indices:
            marker_x_position = int((key_frame / total_frames) * width)
            marker_y_position = bar_y_position - marker_size - 5
            cv2.rectangle(img=overlay,
                          pt1=(marker_x_position - marker_size//2, marker_y_position),
                          pt2=(marker_x_position + marker_size//2, marker_y_position + marker_size),
                          color=(0, 0, 255),
                          thickness=-1)
            
        if key_moments_2 is not None:
            key_frame_indices_2 = [int(t * fps) for t in key_moments_2]
            for key_frame in key_frame_indices_2:
                marker_x_position = int((key_frame / total_frames) * width)
                marker_y_position = bar_y_position - marker_size - 10
                cv2.rectangle(img=overlay,
                            pt1=(marker_x_position - marker_size//2, marker_y_position),
                            pt2=(marker_x_position + marker_size//2, marker_y_position + marker_size),
                            color=(255, 0, 0),
                            thickness=-1)
            

        # Blend the overlay with the frame
        cv2.addWeighted(overlay, bar_opacity, frame, 1 - bar_opacity, 0, frame)

        # Write the frame with the time bar and markers
        out.write(frame)

        frame_number += 1

    # Release everything if job is finished
    cap.release()
    out.release()

    # Use moviepy to add audio back to the new video
    original_clip = VideoFileClip(video_path)
    new_clip = VideoFileClip(temp_video_path)
    final_clip = new_clip.set_audio(original_clip.audio)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='libmp3lame', fps=final_clip.fps)

    # Delete the temporary video file
    os.remove(temp_video_path)
    

import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip

def add_time_bar_with_markers_any(
        video_path, 
        output_path, 
        bar_thickness=6, 
        bar_opacity=0.5, 
        marker_size=5, 
        args=tuple()
    ):
    """
    Adds a time bar and multiple sets of key moment markers to a video.

    This function processes an input video to add a time bar at the bottom and draw markers above the time bar at specified key moments. Each set of key moments is displayed in a different color and at different heights above the time bar.
	args[0] - is on top, args[-1] on the bottom, time bar below all of them 
    
    Arguments:
        video_path (str): Path to the input video file.
        output_path (str): Path to the output video file.
        bar_thickness (int, optional): Thickness of the time bar. Default is 6.
        bar_opacity (float, optional): Opacity of the time bar. Default is 0.5.
        marker_size (int, optional): Size of the markers (squares) to be drawn above the time bar. Default is 5.
        args list[list[float]]: Variable number of lists of key moments in seconds.

    Returns:
        None
    """
    video_path = str(video_path)
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize VideoWriter to save the video with the time bar
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = 'temp_with_time_bar_any_132zxc.mp4'
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    frame_number = 0

    # Generate different colors for each list of key moments
    colors = [
        (0, 255, 255), # Yellow
        (0, 0, 255),   # Red
        (0, 255, 0),   # Green
        (255, 255, 0), # Cyan
        (255, 0, 0),   # Blue
        (255, 0, 255), # Magenta
    ]

    args = args[::-1]
    # Ensure we have enough colors
    while len(colors) < len(args):
        colors.append((np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)))

    # Convert key moments from seconds to frame indices for all key moment lists
    key_frame_indices_list = [[int(t * fps) for t in key_moments] for key_moments in args]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate the position of the time bar
        bar_length = int((frame_number / total_frames) * width)
        bar_y_position = height - bar_thickness - 5

        # Create a transparent overlay
        overlay = frame.copy()
        cv2.rectangle(img=overlay, 
                      pt1=(0, bar_y_position), 
                      pt2=(bar_length, bar_y_position + bar_thickness), 
                      color=(0, 255, 0), 
                      thickness=-1)

        # Draw markers for each list of key moments
        for i, key_frame_indices in enumerate(key_frame_indices_list):
            color = colors[i % len(colors)]
            for key_frame in key_frame_indices:
                marker_x_position = int((key_frame / total_frames) * width)
                marker_y_position = bar_y_position - marker_size - 5 - (i * (marker_size + 5))
                cv2.rectangle(img=overlay,
                              pt1=(marker_x_position - marker_size//2, marker_y_position),
                              pt2=(marker_x_position + marker_size//2, marker_y_position + marker_size),
                              color=color,
                              thickness=-1)

        # Blend the overlay with the frame
        cv2.addWeighted(overlay, bar_opacity, frame, 1 - bar_opacity, 0, frame)

        # Write the frame with the time bar and markers
        out.write(frame)

        frame_number += 1

    # Release everything if job is finished
    cap.release()
    out.release()

    # Use moviepy to add audio back to the new video
    original_clip = VideoFileClip(video_path)
    new_clip = VideoFileClip(temp_video_path)
    final_clip = new_clip.set_audio(original_clip.audio)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='libmp3lame', fps=final_clip.fps)

    # Delete the temporary video file
    os.remove(temp_video_path)

