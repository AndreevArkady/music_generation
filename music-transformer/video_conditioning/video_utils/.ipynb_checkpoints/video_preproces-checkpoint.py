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