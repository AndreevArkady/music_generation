import sys
sys.path.append("/storage/arkady/Glinka/music-transformer/video_conditioning/content/computervision-recipes")

import cv2
import numpy as np
import decord
import torch
from collections import deque
from utils_cv.action_recognition.data import KINETICS
from utils_cv.action_recognition.dataset import get_transforms
from utils_cv.action_recognition.model import VideoLearner
from tqdm import tqdm

LABELS = KINETICS.class_names
NUM_FRAMES = 8
SCORE_THRESHOLD = 0.16

learner = VideoLearner(
    base_model="kinetics",
    sample_length=NUM_FRAMES,
)

transforms = get_transforms(train=False)

def VideoToMoments(path_video, SCORE_THRESHOLD=0.16) -> list:
    cap = cv2.VideoCapture(path_video)
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'Video: {path_video}, FPS: {FPS}')

    video_reader = decord.VideoReader(path_video)
    window = deque(maxlen=NUM_FRAMES)
    scores_cache = deque()
    scores_sum = np.zeros(len(LABELS))
    
    key_video_moments = []
    prev_frame_sec = 0
    prev_sets = []

    for cur_frame, frame in tqdm(enumerate(video_reader)):
        frame = frame.asnumpy()
        window.append(frame)

        if len(window) == NUM_FRAMES:
            # Check if CUDA is available
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            learner.model.to(device)
            learner.model.eval()

            action, set_a = learner.predict_frames(
                window,
                scores_cache,
                scores_sum,
                None,
                30,
                SCORE_THRESHOLD,
                LABELS,
                LABELS,
                transforms,
                None,
            )

            cur_frame_sec = cur_frame // FPS + (cur_frame % FPS) / FPS
            if len(prev_sets) > FPS:
                prev_cumm_set = set().union(*prev_sets[-FPS:])
            else:
                prev_cumm_set = set().union(*prev_sets)

            if action and (cur_frame == 0 or (action not in prev_cumm_set and cur_frame_sec - prev_frame_sec > 0.5)):
                key_video_moments.append(float(f"{cur_frame_sec:.2f}"))
                prev_frame_sec = cur_frame_sec
                prev_sets.append(set_a)

    return key_video_moments

# Example usage
if __name__ == "__main__":
    video_path = '/storage/arkady/Glinka/music-transformer/video_conditioning/content/video_data/video_ad14.mp4'
    key_moments = VideoToMoments(video_path)
    print("Key moments:", key_moments)
