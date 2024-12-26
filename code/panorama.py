import cv2
import numpy as np
from tqdm import tqdm
from sift import keypoint_match, draw_match, transform
from utils import read_video_frames, write_frames_to_video, write_and_show, destroyAllWindows, imshow

video_name = 'image/winter_day.mov'
images, fps = read_video_frames(video_name)
n_image = len(images)

# TODO: init panorama
# 2.1
panorama = ...

trans_sum = np.zeros([H,W,3])
cnt = np.ones([H,W,1])*1e-10
panorama_list = []
for img in tqdm(images[::4], 'processing'):
    # TODO: stitch img to panorama one by one
    # 2.2 align and average
    ...


    panorama = ...
    panorama_list.append(panorama)
    # show
    imshow('results/2_panorama.jpg', panorama)

write_frames_to_video('results/2.3_panorama_list.mp4', panorama_list, fps/2)

# panorama = algined.mean(0)
write_and_show('results/2.4_panorama.jpg', panorama)

destroyAllWindows();
