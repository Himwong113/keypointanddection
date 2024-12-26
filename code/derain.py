import cv2
# from os import path
import os
import numpy as np
from tqdm import tqdm
from sift import draw_match, transform
from utils import (
    imshow, imread,
    write_and_show, destroyAllWindows,
    read_video_frames, write_frames_to_video
)

def orb_keypoint_match(img1, img2, max_n_match=100, draw=True):
    # make sure they are of dtype uint8
    img1 = np.uint8(img1)
    img2 = np.uint8(img2)

    # TODO: convert to grayscale by `cv2.cvtColor`
    img1_gray = ...
    img2_gray = ...


    # TODO: detect keypoints and generate descriptor by by 'orb.detectAndCompute', modify parameters for cv2.ORB_create for more stable results.
    orb = cv2.ORB_create(...)
    keypoints1, descriptors1 = ...
    keypoints2, descriptors2 = ...

    # TODO: convert descriptors1, descriptors2 to np.float32


    # TODO: Knn match and Lowe's ratio test
    matcher = cv2.FlannBasedMatcher_create()
    ...

    # TODO: select best `max_n_match` matches
    ...

    return keypoints1, keypoints2, match


if __name__ == '__main__':
    # read in video
    video_name = 'image/rain2.mov'
    images, fps = read_video_frames(video_name)
    images = np.asarray(images)

    # get stabilized frames
    stabilized = []
    reference = images[0]
    H, W = reference.shape[:2]
    for img in tqdm(images[::2], 'processing'):
        ## TODO find keypoints and matches between each input img and the reference image
        ref_kps, img_kps, match = orb_keypoint_match(...)


        # TODO: align all frames to reference frame (images[0])
        trans = ...



        stabilized.append(trans)
        imshow('trans.jpg', trans)

    # write stabilized frames to a video
    write_frames_to_video('results/3.2_stabilized.mp4', stabilized, fps/2)

    # get rain free images
    stabilized_mean = np.mean(stabilized, 0)
    write_and_show('results/3.3_stabilized_mean.jpg', stabilized_mean)

    destroyAllWindows()
