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
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    # TODO: detect keypoints and generate descriptor by by 'orb.detectAndCompute', modify parameters for cv2.ORB_create for more stable results.
    orb = cv2.ORB_create( nfeatures=5000)




    keypoints1, descriptors1 =  orb.detectAndCompute(img1_gray,None)
    keypoints2, descriptors2 =  orb.detectAndCompute(img2_gray,None)


    # TODO: convert descriptors1, descriptors2 to np.float32
    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)
    print(f'descriptors1={descriptors1}')
    print(f'descriptors2 ={descriptors2}')
    # TODO: Knn match and Lowe's ratio test

    matcher = cv2.FlannBasedMatcher()
    best_2 = matcher.knnMatch(
        queryDescriptors=descriptors1,
        trainDescriptors=descriptors2,
        k=2)
    # Lowe's ratio test
    ratio = 0.7
    match = []
    for m, n in best_2:
        if m.distance < ratio * n.distance:
            match.append(m)

    # TODO: select best `max_n_match` matches
    # sort by distance
    match = sorted(match, key=lambda x: x.distance)


    # TODO: select best `max_n_match` matches
    # take the best 100 matches
    match = match[:100]
    #result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, match, img2_gray, flags=2)
    #plt.rcParams['figure.figsize'] = [14.0, 7.0]
    #plt.title('Best Matching Points')
    #plt.imshow(result)
    #plt.show()
    return keypoints1, keypoints2, match

def convertkps(kps):
    temp = []
    for kps_list in [kps]:
        keypoints_attributes = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kps_list]

        # If you need to use or print the attributes
        for attr in keypoints_attributes:
            temp.append(list(attr[0]))
            # Now 'attr' holds the attributes of each keypoint as a tuple
    return temp

if __name__ == '__main__':
    # read in video
    video_name = 'C:\\Users\\ivanw\\2.KeypointsDetectionAndMatch\\pythonProject\\image\\rain2.mov'
    images, fps = read_video_frames(video_name)
    images = np.asarray(images)

    # get stabilized frames
    stabilized = []
    reference = images[0]
    trans =images[0]
    H, W = reference.shape[:2]

    for img in tqdm(images[::2], 'processing'):

        ## TODO find keypoints and matches between each input img and the reference image
        ref_kps, img_kps, match = orb_keypoint_match(reference ,img)
        keypoints = [ref_kps,img_kps]  # A list of cv2.KeyPoint objects

        # Process keypoints for reference and image separately

        draw_match(reference,ref_kps,img,img_kps,match,savename='keypoiny.jpg')


        # TODO: align all frames to reference frame (images[0])
        #reference =np.float32(reference)
        ref_kps= np.array([ref_kps[m.queryIdx].pt for m in match])
        img_kps= np.array([img_kps[m.trainIdx].pt for m in match])

        print(f'ref_kps = {ref_kps.shape}')
        print(f'img kps ={img_kps.shape}')
        if len(match)>4:
            new_img2 = transform(img,img_kps,ref_kps,H, W)

            #print(f'img={img.shape},newimg1{new_img1.shape},newmg2{new_img2.shape}')
            cnt = np.zeros([H, W, 1]) + 1e-10  # add a tiny value to avoid ZeroDivisionError
            cnt += (new_img2 != 0).any(2, keepdims=True)  # any: or

            print('processing')

            new_img2 = np.float32(new_img2)
            trans = (new_img2 ) / cnt

            stabilized.append(trans)
        else:
            trans=reference
            stabilized.append(trans)






        imshow('trans.jpg', trans)








    # write stabilized frames to a video
    base_dir_result = 'C:\\Users\\ivanw\\2.KeypointsDetectionAndMatch\\pythonProject\\results\\'
    write_frames_to_video(base_dir_result+'3.2_stabilized.mp4', stabilized, fps/4)

    # get rain free images

    stabilized_mean = np.mean(stabilized, 0)
    write_and_show(base_dir_result+'3.3_stabilized_mean.jpg', stabilized_mean)


    destroyAllWindows()
