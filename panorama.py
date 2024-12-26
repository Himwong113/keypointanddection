import cv2
import numpy as np
from tqdm import tqdm
from sift import keypoint_match, draw_match, transform
from utils import read_video_frames, write_frames_to_video, write_and_show, destroyAllWindows, imshow
from PIL import Image

def crop_dark_part(img):

    # Convert image to numpy array
    img_array = np.array(img)

    # Convert the image to grayscale
    gray_img_array = np.mean(img_array, axis=2)

    # Define a threshold value to identify dark areas
    threshold = np.mean(gray_img_array) * 0.5

    # Create a mask where every dark part is set to False
    non_dark_mask = gray_img_array >= threshold

    # Find the bounding box around the non-dark area
    rows = np.any(non_dark_mask, axis=1)
    cols = np.any(non_dark_mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Crop the image using the bounding box
    cropped_img_array = img_array[row_min:row_max, col_min:col_max]

    # Convert array back to Image


    # Return the cropped image
    return cropped_img_array
def stictch(left,right):
    img1 = left
    img2 = right
    keypoints1, keypoints2, match = keypoint_match(img1, img2, max_n_match=1000)
    #draw_match(img1, keypoints1, img2, keypoints2,match,savename=base_dir_result+'winter_match.jpg')

    keypoints1 = np.array([keypoints1[m.queryIdx].pt for m in match])
    keypoints2 = np.array([keypoints2[m.trainIdx].pt for m in match])
    #print(img1.shape)
    H, W = img2.shape[:2]
    W = W +1280
    #print(img2.shape)
    #cv2.imshow(base_dir_result + 'right.jpg', img2)
    new_img2 = transform(img2, keypoints2, keypoints1, H, W)


    cv2.imshow(base_dir_result + 'wintertransformed_right.jpg', img2)
    write_and_show(base_dir_result+'trans.jpg',new_img2)
    # resize img1
    new_img1 = np.concatenate((img1,np.zeros((720,W-1280,3))),axis=1)

    #print(new_img1.shape)
    #cv2.imshow(base_dir_result + 'wintertransformed_left.jpg', new_img1)

    direct_mean = new_img1 / 2 + new_img2 / 2
    #cv2.imshow(base_dir_result + 'wintertransformed_mean.jpg', direct_mean)
    cnt = np.zeros([H, W, 1]) + 1e-10  # add a tiny value to avoid ZeroDivisionError
    cnt += (new_img2 != 0).any(2, keepdims=True)  # any: or
    cnt += (new_img1 != 0).any(2, keepdims=True)

    new_img1 = np.float32(new_img1)
    new_img2 = np.float32(new_img2)
    stack = (new_img2 + new_img1) / cnt
    write_and_show(base_dir_result + 'new_img1.jpg', new_img1)
    write_and_show(base_dir_result + 'new_img2.jpg', new_img2)
    #cv2.imshow(base_dir_result + 'wintertransformed_stack.jpg', stack)
    #print(stack.shape)





    return stack


if __name__  == "__main__":
    base_dir = 'C:\\Users\\ivanw\\2.KeypointsDetectionAndMatch\\pythonProject\\image\\'
    base_dir_result = 'C:\\Users\\ivanw\\2.KeypointsDetectionAndMatch\\pythonProject\\results\\'

    video_name = base_dir+ 'winter_day.mov'
    images, fps = read_video_frames(video_name)
    n_image = len(images)
    #print(n_image)
    #print(fps)
    # TODO: init panorama
    # 2.1
    h, w = images[0].shape[:2]
    H , W = h,34000
    panorama = np.zeros([H,W,3]) # use a large canvas

    h_start = h
    w_start = W-w
    #print(images[0].shape)
    #print(panorama.shape)
    panorama[:, w_start:w_start+w, :] = images[0]

    trans_sum = np.zeros([H,W,3])
    cnt = np.ones([H,W,1])*1e-10
    panorama_list = []


    step = 26
    stack_image= images[0]
    for idx, img in enumerate( tqdm(images[::step], 'processing')):
        # TODO: stitch img to panorama one by one
        # 2.2 align and average

        print(f'idx ={idx}')
        next = min((idx+1)*step,n_image-1)
        print(next)
        img_left  = images[next]
        img_right = stack_image




        stack_image = stictch(left=img_left,right=img_right)
        imshow('stack_image.jpg', stack_image)

        #print(stack_image)
        #panorama[:,0:w*next,:]=stack_image
        #print(panorama.shape)

        # show
        #trans_sum += stack_image
        #cnt += (stack_image != 0).any(2, keepdims=True)
        #panorama = trans_sum / cnt

        panorama_list.append(stack_image)
        imshow('panorama.jpg', stack_image)

        if idx ==12 :break


    # post processing for write to frame
    for idx in range (len(panorama_list)):
        crop =crop_dark_part(panorama_list[idx]) # cut the paddding of image
        max =crop_dark_part(panorama_list[-1])
        H, W, _ = max.shape
        h,w,C = crop.shape
        print(h,w,C)
        w=int(W-w)
        temp = np.concatenate(( crop , np.zeros((h,w,3))),axis=1) # fill the zero as padding
        temp = cv2. resize(temp,(2048,719)) # mp3 only support 2048 x 2048(panorma size is too big to export) resize by cv2 lib
        print(f'temp ={temp.shape}')
        panorama_list[idx] =  temp




    stack_image = np.float32(stack_image)
    write_and_show(base_dir_result + '2.4_panorama.jpg', crop_dark_part(stack_image))
    write_frames_to_video(base_dir_result+'2.3_panorama_list.mp4',  panorama_list, 1)


    # panorama = algined.mean(0)


    destroyAllWindows();
