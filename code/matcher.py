import cv2
import numpy as np
import os
from scipy.stats import mode

def extract_frame_from_video(video_path, frame_to_check):
    """
    Extracts a specific frame from a video.

    Parameters:
        video_path (str): Path to the video file.
        frame_to_check (int): Index of the frame to extract.

    Returns:
        np.ndarray: The extracted frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file.")
    
    frame_index = 0
    target_frame = None

    while frame_index <= frame_to_check:
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Frame {frame_to_check} not found in the video.")
        if frame_index == frame_to_check:
            target_frame = frame
            break
        frame_index += 1

    cap.release()

    if target_frame is None:
        raise ValueError("Failed to extract the frame.")
    
    return target_frame



def locate_image_in_panorama(image_path, panorama_path):
    """
    Finds the position of an input image in a panorama and returns its corner coordinates.
    
    Parameters:
        image_path (str): Path to the input image.
        panorama_path (str): Path to the panorama image.
    
    Returns:
        dict: A dictionary with the corner coordinates (UL, UR, BL, BR).
    """
    # Load the images
    img = cv2.imread(image_path)
    pano = cv2.imread(panorama_path)

    if img is None or pano is None:
        raise ValueError("Failed to load one or both images. Check the file paths.")

    # Convert to grayscale for feature matching
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pano_gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(pano_gray, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Minimum number of matches required
    if len(matches) < 4:
        raise ValueError("Not enough matches found to locate the image in the panorama.")

    # Extract location of good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography matrix
    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if matrix is None:
        raise ValueError("Failed to compute homography.")

    # Get the dimensions of the input image
    h, w = img.shape[:2]

    # Define the corners of the input image
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    # Transform the corners to panorama space
    transformed_corners = cv2.perspectiveTransform(corners, matrix)

    # Extract the corner coordinates
    UL = tuple(map(int, transformed_corners[0][0]))
    UR = tuple(map(int, transformed_corners[1][0]))
    BL = tuple(map(int, transformed_corners[2][0]))
    BR = tuple(map(int, transformed_corners[3][0]))

    corners_dict = {"UL": UL, "UR": UR, "BL": BL, "BR": BR}

    print(f"Corners in panorama space: {corners_dict}")
    return corners_dict

def crop_region_from_panorama(panorama_path, corners):
    """
    Crops a region from the panorama based on the provided corner coordinates.
    
    Parameters:
        panorama_path (str): Path to the panorama image.
        corners (dict): Dictionary with corner coordinates (UL, UR, BL, BR).
    
    Returns:
        np.ndarray: The cropped region from the panorama.
    """
    # Load the panorama
    panorama = cv2.imread(panorama_path)
    if panorama is None:
        raise ValueError("Failed to load panorama image.")

    pano_h, pano_w = panorama.shape[:2]

    # Extract corners
    UL = corners["UL"]
    UR = corners["UR"]
    BL = corners["BL"]
    BR = corners["BR"]

    # Ensure coordinates are within the bounds of the panorama
    def clamp(coord, max_x, max_y):
        return (max(0, min(coord[0], max_x - 1)), max(0, min(coord[1], max_y - 1)))

    UL = clamp(UL, pano_w, pano_h)
    UR = clamp(UR, pano_w, pano_h)
    BL = clamp(BL, pano_w, pano_h)
    BR = clamp(BR, pano_w, pano_h)

    # Define the bounding rectangle
    x_min = min(UL[0], BL[0])
    x_max = max(UR[0], BR[0])
    y_min = min(UL[1], UR[1])
    y_max = max(BL[1], BR[1])

    # Crop the region
    cropped_crood= [y_min,y_max, x_min,x_max]
    cropped_region = panorama[y_min:y_max, x_min:x_max]

    print(f"Cropped region coordinates: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
    return cropped_region ,cropped_crood

if __name__ == "__main__":
    # Paths to the input image and panorama
    video_path = os.path.join(os.getcwd(),'image','winter_day.mov')
    target_frame = extract_frame_from_video(video_path=video_path, frame_to_check=10)
    cv2.imwrite("target.jpg",target_frame)

    image_path = "target.jpg"
    panorama_path = "output_panorama.jpg"

    # Locate the image in the panorama
    try:
        corners = locate_image_in_panorama(image_path, panorama_path)
        print("Corner coordinates in panorama:")
        
        for corner, coord in corners.items():
            print(f"{corner}: {coord}")
        
        cropped_region ,cropped_crood = crop_region_from_panorama(panorama_path, corners)

        panorama = cv2.imread(panorama_path)
        pano_h, pano_w = panorama.shape[:2]
        mapping = np.load(os.path.join(os.getcwd(),"output_panorama_pixel_mapping.npy"))
        mapping = mapping.reshape(pano_h, pano_w )

        y_min,y_max, x_min,x_max = cropped_crood
        print(f'vote map size :{mapping.shape} ')
        voting = mode(mapping[y_min:y_max,x_min:x_max].flatten())
#
        print(f'voting results = {voting.mode}')
        ## Display the cropped region
        #cv2.imshow("Cropped Region", cropped_region)
        ## Save the cropped region if needed
        cv2.imwrite("cropped_region.jpg", cropped_region)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
##
        
#
    except Exception as e:
        print(f"Error: {e}")
