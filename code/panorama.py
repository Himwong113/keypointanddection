import cv2
import numpy as np
from scipy.stats import mode


def sliding_frame_panorama_with_mapping(video_path, output_path='sliding_panorama.jpg', frame_interval=5):
    """
    Creates a panorama from a video by sliding through frames at regular intervals
    and generates a mapping of which frame contributed each pixel.

    Parameters:
        video_path (str): Path to the input video.
        output_path (str): Path to save the generated panorama image.
        frame_interval (int): Number of frames to skip between each sample.

    Returns:
        tuple: Path to the saved panorama image and a NumPy array mapping panorama pixels to source frames.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file.")

    # Initialize variables
    frames = []
    frame_indices = []  # To keep track of the frame order
    frame_count = 0

    # Read frames at the specified interval
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)
            frame_indices.append(frame_count)  # Keep track of frame index

        frame_count += 1

    cap.release()

    print(f"Total frames processed: {len(frames)}")

    # Stitch frames into a panorama
    print("Stitching frames...")
    stitcher = cv2.Stitcher_create()  # Use cv2.createStitcher() for older OpenCV versions
    status, panorama = stitcher.stitch(frames)

    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"Error during stitching: {status}")

    # Initialize mapping array
    height, width, _ = panorama.shape
    pixel_mapping = np.zeros((height, width), dtype=np.int32)  # Initialize with zeros

    # Update pixel mapping
    for i, frame in enumerate(frames):
        # Resize frame to panorama size
        resized_frame = cv2.resize(frame, (width, height))

        # Generate a mask of contributing pixels
        mask = np.any(resized_frame > 0, axis=2)  # Non-black pixels contribute

        # Debug: Check unique values in the mask
        unique_mask_values = np.unique(mask)
        print(f"Frame {frame_indices[i]} Mask Unique Values: {unique_mask_values}")

        # Only update unmapped pixels
        pixel_mapping[mask & (pixel_mapping == 0)] = frame_indices[i]

        # Debug: Log contribution statistics
        contributing_pixels = np.sum(mask & (pixel_mapping == frame_indices[i]))
        print(f"Frame {frame_indices[i]} contributes to {contributing_pixels} pixels.")

    # Save the panorama image
    cv2.imwrite(output_path, panorama)
    print(f"Sliding panorama saved at {output_path}")

    return output_path, pixel_mapping


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

def match_frame_to_panorama(frame, panorama_path):
    """
    Matches the extracted frame to the panorama and calculates the contributing region.

    Parameters:
        frame (np.ndarray): The extracted frame to match.
        panorama_path (str): Path to the panorama image file.

    Returns:
        tuple: Dimensions of the contributing region (width, height).
    """
    panorama = cv2.imread(panorama_path)
    if panorama is None:
        raise ValueError("Failed to load panorama image.")

    # Resize the frame to match the panorama size
    panorama_height, panorama_width, _ = panorama.shape
    resized_frame = cv2.resize(frame, (panorama_width, panorama_height))

    # Create a mask to find the matching region
    mask = np.any(resized_frame > 0, axis=2)  # Non-black pixels in the resized frame
    region_width = np.sum(mask, axis=0).max()
    region_height = np.sum(mask, axis=1).max()

    print(f"Contributing region: Width={region_width}, Height={region_height}")
    return region_width, region_height



if __name__ == "__main__":
    # Define input video and output paths
    import os 
    video_path = os.path.join(os.getcwd(),'image','winter_day.mov')  # Replace with your video file path
    output_path = "output_panorama.jpg"

    # Set frame interval (optional)
    frame_interval = 50

    # Generate panorama and pixel mapping
    try:
        panorama_path, pixel_mapping = sliding_frame_panorama_with_mapping(video_path, output_path, frame_interval)

        # Save the pixel mapping as a NumPy file
        mapping_path = output_path.replace('.jpg', '_pixel_mapping.npy')
        np.save(mapping_path, pixel_mapping)
        print(f"Pixel mapping saved at {mapping_path}")

        # Step 1: Extract the frame
        frame_to_check = 100
        frame = extract_frame_from_video(video_path, frame_to_check)

        # Step 2: Match the frame to the panorama

        region_width, region_height = match_frame_to_panorama(frame, os.path.join(os.getcwd(),"output_panorama.jpg"))

        print(f"Frame {frame_to_check} contributes to a region with dimensions:")
        print(f"Width = {region_width}, Height = {region_height}")
 
        #print(f'mode of mapping : {mode(voting)}')
                          


    except Exception as e:
        print(f"Error: {e}")


