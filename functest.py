from PIL import Image
import numpy as np

def crop_dark_part(image_path):
    # Load the image
    img = Image.open(image_path)

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
    cropped_img = Image.fromarray(cropped_img_array)

    # Return the cropped image
    return cropped_img

# Usage
image_path = 'C:\\Users\\ivanw\\2.KeypointsDetectionAndMatch\\pythonProject\\results\\2.4_panorama.jpg'  # Replace with your image path
cropped_image = crop_dark_part(image_path)
cropped_image_path = 'path_to_save_cropped_image.jpg'  # Replace with your desired path
cropped_image.save(cropped_image_path)
