from PIL import Image
import numpy as np

def remove_background_advanced(image):
    # Convert the image to RGBA mode
    image = image.convert("RGBA")

    # Convert PIL image to numpy array
    data = np.array(image)

    # Set threshold for black pixels
    lower_black = np.array([0, 0, 0, 255])
    upper_black = np.array([10, 10, 10, 255])

    # Create a mask to identify black pixels
    mask = np.all(data[:, :, :3] <= upper_black[:3], axis=-1) & np.all(data[:, :, :3] >= lower_black[:3], axis=-1)

    # Apply the mask to remove black background
    data[mask] = [0, 0, 0, 0]  # Set black pixels to transparent

    # Convert the numpy array back to a PIL image
    image = Image.fromarray(data, 'RGBA')
    return image

def merge_images(background_path, foreground_path, output_path):
    # Open the background and foreground images
    background = Image.open(background_path)
    foreground = Image.open(foreground_path)

    # Remove black background from the foreground image
    foreground = remove_background_advanced(foreground)

    # Convert images to RGBA mode if they're not already
    background = background.convert('RGBA')
    foreground = foreground.convert('RGBA')

    # Resize the foreground image to match the background size
    foreground = foreground.resize(background.size)

    # Merge the images by pasting the foreground onto the background
    merged_image = Image.alpha_composite(background, foreground)

    # Save the merged image
    merged_image.save(output_path)

# Paths to your background and foreground PNG images
background_image_path = r'D:\Neural-Style-Transfer\Images\background.png'
foreground_image_path = r'D:\Neural-Style-Transfer\Images\subject.png'

# Path to save the output merged image
output_image_path = r'D:\Neural-Style-Transfer\Images\output_merged_image.png'

# Merge the images with specified paths
merge_images(background_image_path, foreground_image_path, output_image_path)
