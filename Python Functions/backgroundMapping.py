
from rembg import remove
from PIL import Image
import numpy as np

input_path = r'D:\Neural-Style-Transfer\Images\dancing.jpg'
output_path = r'D:\Neural-Style-Transfer\Images\output.png'
background_path = r'D:\Neural-Style-Transfer\Images\background.png'
subject_path = r'D:\Neural-Style-Transfer\Images\subject.png'

# Load input image
input_image = Image.open(input_path)
print("Input image loaded.")

# Remove background
output_image = remove(input_image)
print("Background removed.")

# Convert images to NumPy arrays
input_rgb = np.array(input_image)[:, :, :3]  # Extract RGB channels.
output_rgba = np.array(output_image)  # Output image in RGBA.

# Extract alpha channel
alpha = output_rgba[:, :, 3]
alpha3 = np.dstack((alpha, alpha, alpha))  # Convert to 3 channels

# Calculate background without subject
background_rgb = input_rgb.astype(np.float64) * (1 - alpha3.astype(np.float64) / 255)
background_rgb = background_rgb.astype(np.uint8)  # Convert back to uint8

# Convert background to PIL image
background = Image.fromarray(background_rgb)
print("Computed background without subject.")

# Extract and save subject (main object)
subject_rgb = input_rgb * (alpha3 / 255)
subject_rgb = subject_rgb.astype(np.uint8)  # Convert to uint8

# Convert subject to PIL image
subject = Image.fromarray(subject_rgb)
print("Extracted subject.")

# Save output images
output_image.save(output_path)
background.save(background_path)
subject.save(subject_path)
print("Output images saved.")
