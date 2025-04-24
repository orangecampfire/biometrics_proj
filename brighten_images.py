import cv2
import os

# Directory where the caltech images are
img_directory = r"C:/Users/david/Documents/USF/USF/Classes/Spring_25/Biometrics/Proj/caltech_copy/"
#Directory where you want the brightened images to go
output_directory = r"C:/Users/david/Documents/USF/USF/Classes/Spring_25/Biometrics/Proj/caltech_brighter/"

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through each image in the directory
for root, dirs, files in os.walk(img_directory):
    for image_name in files:
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(root, image_name)
            image = cv2.imread(image_path)

            if image is not None:
                brightness_factor = 1.5
                image_adjusted = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

                # Maintain folder structure in output directory
                relative_path = os.path.relpath(root, img_directory)
                save_folder = os.path.join(output_directory, relative_path)
                os.makedirs(save_folder, exist_ok=True)

                output_image_path = os.path.join(save_folder, image_name)
                cv2.imwrite(output_image_path, image_adjusted)
                print(f"Saved brightened image: {output_image_path}")
            else:
                print(f"Failed to load image: {image_name}")
