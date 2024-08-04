from datasets import load_dataset
from PIL import Image
import os
from pyprojroot import here
# Load your dataset
dataset = load_dataset('naver-clova-ix/cord-v2', split='validation')  # Adjust dataset loading as needed

# Directory to save images
save_dir = here("data")
os.makedirs(save_dir, exist_ok=True)

# Number of images to save
num_images_to_save = 20

for i in range(num_images_to_save):
    # Load image from dataset
    image = dataset[i]['image']
    
    # If the image is a PIL Image object
    if isinstance(image, Image.Image):
        # Save image to local directory
        image_path = os.path.join(save_dir, f'image_{i}.png')
        image.save(image_path)
    else:
        # If the image is not already a PIL Image, convert it
        image = Image.fromarray(image)
        image_path = os.path.join(save_dir, f'image_{i}.png')
        image.save(image_path)

    print(f"Saved image {i} to {image_path}")

print("All images have been saved.")
