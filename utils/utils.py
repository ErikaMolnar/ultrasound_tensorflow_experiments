import os
import cv2
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import load_img, img_to_array, save_img, ImageDataGenerator


# Function to split data and copy files
def split_data(category, original_dataset_dir, train_dir, validation_dir, test_dir):
    files = os.listdir(os.path.join(original_dataset_dir, category))
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    
    # Create category directories inside train, val, and test
    train_category_dir = os.path.join(train_dir, category)
    val_category_dir = os.path.join(validation_dir, category)
    test_category_dir = os.path.join(test_dir, category)
    
    os.makedirs(train_category_dir, exist_ok=True)
    os.makedirs(val_category_dir, exist_ok=True)
    os.makedirs(test_category_dir, exist_ok=True)
    
    # Function to copy files
    def copy_files(files, dest_dir):
        for fname in files:
            src = os.path.join(original_dataset_dir, category, fname)
            dst = os.path.join(dest_dir, fname)
            shutil.copyfile(src, dst)
    
    # Copy files
    copy_files(train_files, train_category_dir)
    copy_files(val_files, val_category_dir)
    copy_files(test_files, test_category_dir)


# Function to get image paths
def get_image_paths(category_path, oversample=False):
    if oversample:
        # Path to the oversampled images
        oversample_path = os.path.join(category_path, 'oversampled')
        if os.path.exists(oversample_path):
            return [os.path.join(oversample_path, img) for img in os.listdir(oversample_path) if img.endswith('.png')]
    # Path to original images
    return [os.path.join(category_path, img) for img in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, img)) and img.endswith('.png')]


def visualize_images(image_paths, num_images=5):
    plt.figure(figsize=(20, 4))
    
    for i, image_path in enumerate(image_paths[:num_images]):
        # Load the image
        img = load_img(image_path, target_size=(224, 224))
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.title(f"Image: {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# Function to oversample a category
def oversample_category(category_path, desired_size, datagen):
    image_paths = [os.path.join(category_path, fname) for fname in os.listdir(category_path)]
    n_samples_needed = desired_size - len(image_paths)
    
    # If no oversampling is needed, just return the original paths
    if n_samples_needed <= 0:
        return image_paths

    # Resample the images with replacement
    oversampled_images = resample(image_paths, replace=True, n_samples=n_samples_needed, random_state=0)

    # Create a temporary directory within the category directory to store oversampled images
    temp_dir = os.path.join(category_path, 'oversampled')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Apply transformations to the oversampled images using the data generator and return paths
    oversampled_transformed = []
    for image_path in oversampled_images:
        if not image_path.lower().endswith('.png'):
            continue  # Skip non-png files
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = datagen.random_transform(image)
        # Save the transformed image with a unique filename to avoid collisions
        temp_filename = f"oversampled_{os.path.basename(image_path)}"
        temp_path = os.path.join(temp_dir, temp_filename)

        save_img(temp_path, image)
        oversampled_transformed.append(temp_path)

    return oversampled_transformed + image_paths


def mask_images(img_dir):
    # Define colors for the masks
    colors = {
        'green': (0, 255, 0),
        'red': (255, 0, 0),
        'blue': (0, 0, 255)
    }

    # Remove previously masked images
    for filename in os.listdir(img_dir):
        if filename.endswith("_masked.png"):
            os.remove(os.path.join(img_dir, filename))
    
    # Get all image filenames that need masking
    image_filenames = [f for f in os.listdir(img_dir) if f.lower().endswith('.png') 
                       and not f.endswith('_mask.png') 
                       and not f.endswith('_mask_1.png') 
                       and not f.endswith('_mask_2.png')]

    for image_filename in image_filenames:
        image_path = os.path.join(img_dir, image_filename)
        mask_filename = os.path.splitext(image_filename)[0] + '_mask.png'
        mask_path = os.path.join(img_dir, mask_filename)

        # Load the original image in color (RGB)
        image = load_img(image_path)
        image_array = img_to_array(image) / 255.0

        # Load the mask in grayscale, convert it to a binary mask, and then invert it
        mask = load_img(mask_path, color_mode='grayscale')
        mask_array = img_to_array(mask) / 255.0
        mask_array = 1 - (mask_array > 0.5)

        # Choose a random color for the mask
        color_value = random.choice(list(colors.items()))

        # Create a colored mask
        colored_mask = np.zeros_like(image_array)
        for i in range(3):
            colored_mask[:, :, i] = color_value[i] / 255.0

        masked_image = np.where(mask_array, colored_mask, image_array)

        # Save the masked image
        masked_image_path = os.path.join(img_dir, os.path.splitext(image_filename)[0] + f'_masked.png')
        cv2.imwrite(masked_image_path, masked_image * 255)  # Multiply by 255 to convert back to 0-255 scale
