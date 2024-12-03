from functools import lru_cache
import os
import cv2

@lru_cache(maxsize=1)
def get_front_reference_images():
    """
    Load and cache back reference images as numpy arrays for alignment.
    """
    reference_folder = './public/front_references'
    reference_images = []

    for filename in os.listdir(reference_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(reference_folder, filename)
            image = cv2.imread(image_path)
            if image is not None:
                reference_images.append(image)

    return reference_images

@lru_cache(maxsize=1)
def get_back_reference_images():
    """
    Load and cache back reference images as numpy arrays for alignment.
    """
    reference_folder = './public/back_references'
    reference_images = []

    for filename in os.listdir(reference_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(reference_folder, filename)
            image = cv2.imread(image_path)
            if image is not None:
                reference_images.append(image)

    return reference_images
