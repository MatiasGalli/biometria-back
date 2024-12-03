import cv2
import numpy as np
import os
import hashlib
import time

def load_image(image_path):
    """
    Load an image from the specified path using OpenCV.
    """
    return cv2.imread(image_path)

def read_image(file):
    """
    Read an image from a file without resizing it.
    """
    image_cv2 = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(image_cv2, cv2.IMREAD_COLOR)

def save_image_temp(image, folder='./tmp'):
    """
    Save an image temporarily in the specified folder with a unique name based on a hash.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = str(time.time()).encode('utf-8')
    file_hash = hashlib.md5(timestamp).hexdigest()
    temp_image_path = os.path.join(folder, f'normalized_image_{file_hash}.jpg')

    cv2.imwrite(temp_image_path, image)
    return temp_image_path

def delete_temp_file(file_path):
    """
    Delete the specified temporary file.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

def preprocesar_segmento(image):
    """
    Apply grayscale conversion and adaptive thresholding to enhance the image for OCR.
    """
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # suavizado = cv2.GaussianBlur(gris, (3, 3), 0)
    _, binarizado = cv2.threshold(gris, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # kernel = np.ones((1, 1), np.uint8)
    # erosionado = cv2.erode(binarizado, kernel, iterations=1)
    # dilatado = cv2.dilate(erosionado, kernel, iterations=1)
    return binarizado
