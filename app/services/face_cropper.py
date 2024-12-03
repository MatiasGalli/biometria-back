import cv2
import numpy as np
import os
import time
import hashlib

def crop_faces(image):
    # Definir las coordenadas de recorte para las dos fotos
    height, width = image.shape[:2]

    # Coordenadas para la cara grande (x, y, ancho, alto)
    # Posición inicial 42 X(px), 130 Y(px)
    face1_coords = (30, 134, int(width * 0.25), int(height * 0.55))  # Ajusta el ancho y alto según necesidad

    # Coordenadas para la cara pequeña (ajusta según la posición deseada)
    face2_coords = (705, 215, int(width * 0.075), int(height * 0.15))  # Ajusta según necesidad

    # Recortar las imágenes
    face1 = image[face1_coords[1]:face1_coords[1] + face1_coords[3],
                   face1_coords[0]:face1_coords[0] + face1_coords[2]]

    face2 = image[face2_coords[1]:face2_coords[1] + face2_coords[3], 
                   face2_coords[0]:face2_coords[0] + face2_coords[2]]

    # Guardar las imágenes de las caras en la carpeta temporal
    face1_path = save_faces_temp(face1, "face1")
    face2_path = save_faces_temp(face2, "face2")

    return face1, face2, face1_path, face2_path

def save_faces_temp(image, prefix, folder='./static'):
    """
    Save a face image temporarily in the specified folder with a unique name based on a hash.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = str(time.time()).encode('utf-8')
    file_hash = hashlib.md5(timestamp).hexdigest()
    temp_image_path = os.path.join(folder, f'{prefix}_image_{file_hash}.jpg')

    cv2.imwrite(temp_image_path, image)
    return temp_image_path
