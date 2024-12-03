import cv2
from pyzbar.pyzbar import decode
import re

# Función para detectar el QR en la imagen
def detect_qr(image):
    # Verificación de la carga de imagen
    if image is None:
        print("Error: La imagen no se ha cargado correctamente.", flush=True)
        return None

    imagen_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imagen_gris = cv2.GaussianBlur(imagen_gris, (5, 5), 0)
    imagen_procesada = cv2.adaptiveThreshold(
        imagen_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    codigos_qr = decode(imagen_procesada)

    if not codigos_qr:
        print("No se encontraron códigos QR en la imagen.", flush=True)
        return None

    # Paso 5: Verificación del contenido del QR
    for codigo in codigos_qr:
        data = codigo.data.decode('utf-8')
        print("Datos del QR detectado:", data, flush=True)
        if es_url(data):
            print("El QR contiene una URL válida:", data, flush=True)
            return data
        else:
            print("El QR no contiene una URL válida.", flush=True)

    return None

# Función para verificar si una cadena es una URL
def es_url(cadena):
    patron_url = re.compile(
        r'^(http|https)://[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)+(:[0-9]+)?(/.*)?$'
    )
    return re.match(patron_url, cadena) is not None