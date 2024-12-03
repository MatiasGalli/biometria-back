import cv2
import numpy as np

# Función para alinear la imagen basada en puntos SIFT
def align_carnet(image, reference):
    # Convertir las imágenes a escala de grises
    gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Crear objeto SIFT
    sift = cv2.SIFT_create()

    # Detectar puntos clave y descriptores
    keypoints_original, descriptors_original = sift.detectAndCompute(gray_original, None)
    keypoints_reference, descriptors_reference = sift.detectAndCompute(gray_reference, None)

    # Usar FLANN para encontrar coincidencias
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_original, descriptors_reference, k=2)

    # Almacenar las buenas coincidencias según el ratio de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Verificar que se encontraron suficientes coincidencias
    if len(good_matches) > 10:
        # Obtener las coordenadas de los puntos clave
        src_pts = np.float32([keypoints_original[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_reference[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Calcular la transformación de perspectiva (Homografía)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Aplicar la transformación de perspectiva a la imagen original
        h, w = reference.shape[:2]
        transformed_image = cv2.warpPerspective(image, M, (w, h))

        return transformed_image
    else:
        return None