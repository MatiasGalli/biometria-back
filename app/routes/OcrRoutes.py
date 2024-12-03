from flask import Blueprint, jsonify, request
from services import normalize, face_cropper, front_ocr, back_ocr, back_normalize, detect_qr
from utils import image_utils, path_utils
import os

main = Blueprint('ocr_blueprint', __name__)

@main.route('/front', methods=['POST'])
def front_ocr_route():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        # Leer la imagen subida sin redimensionar
        file = request.files['image']
        image_cv2 = image_utils.read_image(file)

        # Obtener las imágenes de referencia cacheadas
        reference_images = path_utils.get_front_reference_images()

        # Alinear la imagen con las imágenes de referencia
        normalized_image = back_normalize.align_image_with_references(image_cv2, reference_images)

        if normalized_image is None:
            return jsonify({"error": "Alignment failed"}), 400

        # Procesar OCR completo usando el servicio OCR
        resultado_ocr = front_ocr.procesar_ocr_completo(normalized_image)

        # Guardar la imagen temporalmente en la carpeta tmp
        temp_image_path = image_utils.save_image_temp(normalized_image)

        # Recortar las caras y guardar las imágenes
        face1, face2, face1_temp_path, face2_temp_path = face_cropper.crop_faces(normalized_image)

        return jsonify({
            "text": resultado_ocr,
            "temp_image_path": temp_image_path,
            "face1_temp_path": face1_temp_path,
            "face2_temp_path": face2_temp_path
        }), 200

    except Exception as e:
        return jsonify({"error": "An error occurred", "message": str(e)}), 500


@main.route('/back', methods=['POST'])
def back_ocr_route():
    try:
        # Verificación de carga de imagen
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        # Leer la imagen subida sin redimensionar
        file = request.files['image']
        image_cv2 = image_utils.read_image(file)

        # Obtener las imágenes de referencia cacheadas
        reference_images = path_utils.get_back_reference_images()

        # Alinear la imagen con las imágenes de referencia
        normalized_image = back_normalize.align_image_with_references(image_cv2, reference_images)

        if normalized_image is None:
            return jsonify({"error": "Alignment failed"}), 400

        # Procesar OCR en la imagen alineada
        resultado_ocr = back_ocr.procesar_ocr_reverso(normalized_image)
        qr_value = detect_qr.detect_qr(normalized_image)


        # Guardar la imagen alineada temporalmente
        temp_image_path = image_utils.save_image_temp(normalized_image)

        return jsonify({
            "qr": qr_value,
            "text": resultado_ocr,
            "temp_image_path": temp_image_path
        }), 200

    except Exception as e:
        return jsonify({"error": "An error occurred", "message": str(e)}), 500
