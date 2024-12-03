from flask import Blueprint, request, jsonify
from services import validate
from utils.logger import logger

# Contadores globales
validation_success = 0
validation_failure = 0

main = Blueprint('validate_blueprint', __name__)

@main.route('/', methods=['POST'])
def validate_data():
    global validation_success, validation_failure
    try:
        data = request.get_json()

        # Extraer los datos de la solicitud
        front_info = data.get('front_data', {})
        back_info = data.get('back_data', {})
        ruta_img_1 = data.get('img_1_route')
        ruta_img_2 = data.get('img_2_route')
        qr_data = data.get('qr')

        # Llamar al servicio de validación
        validation_results = validate.validate_data(front_info, back_info, ruta_img_1, ruta_img_2, qr_data)

        # Detectar si hubo fallos
        failed_checks = [key for key, value in validation_results.items() if not value]

        if failed_checks:
            # Incrementar el contador de fallos y registrar los detalles
            validation_failure += 1
            logger.warning(
                f"Validation failed. Failures in: {failed_checks}. "
                f"Total Success: {validation_success}, Total Failures: {validation_failure}"
            )
        else:
            # Incrementar el contador de éxitos y registrar los detalles
            validation_success += 1
            logger.info(
                f"Validation successful. Total Success: {validation_success}, Total Failures: {validation_failure}"
            )

        # Retornar la misma estructura de JSON que tienes
        return jsonify(validation_results), 200
    except Exception as e:
        validation_failure += 1
        logger.error(f"An error occurred in validation: {str(e)}")
        return jsonify({"error": "An error occurred", "message": str(e)}), 500