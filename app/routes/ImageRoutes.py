from flask import Blueprint, send_from_directory
import os

main = Blueprint('image_blueprint', __name__)


@main.route('/<filename>')
def get_image(filename):
    # Aquí usamos una ruta absoluta a la carpeta static en la raíz del proyecto
    static_folder = os.path.join(os.path.dirname(__file__), '../../static')
    return send_from_directory(static_folder, filename)