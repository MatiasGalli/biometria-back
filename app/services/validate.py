import cv2
import re
from unidecode import unidecode
from difflib import SequenceMatcher
from services.face_compare import compare_faces, model, device, transform
from utils import image_utils

def validate_data(front_data, back_data, ruta_img_1, ruta_img_2, qr, threshold=0.8):
    results = {
        "face": validate_face(ruta_img_1, ruta_img_2),
        "rut": validate_rut(front_data, back_data, threshold),
        "doc_id": validate_doc_id(front_data, back_data, threshold),
        "names": validate_names_partial(front_data, back_data, threshold),
        "dates": validate_dates(front_data, back_data, threshold),
        "qr": validate_qr(qr, front_data, back_data, threshold)  # Implementación pendiente
    }
    return results

def validate_face(ruta_img_1, ruta_img_2):
    # Cargar las imágenes desde las rutas de archivo
    img1 = cv2.imread(ruta_img_1)
    img2 = cv2.imread(ruta_img_2)

    if img1 is None or img2 is None:
        return False  # Fallo al cargar una o ambas imágenes

    distance = compare_faces(ruta_img_1, ruta_img_2, model, transform, device)
    return distance < 1  # Umbral de distancia

def validate_rut(front_data, back_data, threshold):
    run_front = f"{front_data['RUN']}"
    run_back = back_data.get("rut", "")
    similarity = calculate_similarity(run_front, run_back)
    print("RUN Similarity:", similarity)
    return similarity >= threshold

def validate_doc_id(front_data, back_data, threshold):
    doc_id_front = front_data.get("numero_documento", "")
    doc_id_back = back_data.get("numeroDocumento_MRZ", "")
    similarity = calculate_similarity(doc_id_front, doc_id_back)
    print("Document ID Similarity:", similarity)
    return similarity >= threshold

def validate_dates(front_data, back_data, threshold):
    # Convertir fechas de nacimiento y vencimiento a YYMMDD
    fecha_nacimiento_front = convert_date_to_mrz_format(front_data.get("fecha_nacimiento", ""))
    fecha_nacimiento_back = back_data.get("fechaNacimiento_MRZ", "")

    fecha_vencimiento_front = convert_date_to_mrz_format(front_data.get("fecha_vencimiento", ""))
    fecha_vencimiento_back = back_data.get("fechaVencimiento_MRZ", "")

    similarity_nacimiento = calculate_similarity(fecha_nacimiento_front, fecha_nacimiento_back)
    similarity_vencimiento = calculate_similarity(fecha_vencimiento_front, fecha_vencimiento_back)

    print("Fecha Nacimiento Similarity:", similarity_nacimiento)
    print("Fecha Vencimiento Similarity:", similarity_vencimiento)

    return (similarity_nacimiento >= threshold and similarity_vencimiento >= threshold)

def validate_names_partial(front_data, back_data, threshold):
    # Normalizar apellidos y nombres para comparación flexible
    apellido_paterno_front = unidecode(front_data.get("apellido_paterno", "").upper())
    apellido_materno_front = unidecode(front_data.get("apellido_materno", "").upper())
    nombres_front = unidecode(front_data.get("nombres", "").upper())

    apellido_paterno_back = unidecode(back_data.get("apellido_paterno", "").upper())
    apellido_materno_back = unidecode(back_data.get("apellido_materno", "").upper())
    nombres_back = unidecode(back_data.get("nombres", "").upper())

    # Si el apellido materno en back_data está vacío, separar el apellido paterno back en la "X"
    if apellido_materno_back == "":
        if "X" in apellido_paterno_back:
            parts = apellido_paterno_back.split("X", 1)  # Dividir en dos partes máximo
            apellido_paterno_back = parts[0]  # Asignar la parte antes de la "X" como apellido paterno
            apellido_materno_back = parts[1]  # Asignar la parte después de la "X" como apellido materno

    # Dividir nombres si están en un solo bloque en back_data
    nombres_back_split = nombres_back.split(" ")
    if len(nombres_back_split) == 1:  # Si los nombres no están separados
        # Dividir según las longitudes de los nombres en front_data
        nombres_parts = nombres_front.split(" ")
        nombres_back_split = []
        start_idx = 0
        for part in nombres_parts:
            length = len(part)
            nombres_back_split.append(nombres_back[start_idx:start_idx + length])
            start_idx += length
        nombres_back = " ".join(nombres_back_split)

    # Validar inclusión flexible en nombres
    similarity_apellido_paterno = calculate_name_similarity(apellido_paterno_front, apellido_paterno_back)
    similarity_apellido_materno = calculate_name_similarity(apellido_materno_front, apellido_materno_back)
    similarity_nombres = calculate_name_similarity(nombres_front, nombres_back)

    print("Apellido Paterno Similarity:", similarity_apellido_paterno)
    print("Apellido Materno Similarity:", similarity_apellido_materno)
    print("Nombres Similarity:", similarity_nombres)

    return (similarity_apellido_paterno >= threshold and
            similarity_apellido_materno >= threshold and
            similarity_nombres >= threshold)

def calculate_name_similarity(back_string, front_string):
    """
    Calcula la similitud entre dos cadenas, considerando:
    - Coincidencias exactas de caracteres en orden.
    - Permitiendo caracteres faltantes en una cadena.
    - Penalizando sustituciones o errores.
    
    :param back_string: Cadena más corta o incompleta.
    :param front_string: Cadena más larga o completa.
    :return: Porcentaje de coincidencia (0-100%).
    """
    # Asegurar que back_string sea más corto
    if len(back_string) > len(front_string):
        back_string, front_string = front_string, back_string

    back_index = 0
    errors = 0

    # Recorrer la cadena más larga (front_string)
    for char in front_string:
        if back_index < len(back_string) and back_string[back_index] == char:
            back_index += 1  # Coincidencia correcta, avanzamos en back_string
        elif back_index < len(back_string) and back_string[back_index] != char:
            errors += 1  # Error: carácter sustituido

    # Calcular coincidencias correctas y porcentaje
    matches = len(back_string) - errors
    total_characters = len(back_string)
    similarity_percentage = (matches / total_characters)

    return similarity_percentage

def validate_qr(qr, front_data, back_data, threshold):
    # Extraer los parámetros del QR usando expresiones regulares
    qr_data = {}
    qr_match = re.search(r"RUN=(\d{8})-\d&type=[^&]*&serial=(\d+)&mrz=(\d+)", qr)
    if qr_match:
        qr_data['RUN'] = qr_match.group(1)
        qr_data['serial'] = qr_match.group(2)
        qr_data['mrz'] = qr_match.group(3)
    else:
        return {"valid": False, "reason": "QR format invalid"}

    # Extraer el MRZ del textoGeneral_MRZ
    texto_general_mrz = back_data.get("textoGeneral_MRZ", "")
    # print(texto_general_mrz)
    extracted_mrz = extract_mrz_from_texto_general(texto_general_mrz)

    # Calcular similitudes
    similarity_run_qr_front = calculate_similarity(qr_data['RUN'], front_data['RUN'])
    similarity_run_qr_back = calculate_similarity(qr_data['RUN'], back_data['rut'])

    similarity_serial_qr_front = calculate_similarity(qr_data['serial'], front_data['numero_documento'])
    similarity_serial_qr_back = calculate_similarity(qr_data['serial'], back_data['numeroDocumento_MRZ'])

    similarity_mrz_qr_back = calculate_similarity(qr_data['mrz'], extracted_mrz)
    # print(qr_data['mrz'], extracted_mrz)

    # Mostrar las similitudes calculadas
    print("Similarity between QR RUN and Front RUN:", similarity_run_qr_front)
    print("Similarity between QR RUN and Back RUN:", similarity_run_qr_back)
    print("Similarity between QR Serial and Front Serial:", similarity_serial_qr_front)
    print("Similarity between QR Serial and Back Serial:", similarity_serial_qr_back)
    print("Similarity between QR MRZ and Back MRZ:", similarity_mrz_qr_back)

    # Validar si todas las similitudes están por encima del umbral
    valid = (
        similarity_run_qr_front >= threshold and
        similarity_run_qr_back >= threshold and
        similarity_serial_qr_front >= threshold and
        similarity_serial_qr_back >= threshold and
        similarity_mrz_qr_back >= threshold
    )

    if valid:
        return {"valid": True, "reason": "All data matches correctly"}

    return {"valid": False, "reason": "Some data in QR does not match front or back data"}


def calculate_similarity(str1, str2):
    # Usa la métrica de similitud para calcular el porcentaje de coincidencia
    return SequenceMatcher(None, str1, str2).ratio()

def convert_date_to_mrz_format(date_str):
    month_map = {
        "ENE": "01", "FEB": "02", "MAR": "03", "ABR": "04", "MAYO": "05", "JUN": "06",
        "JUL": "07", "AGO": "08", "SEPT": "09", "OCT": "10", "NOV": "11", "DIC": "12"
    }
    try:
        match = re.match(r"(\d{2}) (\w+) (\d{4})", date_str.upper())
        if not match:
            return ""
        day, month_str, year = match.groups()
        month = month_map.get(month_str, "00")
        return f"{year[-2:]}{month}{day.zfill(2)}"
    except Exception:
        return ""

def extract_mrz_from_texto_general(texto_general):
    # Limpiar texto eliminando espacios y caracteres '<'
    texto_limpio = texto_general.replace(" ", "").replace("<", "")
    partes = texto_limpio.split("CHL")
    
    if len(partes) > 1:
        # Tomar la parte relevante después de "CHL"
        seccion_interes = partes[1]
        
        # Extraer los primeros 10 dígitos
        primeros_10_digitos = ''.join([char for char in seccion_interes[:10] if char.isdigit()])
        
        # Procesar el resto después de los primeros 10 dígitos
        resto = seccion_interes[10:]
        resto_sin_patron = re.sub(r'[A-Z]\d{2}', '', resto, count=1)  # Elimina el patrón de 1 letra y 2 números

        # Combinar los primeros 10 dígitos con el resto limpio
        resultado = primeros_10_digitos + ''.join([char for char in resto_sin_patron if char.isdigit()])
        # print(resultado)  
        return resultado

    return ""

