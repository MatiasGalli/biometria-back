import cv2
import pytesseract
import re
import json
import os

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

# Función para detectar problemas de flash en la zona MRZ
def detectar_problemas_flash_mrz(image, x1, y1, x2, y2, umbral_brillo=240, area_minima=500):
    segmento_imagen = image[y1:y2, x1:x2]
    gris = cv2.cvtColor(segmento_imagen, cv2.COLOR_BGR2GRAY)
    _, mascara_sobreexpuesta = cv2.threshold(gris, umbral_brillo, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(mascara_sobreexpuesta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contorno in contornos:
        if cv2.contourArea(contorno) > area_minima:
            return True, "Problema detectado: Áreas sobreexpuestas en la MRZ probablemente debido al flash."
    return False, "No se detectaron problemas de sobreexposición en la zona MRZ."

# Función para preprocesar la sección MRZ
def preprocesar_segmento(segmento_imagen):
    gris = cv2.cvtColor(segmento_imagen, cv2.COLOR_BGR2GRAY)
    _, binarizado = cv2.threshold(gris, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarizado

def resize_image(image, scale):
    height, width = image.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized_image

# Función para realizar OCR en la MRZ
def realizar_ocr_mrz(image):
    config = "--psm 1 --oem 1"
    return pytesseract.image_to_string(image, config=config, lang='mrz').strip()

# Funciones auxiliares para extraer datos
def extraer_numerodoc_mrz(linea_raw):
    digitos = re.findall(r'\d', linea_raw)
    return ''.join(digitos)[:9]

def extraer_run_mrz(mrz_raw):
    """
    Extrae el RUN y el dígito verificador de la segunda línea de la MRZ.
    Garantiza que las tres letras iniciales sean alfabéticas, corrigiendo errores comunes como 'B0L' -> 'BOL'.
    """
    # Corrección de caracteres comunes que el OCR puede confundir
    def corregir_letras(texto):
        # Mapeo para corregir caracteres confusos
        correcciones = {
            "0": "O",  # Cero a O
            "1": "I",  # Uno a I
            "5": "S",  # Cinco a S (menos común)
        }
        return ''.join(correcciones.get(c, c) for c in texto)

    # Regex para detectar el patrón de tres letras + números + '<'
    match = re.search(r'([A-Z0-9]{3})(\d+)<(\w)', mrz_raw)
    if match:
        letras = match.group(1)  # Tres caracteres iniciales
        run = match.group(2)  # RUN
        digito_verificador = match.group(3)  # Dígito verificador

        # Corregir las letras iniciales
        letras_corrigidas = corregir_letras(letras)
        if not letras_corrigidas.isalpha():  # Si no son solo letras, forzar alfabéticas
            letras_corrigidas = re.sub(r'[^A-Z]', 'O', letras_corrigidas)  # Sustituir cualquier no letra por 'O'

        # Retornar el RUN y el dígito verificador
        return run, digito_verificador
    return "", ""  # Si no encuentra el patrón, retornar vacío


def extraer_fechas_mrz(linea_2):
    """
    Extrae fechas de nacimiento y vencimiento de la segunda línea de la MRZ.
    Busca patrones coherentes con DDMMYY.
    """
    fecha_nacimiento = re.search(r'\d{6}', linea_2[:8])  # Buscar la fecha de nacimiento en los primeros 14 caracteres
    fecha_vencimiento = re.search(r'\d{6}', linea_2[8:])  # Buscar la fecha de vencimiento después

    return {
        "fechaNacimiento": fecha_nacimiento.group(0) if fecha_nacimiento else "",
        "fechaVencimiento": fecha_vencimiento.group(0) if fecha_vencimiento else ""
    }

def extraer_nombres_apellidos_mrz(linea_3):
    partes = linea_3.split("<<")
    apellidos = partes[0].replace("<", " ").strip().split()
    nombres = partes[1].replace("<", " ").strip().split() if len(partes) > 1 else []
    return {
        "apellido_paterno": apellidos[0] if len(apellidos) > 0 else "",
        "apellido_materno": apellidos[1] if len(apellidos) > 1 else "",
        "nombres": " ".join(nombres)
    }

def corregir_caracteres_especificos(texto):
    """
    Reemplaza caracteres específicos malinterpretados por Tesseract OCR.
    """
    correcciones = {
        "/": "7",  # Reemplaza '/' por '7'
    }
    return ''.join(correcciones.get(char, char) for char in texto)

# Función principal para procesar el reverso de la cédula
def procesar_ocr_reverso(image):
    # Coordenadas únicas del segmento MRZ
    x1, y1, x2, y2 = 30, 345, 810, 490
    print(f"Coordenadas del segmento MRZ: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # Extraer la zona MRZ completa
    zona_mrz = image[y1:y2, x1:x2]

    # Detectar problemas de flash
    problema_flash, mensaje_flash = detectar_problemas_flash_mrz(image, x1, y1, x2, y2)
    if problema_flash:
        return {
            "text": {
                "datosMRZ": {},
                "datosOCR": {},
                "flashWarning": mensaje_flash
            }
        }

    # Preprocesar la imagen de la MRZ
    zona_mrz_preprocesada = preprocesar_segmento(zona_mrz)
    resized = resize_image(zona_mrz_preprocesada, 5)

    # Realizar OCR en la MRZ
    texto_ocr = realizar_ocr_mrz(resized)
    texto_ocr = corregir_caracteres_especificos(texto_ocr)
    lineas = texto_ocr.split("\n")
    lineas = [linea for linea in lineas if len(linea.strip()) > 10]

    if len(lineas) < 3:
        return {
            "text": {
                "datosMRZ": {},
                "datosOCR": {},
                "flashWarning": "No se pudieron identificar las tres líneas de la MRZ"
            }
        }

    # Extraer información específica de las líneas
    linea_1, linea_2, linea_3 = lineas[0], lineas[1], lineas[2]
    documento_id = extraer_numerodoc_mrz(linea_1)
    run, digito_verificador = extraer_run_mrz(linea_2)
    fechas = extraer_fechas_mrz(linea_2)
    nombres_apellidos = extraer_nombres_apellidos_mrz(linea_3)

    # Crear el resultado final en el formato esperado
    resultado_final = {
        
        "datosMRZ": {
                "apellido_materno": nombres_apellidos["apellido_materno"],
                "apellido_paterno": nombres_apellidos["apellido_paterno"],
                "digito_verificador": digito_verificador,
                "fechaNacimiento_MRZ": fechas["fechaNacimiento"],
                "fechaVencimiento_MRZ": fechas["fechaVencimiento"],
                "nombres": nombres_apellidos["nombres"],
                "numeroDocumento_MRZ": documento_id,
                "rut": run,
                "textoGeneral_MRZ": " ".join(lineas)
            },
            "datosOCR": {
                "linea_1": linea_1,
                "linea_2": linea_2,
                "linea_3": lineas[2]
            },
            "flashWarning": mensaje_flash
        }
    

    return resultado_final