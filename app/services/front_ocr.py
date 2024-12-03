import cv2
import pytesseract
import threading
import re
import os

# Configuración de la ruta de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Crear una carpeta temporal para guardar imágenes
if not os.path.exists('tmp_segments'):
    os.makedirs('tmp_segments')

# Segmentos definidos con ajustes en las coordenadas
segmentos = {
    "apellido_paterno": (250, 90, 800, 120),
    "apellido_materno": (250, 115, 800, 150),
    "nombres": (250, 165, 900, 200),
    "nacionalidad": (250, 220, 460, 265),
    "sexo": (470, 220, 530, 250),
    "fecha_nacimiento": (270, 270, 480, 310),
    "numero_documento": (470, 270, 665, 320),
    "fecha_emision": (250, 330, 480, 370),
    "fecha_vencimiento": (470, 325, 685, 375),
    "numero_identificador": (10, 450, 275, 510)
}

def normalizar_fecha(valor):
    """
    Normaliza las fechas en el formato DD MMM YYYY.
    Maneja números consecutivos (725 SEPT 2027) extrayendo los últimos dos dígitos como día.
    """
    # Mapeo de meses válidos
    month_map = {
        "ENE": "ENE", "FEB": "FEB", "MAR": "MAR", "ABR": "ABR", "MAYO": "MAYO", "MAY": "MAYO",
        "JUN": "JUN", "JUL": "JUL", "AGO": "AGO", "SEPT": "SEPT", "OCT": "OCT", "NOV": "NOV", "DIC": "DIC"
    }

    # Eliminar caracteres no deseados y normalizar texto
    valor = re.sub(r"[^A-Za-z0-9\s]", "", valor).strip().upper()
    partes = valor.split()

    dia, mes, anio = None, None, None

    # Buscar el mes
    for idx, parte in enumerate(partes):
        if parte in month_map:  # Identificar el mes
            mes = month_map[parte]

            # Buscar el número correcto antes del mes
            dia_candidato = None
            for i in range(idx - 1, -1, -1):  # Revisar hacia atrás
                if partes[i].isdigit():
                    if len(partes[i]) == 3:  # Número de tres dígitos encontrado
                        dia_candidato = partes[i][-2:]  # Tomar los últimos 2 caracteres
                        break
                    elif len(partes[i]) == 2:  # Número de dos dígitos encontrado
                        dia_candidato = partes[i]
                        break
                    elif len(partes[i]) == 1 and int(partes[i]) < 10:  # Número menor a 10
                        dia_candidato = f"0{partes[i]}"
                        break

            # Asignar el día candidato si se encontró
            dia = dia_candidato

            # Buscar el año (parte numérica después del mes)
            if idx + 1 < len(partes) and partes[idx + 1].isdigit():
                anio = partes[idx + 1].zfill(4)
            break

    # Validar y asignar valores predeterminados
    dia = dia or "01"
    anio = anio or "0000"

    if mes:  # Si se encontró un mes, devolver la fecha
        return f"{dia} {mes} {anio}"
    return valor  # Si no, devolver el valor original

def detectar_problemas_flash_mrz(image, segmentos, umbral_brillo=240, area_minima=500):
    for campo, coords in segmentos.items():
        x1, y1, x2, y2 = coords
        segmento_imagen = image[y1:y2, x1:x2]
        
        # Convertir a escala de grises
        gris = cv2.cvtColor(segmento_imagen, cv2.COLOR_BGR2GRAY)
        
        # Crear una máscara de áreas sobreexpuestas
        _, mascara_sobreexpuesta = cv2.threshold(gris, umbral_brillo, 255, cv2.THRESH_BINARY)
        
        # Encontrar contornos en la máscara
        contornos, _ = cv2.findContours(mascara_sobreexpuesta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area > area_minima:
                return True, f"Problema detectado: La zona tiene áreas sobreexpuestas probablemente debido al uso del flash en el segmento {campo}."
    
    return False, "No se detectaron problemas de sobreexposición."

def preprocesar_segmento(segmento_imagen, binary_number= 55):
    """
    Aplica preprocesamiento a un segmento específico para mejorar el OCR.
    """
    gris = cv2.cvtColor(segmento_imagen, cv2.COLOR_BGR2GRAY)
    umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, binary_number, 25)
    return umbral

def procesar_segmento(image, segmento, campo, resultado_ocr):
    """
    Procesa un segmento de la imagen aplicando OCR y retorna el texto detectado.
    """
    try:
        x1, y1, x2, y2 = segmento
        segmento_imagen = image[y1:y2, x1:x2]

        # Aplicar preprocesamiento individual al segmento
        segmento_preprocesado = None
        binary_number = 55
        if campo == "numero_documento":
            binary_number = 95
            segmento_preprocesado = preprocesar_segmento(segmento_imagen, binary_number)
        else:
            segmento_preprocesado = preprocesar_segmento(segmento_imagen, binary_number)

        # Guardar segmento preprocesado (opcional)
        binarized_path = f'tmp_segments/{campo}.jpg'
        cv2.imwrite(binarized_path, segmento_preprocesado)

        # Configuración específica de Tesseract
        config = "--psm 6"
        if campo == "numero_documento":
            config = "--psm 8"  # Usar configuración enfocada en una sola línea de texto

        texto_segmento = pytesseract.image_to_string(segmento_preprocesado, config=config, lang='spa').strip()
        # Limpiar los datos obtenidos
        texto_limpio = limpiar_datos(campo, texto_segmento)

        # Almacenar los resultados de OCR
        if campo == "numero_identificador":
            if isinstance(texto_limpio, tuple):
                numero, digito_verificador = texto_limpio
                resultado_ocr["RUN"] = numero
                resultado_ocr["digito_verificador"] = digito_verificador
            else:
                resultado_ocr[campo] = texto_limpio
        else:
            resultado_ocr[campo] = texto_limpio

    except Exception as e:
        print(f"Error al procesar el campo {campo}: {e}")
        resultado_ocr[campo] = f"Error en {campo}"

def limpiar_datos(campo, valor):
    """
    Limpia los datos eliminando caracteres no deseados y ajusta el formato específico para cada campo.
    """
    if campo == "numero_identificador":
        valor = re.sub(r"[^0-9-]", "", valor)
        try:
            data = valor.split("-")
            numero = data[0]
            digito_verificador = data[1]
            return numero, digito_verificador
        except:
            return "Error en numero_identificador"

    elif campo == "numero_documento":
        valor = re.sub(r"\.", "", valor)  # Eliminar puntos
        valor = re.sub(r"[^0-9]", "", valor)  # Retener solo números

    elif campo in ["nombres", "apellido_paterno", "apellido_materno"]:
        valor = re.sub(r"[^A-ZÁÉÍÓÚÑ\s]", "", valor)  # Eliminar caracteres no deseados
        valor = " ".join([palabra for palabra in valor.split() if len(palabra) > 1])  # Palabras válidas

    elif campo == "nacionalidad":
        valor = valor.replace("NACIONALIDAD", "").strip()
        if "CHILENA" in valor:
            return "CHILENA"
        match = re.search(r"[A-Z]{3}", valor)
        if match:
            return match.group(0)

    elif campo in ["fecha_nacimiento", "fecha_vencimiento"]:
        valor = normalizar_fecha(valor)  # Llamar a la función para normalizar las fechas
    elif campo == "sexo":
        valor = re.sub(r"[^MF]", "", valor.upper())  # Solo iniciales de sexo válidas (M/F)

    else:
        # Limpiar genéricamente eliminando caracteres no deseados
        valor = re.sub(r"[-—\.]", "", valor).replace("\n", " ").strip()
        valor = " ".join(valor.split())

    return valor

def procesar_ocr_completo(image):
    """
    Función principal que procesa la imagen, la segmenta y aplica OCR a cada segmento.
    """
    # Detectar problemas de sobreexposición
    problema_detectado, mensaje = detectar_problemas_flash_mrz(image, segmentos)
    if problema_detectado:
        print(mensaje)
        return {"error": mensaje}

    resultado_ocr = {}
    hilos = []
    for campo, coordenadas in segmentos.items():
        hilo = threading.Thread(target=procesar_segmento, args=(image, coordenadas, campo, resultado_ocr))
        hilos.append(hilo)
        hilo.start()
    for hilo in hilos:
        hilo.join()
    return resultado_ocr