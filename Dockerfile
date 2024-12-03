FROM python:3.12.7

# Define el directorio de trabajo
WORKDIR /usr/src

# Instalar las dependencias necesarias para compilar Tesseract desde fuente
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libtesseract-dev \
    libleptonica-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    libtiff-dev \
    zlib1g-dev \
    libicu-dev \
    libpango1.0-dev \
    libcairo2-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Descargar e instalar Tesseract desde fuente
RUN wget https://github.com/tesseract-ocr/tesseract/archive/refs/tags/5.5.0.tar.gz && \
    tar -xzf 5.5.0.tar.gz && \
    cd tesseract-5.5.0 && \
    mkdir build && cd build && \
    cmake .. && \
    cmake --build . --target install && \
    ldconfig && \
    cd /usr/src && rm -rf tesseract-5.5.0 5.5.0.tar.gz

# Instalar dependencias adicionales (zbar para QR y código de barras)
RUN apt-get update && apt-get install -y \
    libzbar0 \
    zbar-tools \
    && rm -rf /var/lib/apt/lists/*

# Descargar eng.traineddata al directorio tessdata
COPY app/models/traneddata/eng.traineddata /usr/share/tesseract-ocr/5/tessdata/eng.traineddata
COPY app/models/traneddata/spa.traineddata /usr/share/tesseract-ocr/5/tessdata/spa.traineddata

# Copiar el archivo mrz.traineddata al directorio tessdata
COPY app/models/traneddata/mrz.traineddata /usr/share/tesseract-ocr/5/tessdata/mrz.traineddata

# Copiar los archivos de requerimientos primero
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos del proyecto
COPY . .

# Expone el puerto para Flask
EXPOSE 5000

# Configurar variables de entorno para Flask
ENV FLASK_APP=/usr/src/app/app.py
ENV FLASK_ENV=development

# Ejecutar la aplicación Flask
CMD ["python", "/usr/src/app/app.py"]