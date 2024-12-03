from dotenv import load_dotenv
import os

# Cargar las variables del archivo .env
load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')

class DevelopmentConfig(Config):
    DEBUG = True

config = {
    'development': DevelopmentConfig
}
