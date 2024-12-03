from flask import Flask
from settings import config
from flask_cors import CORS


#Routes
from routes import OcrRoutes,ImageRoutes, ValidateRoutes

app = Flask(__name__)

CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

def page_not_found(error):
    return 'Esta pÃ¡gina no existe', 404

if __name__ == '__main__':
    #Config
    app.config.from_object(config['development'])

    #Blueprints
    app.register_blueprint(OcrRoutes.main, url_prefix='/api/ocr')
    app.register_blueprint(ImageRoutes.main, url_prefix='/api/static')
    app.register_blueprint(ValidateRoutes.main, url_prefix='/api/validate')

    #Error handlers
    app.register_error_handler(404, page_not_found)
    app.run(host='0.0.0.0', port=5000)