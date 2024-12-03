import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps

# Definir la red siamesa
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096)
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return F.normalize(output, p=2, dim=1)

# Función para comparar imágenes
def compare_faces(img1_path, img2_path, model, transform, device, should_invert=True):
    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')

    if should_invert:
        img1 = ImageOps.invert(img1)
        img2 = ImageOps.invert(img2)

    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding1 = model.forward_once(img1)
        embedding2 = model.forward_once(img2)

    return F.pairwise_distance(embedding1, embedding2).item()

# Configuraciones y carga de modelo
model_path = os.path.join(os.path.dirname(__file__), '../models/siamese_model_final.pth')
model_path = os.path.abspath(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

__all__ = ['model', 'device', 'transform', 'compare_faces']
# Rutas de las imágenes a comparar
# ruta_img1 = "/Users/matiasgalli/Documents/BACKWENO/static/face1_image_49a31c9db9f493d0b85b126a5187c659.jpg"
# ruta_img2 = "/Users/matiasgalli/Documents/BACKWENO/static/face2_image_284ee91d3a62f3f7f619801be4ca6932.jpg"

# distance = compare_faces(ruta_img1, ruta_img2, model, transform, device)
# print(f"La distancia euclidiana entre las imágenes es: {distance:.4f}")
