from flask import Flask, render_template, request, jsonify
import torch
import base64
import io
from PIL import Image
import torchvision.transforms as transforms

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from ViT import ViT

app = Flask(__name__)

# --- ViTモデルの準備 ---
model = ViT(
    image_size=28,
    patch_size=7,
    n_classes=10, 
    channels=1,
    dim=128, 
    depth=6, 
    n_heads=8, 
    mlp_dim=256
)
model.load_state_dict(torch.load('vit_mnist_model.pth'))
model.eval()

def transform_image(image_bytes):
    transform = transforms.Compose([
        # モデルの定義 image_size=28 に合わせる
        transforms.Resize((28, 28)), 
        # グレースケールに変換（モデルの channels=1 に合わせる）
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(),
        # MNISTの標準的な正規化、または学習時と同じ設定に
        transforms.Normalize((0.5,), (0.5,))
    ])
    # convert('L') でグレースケールとして読み込み
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    return transform(image).unsqueeze(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    # Base64データをデコード
    image_data = base64.b64decode(data['image'].split(',')[1])
    tensor = transform_image(image_data)
    
    # 推論
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = predicted.item()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)