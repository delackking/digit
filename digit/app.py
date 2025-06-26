
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

def process_image(img):
    image = Image.open(img).convert('L').resize((8, 8))
    image = np.array(image)
    image = image / 255.0 * 16.0
    image = image.flatten().reshape(1, -1)
    scaled = scaler.transform(image)
    pca_img = pca.transform(scaled)
    return pca_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    processed = process_image(img)
    prediction = model.predict(processed)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
