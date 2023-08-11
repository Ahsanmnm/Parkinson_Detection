import os
from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image):
    image = image.resize((256, 256))
    image_array = np.array(image)
    return image_array

def threshold_prediction(prediction, threshold=0.5):
    if prediction >= threshold:
        return "Parkinson's Disease"
    else:
        return "Healthy"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image = Image.open(image_file)
    processed_image = preprocess_image(image)
    reshaped_image = processed_image.reshape((1, 256, 256, 3))
    prediction = model.predict(reshaped_image)
    thresholded_prediction = threshold_prediction(prediction)
    
    # Save the uploaded image in the uploads folder
    uploaded_image_filename = 'uploaded_image.jpg'
    uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image_filename)
    image.save(uploaded_image_path)
    
    return render_template('index.html', prediction_text=thresholded_prediction, uploaded_image=uploaded_image_filename)

if __name__ == "__main__":
    app.run(debug=True)
import os
from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image):
    image = image.resize((256, 256))
    image_array = np.array(image)
    return image_array

def threshold_prediction(prediction, threshold=0.5):
    if prediction >= threshold:
        return "Parkinson's Disease"
    else:
        return "Healthy"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
# ... (other imports and code) ...

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image = Image.open(image_file)
    processed_image = preprocess_image(image)
    reshaped_image = processed_image.reshape((1, 256, 256, 3))
    prediction = model.predict(reshaped_image)
    thresholded_prediction = threshold_prediction(prediction)
    
    # Save the uploaded image in the uploads folder
    uploaded_image_filename = 'uploaded_image.jpg'
    uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image_filename)
    image.save(uploaded_image_path)
    
    return render_template('index.html', prediction_text=thresholded_prediction, uploaded_image=uploaded_image_filename)

if __name__ == "__main__":
    app.run(debug=True)

