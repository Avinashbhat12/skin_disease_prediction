# -*- coding: utf-8 -*-
"""

"""

from __future__ import division, print_function
import sys
import os
import glob
import numpy as np
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Skin_Diseasess.h5'

# Load your trained model
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(64, 64))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        # Predict class
        pred_probs = model.predict(x)
        pred_class = np.argmax(pred_probs)
        index = ['BA- cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles']
        text = "Prediction : " + index[pred_class]

        return text
    
if __name__ == '__main__':
    app.run(debug=False)
