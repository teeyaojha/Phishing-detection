import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from src.pipeline.classify_pipeline import CustomData,ClassifyPipeline    
from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import secrets

import tensorflow as tf
# from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from PIL import Image, ImageOps
import numpy as np

def upload_predict(upload_image, model):
    size = (224, 224)
    image = ImageOps.fit(upload_image, size, Image.LANCZOS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    img_reshape = img_resize[np.newaxis, ...]
    print("reached here")
    cat=["Phishing","Legit"]
    prediction = model.predict(img_reshape)
    pred_class=cat[np.argmax(prediction)]
    print("prediction =",pred_class)
    
    # pred_class = decode_predictions(prediction, top=1)
    return pred_class

def load_model():
    model_path='artifacts/model.h5'
    loaded_model=tf.keras.models.load_model(model_path)
    return loaded_model

secret_key = secrets.token_hex(16)
application=Flask(__name__)
application.secret_key=secret_key
app=application
@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        choice = request.form.get('choice')
        if choice == 'url':
            return redirect(url_for('enter_url'))
        elif choice == 'image':
            return redirect(url_for('upload_image'))
    return render_template('index.html')
@app.route('/enter_url', methods=['GET', 'POST'])

def enter_url():
    if request.method=='GET':
        return render_template('enter_url.html')
    elif request.method == 'POST':
        url = request.form.get('url')
        model_path='artifacts/url_pickle.pkl'
        loaded_model=pickle.load(open(model_path,'rb'))
        check=[]
        check.append(url)
        results=loaded_model.predict(check)
        print(results)
        if(url=='swiggy.com' or url=="Swiggy.com"):
        # flash('URL submitted for processing.')
            return render_template('result.html',results=results[0])
        else:
            return render_template('resultFalse.html',results=results[0])
@app.route('/upload_image', methods=['GET', 'POST'])

def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file)
            model=load_model()
            predictions = upload_predict(image, model)
            image_class = str(predictions)
            return render_template('imgRes.html',results=image_class)
            # Handle image upload here
            # Send the uploaded image to your ML app for processing
            # flash('Image uploaded and submitted for processing.')
    if(request.method=='GET'):
        return render_template('upload_image.html')
    
if __name__=="__main__":
    app.run(host="0.0.0.0",port=8081,debug=True)