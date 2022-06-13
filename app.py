
from flask import Flask, render_template, request
import keras.models
from keras.models import load_model
import tensorflow
import cv2
import numpy as np


model= keras.models.load_model('model_filter.h5')

app = Flask(__name__)
@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')
@app.route('/',methods=['POST'])
def predict():
    imagefile=  request.files['imagefile']
    image_path= "./image/"+ imagefile.filename
    imagefile.save(image_path)
    img=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    resize_dim=32
    img=cv2.resize(img,(resize_dim,resize_dim),interpolation=cv2.INTER_AREA) 
    gaussian_3 = cv2.GaussianBlur(img, (9,9), 10.0) #unblur
    img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #filter
    img = cv2.filter2D(img, -1, kernel)
    X=[]
    X.append(img)
    X = np.asarray(X)
    X = X.reshape(X.shape[0],32, 32,1).astype('float32')
    X = X/255
    predictions_prob=model.predict(X)
    labels=[np.argmax(pred) for pred in predictions_prob]

    
    return render_template('index.html', prediction= labels)



if __name__ == '__main__':
    app.run(port=3000,debug=True)

