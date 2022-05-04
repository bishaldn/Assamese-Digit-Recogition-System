
from flask import Flask, render_template, request
import pickle

import cv2
import numpy as np


model=pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)
@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')
@app.route('/',methods=['POST'])
def predict():
    imagefile=  request.files['imagefile']
    image_path= "./image/"+ imagefile.filename
    imagefile.save(image_path)


    im = cv2.imread(image_path)
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im_gray  =cv2.GaussianBlur(im_gray, (15,15), 0)
   
   #Threshold the image
    ret, im_th = cv2.threshold(im_gray,100, 255, cv2.THRESH_BINARY)
    roi = cv2.resize(im_th, (28,28), interpolation  =cv2.INTER_AREA)
   
    rows,cols=roi.shape
   
    X = []
   
   ## Add pixel one by one into data array
    for i in range(rows):
       for j in range(cols):
           k = roi[i,j]
           if k>100:
               k=1
           else:
               k=0
           X.append(k)
           
    classification  =model.predict([X])
    
    return render_template('index.html', prediction= classification)



if __name__ == '__main__':
    app.run(port=3000,debug=True)
