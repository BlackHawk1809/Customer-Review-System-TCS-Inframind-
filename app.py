from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import cv2
import re
import os
from tensorflow.keras.models import load_model
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from app_recommend import similar_prods, prod_ranking_model
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import pickle





#######################
IMAGE_FOLDER = os.path.join('static', 'img_pool')
######################



model = keras.models.load_model("./models/Model_Text_Analysis.h5")
with open("./models/tokenizer.pickle", "rb") as tok:
	tokenizer = pickle.load(tok)


def preprocess_text(text):
	text = text.lower()
	text = [text]
	seq_text = tokenizer.texts_to_sequences(text)
	final_text = pad_sequences(seq_text, maxlen=20, dtype="int32", value=0)
	return final_text


app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route("/", methods=["GET","POST"])
def home():
	return(render_template("FIRST.html"))

@app.route("/predict", methods=["GET", "POST"])
def predict():
	if request.method=="POST":
		text = request.form['text']
		ready_text = preprocess_text(text)
		prediction = model.predict(ready_text)
		prediction = round(float(prediction))
		result = ""
		if prediction == 0:
			result = "Negative"
			img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Sad_Emoji.png')
		elif prediction == 1:
			result = "Positive"
			img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Smiling_Emoji.png')

		return render_template("FIRST.html", prediction = prediction, image=img_filename)
	else:
		return render_template("FIRST.html")



#########################End Code for  Text Sentiment Analysis  ########################

######################### Code for Image Sentiment
@app.route('/image_sentiment')
def index():
    return render_template('index.html')



@app.route('/image_sentiment/after', methods=['GET', 'POST'])
def after():
    img = request.files['file1']

    img.save('static/file.jpg')

    ####################################
    img1 = cv2.imread('static/file.jpg')
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 3)

    for x,y,w,h in faces:
        cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 2)

        cropped = img1[y:y+h, x:x+w]

    cv2.imwrite('static/after.jpg', img1)

    try:
        cv2.imwrite('static/cropped.jpg', cropped)

    except:
        pass

    #####################################

    try:
        image = cv2.imread('static/cropped.jpg', 0)
    except:
        image = cv2.imread('static/file.jpg', 0)

    image = cv2.resize(image, (48,48))
    image = image/255.0
    image = np.reshape(image, (1,48,48,1))
    model = load_model('model_image_analysis.h5')
    prediction = model.predict(image)
    label_map =   ['Anger','Neutral' , 'Fear', 'Happy', 'Sad', 'Surprise']

    prediction = np.argmax(prediction)

    final_prediction = label_map[prediction]

    return render_template('FIRST.html', data=final_prediction)

    #########################################
    #################### Recommendation #####################

@app.route("/view")
def view():
    prod_name = str(request.args.get('prod')).upper()

    if prod_name in prod_ranking_model['Product'].unique():
        prod_price = similar_prods(prod_name)
        return render_template('prod_view.html',prod=prod_name,price=prod_price,exists='y')
    else:
        return render_template('prod_view.html',prod=prod_name,exists='n')


        ############################ End Recommendation ################


    ###################### chat bot code ###############
@app.route('/chat')
def chat():
    return render_template('demo-ddg.html')

        #####################################


if __name__ == "__main__":
    # init()
    app.run()
