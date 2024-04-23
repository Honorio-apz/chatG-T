import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import re
import random

# TensorFlow and Keras imports
# Optional imports for data manipulation and preprocessing (uncomment if needed)
import numpy as np
import pandas as pd
import json

# Optional import for text classification (uncomment if needed)
from sklearn.preprocessing import LabelEncoder
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
sess = tf.compat.v1.Session(config=config)

# Optional import for serialization (uncomment if needed)
import pickle
uploaded = "lstm/chatbot.json"

with open("lstm/chatbot.json", 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['intents'])

# Almacenar datos que serán convertidos a DataFrame
dic = {"tag": [], "patterns": [], "responses": []}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]

    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)

# Crear un nuevo DataFrame a partir del diccionario
df = pd.DataFrame.from_dict(dic)

# Mostrar el DataFrame para verificar
df.head()

# Obtener las etiquetas únicas
df['tag'].unique()
# Load the tokenizer, label encoder, and trained model
with open('lstm/tokenizer_lstm.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('lstm/label_encoder_lstm.pkl', 'rb') as handle:
    lbl_enc = pickle.load(handle)

# Replace 'my_model.keras' with the actual filename of your saved model
model = load_model('lstm/my_lstm_model.keras')  # (LSTM, BiLSTM, GRU, or BiGRU)



def input_user(pattern, tokenizer, maxlen=18):
    text = re.sub(r"[^a-zA-Z\']", ' ', pattern).lower()
    text = text.split()
    text = " ".join(text)
    x_test = tokenizer.texts_to_sequences([text])
    x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
    return x_test

def predict_response(pattern, model, tokenizer, lbl_enc, df):
    x_test = input_user(pattern, tokenizer)
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax(axis=1)
    tag = lbl_enc.inverse_transform(y_pred)[0]
    responses = df[df['tag'] == tag]['responses'].values[0]
    if(max(max(model.predict(x_test)))>0.7):
        return random.choice(responses)
    else:
        return "En este momento no podemos asistirle con esta consulta. Le rogamos que intente formular una pregunta relacionada con nuestro ámbito de especialización."



###################################################################################
###################################################################################
###################################################################################
###################################################################################
######################################FLASK########################################
###################################################################################
###################################################################################
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
app.static_folder = 'static'
@app.route('/')
def home():

    return render_template('index.html')

@app.route("/get")
def saludar():
    userText = request.args.get('msg')
    #print(chatbot_response(userText))
    return predict_response(userText, model, tokenizer, lbl_enc, df)
    
if __name__ == "__main__":
    app.run(host='127.0.0.95',port=8083,debug=True)
