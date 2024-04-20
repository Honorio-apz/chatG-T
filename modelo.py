import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from tensorflow.keras.optimizers import SGD
import random

import tensorflow as tf
import keras

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('new_dataset_admision_v3.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        #agregar documentos en el corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
    

    # lemmatize, lower each word and remove duplicates
    # lematizar, bajar cada palabra y eliminar duplicados
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    #print(words)
    # sort classes
    classes = sorted(list(set(classes)))
    # documents = combination between patterns and intents
    # documentos = combinación entre patrones e intenciones
    print (len(documents), "documents")
    # classes = intents
    print (len(classes), "classes", classes)
    # words = all words, vocabulary
    print (len(words), "unique lemmatized words", words)
    pickle.dump(words,open('words_chatbot__parte.pkl','wb'))
    pickle.dump(classes,open('classes_chatbot_parte.pkl','wb'))



# create our training data
# crear nuestros datos de entrenamiento
training = []
# create an empty array for our output
# crear una matriz vacía para nuestra salida
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
# conjunto de entrenamiento, bolsa de palabras para cada oración
for doc in documents:
    # initialize our bag of words
    # Inicializar nuestra bolsa de palabras.
    bag = []
    # list of tokenized words for the pattern
    # lista de palabras tokenizadas para el patrón
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    # lematizar cada palabra: crear una palabra base, en un intento de representar palabras relacionadas
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    # cree nuestra matriz de bolsa de palabras con 1, si se encuentra una coincidencia de palabras en el patrón actual
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    # la salida es un '0' para cada etiqueta y un '1' para la etiqueta actual (para cada patrón)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])


print("¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡")
#print(training)
# shuffle our features and turn into np.array / barajar nuestras funciones y convertirlas en np.array
random.shuffle(training)
training = np.array(training, dtype=object)
# create train and test lists. X - patterns, Y - intents / crear listas de trenes y pruebas. X - patrones, Y - intenciones
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")



# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
#Crear modelo - 3 capas. La primera capa tiene 128 neuronas, la segunda capa tiene 64 neuronas y la tercera capa de salida contiene varias neuronas.
# equal to number of intents to predict output intent with softmax
#igual al número de intentos para predecir el intento de salida con softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
#Compilar modelo. El descenso de gradiente estocástico con gradiente acelerado de Nesterov da buenos resultados para este modelo
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd=tf.keras.optimizers.legacy.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('Model_chatbot_parte.h5', hist)
print("model created")