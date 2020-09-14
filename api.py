import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

import pickle
import pandas as pd
import numpy as np
import tensorflow as tf


data = pickle.load( open( "smart-reply-data.pkl", "rb" ) )
words = data['words']
classes = data['classes']
#response = data['']
# import our intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))



model = tf.keras.models.load_model('models/smartmodel')

app = Flask(__name__)
CORS(app)

@app.route("/api/v1/smartreply", methods=['POST'])
def classify():
    ERROR_THRESHOLD = 0.20
    
    sentence = request.json['sentence']
    
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    replies = []
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        for intent in intents['intents']:
            if intent['tag'] in classes[r[0]]:
                replies.append(intent['responses'])
        return_list.append({"intent": classes[r[0]], "probability": str(r[1]), "responses": replies})
    # return tuple of intent and probability
    
    response = jsonify(return_list)
    return response

# running REST interface
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)


