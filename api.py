import random
import pickle
import re
import json

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import pandas as pd
import numpy as np
import tensorflow as tf

data = pickle.load( open( "smart-reply-data.pkl", "rb" ) )
words = data['words']
classes = data['classes']

# import our intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return(np.array(bag))


model = tf.keras.models.load_model('models/smartmodel')

app = Flask(__name__)
CORS(app)

@app.route("/api/v1/update", methods=['POST'])
def train_model():
    print("TEST")
    data={}
    data['intents']=[]
    data['intents'].append({
        'tag': 'tag',
        'patterns': 'patterns',
        'response': 'response'
    })
    response = {"success":False,"message":"Tag not found"}
    tag = request.json['tag']
    new_pattern = request.json['pattern']
    for intent in intents['intents']:
        print("WHY not", tag, intent['tag'])
        if intent['tag'] in tag:
            print("TEST221")                    
            if new_pattern in intent['patterns']:
                print("TESTfalse")
                response = {"success":False,"message":"Pattern already found in the training data"}
                return response
            else:
                
                print("updating file")
                intent['patterns'].append(new_pattern)
                data = intents
                with open('intents.json', 'w') as training:
                    json.dump(data, training)
                import training
                response = {"success":True,"message":"Model training finished, new model is being used"}
                
    return response

@app.route("/api/v1/addresponse", methods=['POST'])
def add_response():
    print("TEST")
    data={}
    data['intents']=[]
    data['intents'].append({
        'tag': 'tag',
        'patterns': 'patterns',
        'response': 'response'
    })

    tag = request.json['tag']
    new_response = request.json['response']
    for intent in intents['intents']:
        if intent['tag'] in tag:
            if new_response in intent['responses']:
                response = {"success":False,"message":"Response already found in the training data"}
                return response
            else:
                
                print("updating file")
                intent['responses'].append(new_response)
                data = intents
                with open('intents.json', 'w') as training:
                    json.dump(data, training)    
                    
    response = {"success":True,"message":"New response added!"}
    return response

@app.route("/api/v1/smartreply", methods=['POST'])
def classify():
    
    
    ERROR_THRESHOLD = 0.30
    
    USER = ''
    EMAIL = ''
    PHONE = ''
    WEBSITE = ''
    
    try:
        sentence = request.json['sentence']
        USER = request.json['username']
        EMAIL = request.json['email']
        PHONE = request.json['phone']
        WEBSITE = request.json['website']        
        data = pickle.load( open( "smart-reply-data.pkl", "rb" ) )
        words = data['words']
        classes = data['classes']
        model = tf.keras.models.load_model('models/smartmodel')
    except Exception as e:
        response = {"Error please add ":str(e)}
        return response
    
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
                replies.append(list(np.random.choice(intent['responses'], 3, replace=False)))
                crafted_replies = [re.sub(r'{user}', USER, reply) for reply in replies[0]]
                crafted_replies = [re.sub(r'{phone}', PHONE, reply) for reply in crafted_replies]
                crafted_replies = [re.sub(r'{email}', EMAIL, reply) for reply in crafted_replies]
                crafted_replies = [re.sub(r'{website}', WEBSITE, reply) for reply in crafted_replies]
                return_list.append({"intent": classes[r[0]], "probability": str(r[1]), "responses": crafted_replies})
        break # Uncomment this to remove limit responses to 1 intent prediction

    # return tuple of intent and probability
    response = jsonify(return_list)
    return response

# running REST interface
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)


