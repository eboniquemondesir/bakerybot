#code created with help from a project created by the DataFlair Team - https://data-flair.training/blogs/python-chatbot-project/
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

#load the trained model and use a GUI that predicts responses from the bot - import necessart packages
from keras.models import load_model
model = load_model('bakebot_model.tf', compile=False)
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

#predcit the class, create functions that perform text preprocessing and then predict class
def clean_up_sentence(sentence):
    #tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    #stem each word - create short version of word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
#return bag of words - 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    #tokenize pattern
    sentence_words = clean_up_sentence(sentence)
    #bag of words = matrix of N words, vocab matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                #assign 1 if current word is in vocab position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    #filter out predictions below the threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    #sort by the strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

#get random response from list of intents after predicting the class
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res