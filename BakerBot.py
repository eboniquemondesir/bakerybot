#code created with help from a project created by the DataFlair Team - https://data-flair.training/blogs/python-chatbot-project/
#import the necessary packages
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

#creating arrays for json data - patterns, responses and tags will be put into arrays
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

#iterate through patterns and tokenize the sentences using nltk word tokenizer, create a list of classes for tags
for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the collection
        documents.append((w, intent['tag']))

        #add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#lemmatize, lower each word, remove duplicates, create pickle files to store python objects to predict
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
#sort the classes
classes = sorted(list(set(classes)))
#documents - patterns and intents combined
print (len(documents), "documents")
#classes are the intents
print (len(classes), "classes", classes)
#words are all the words in the vocabulary
print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#training and testing data needs to be created - convert data into numbers so computer can comprehend

#create training data 
training = []
#create empty array for output
output_empty = [0] * len(classes)
#training set, bag of words for each sentence
for doc in documents:
    #initialise bag of words
    bag = []
    #list of tokenized words for the pattern
    pattern_words = doc[0]
    #lemmatize each word - attempt to represent related words by creating a base word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    #create our bag of words array with 1, if word match is found in pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    #output is '0' for each tag and '1' for current tag - for each pattern
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
#shuffle features and turn into numpy array
random.shuffle(training)
training = np.array(training)
#create train and test lists. x = patterns, y = intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data has been created")

#create model with 3 layers. layer 1 - 128 neurons, layer 2 - 64 neurons, layer 3 - output layer contains number of neurons
#equal to number of intents to predict output intent
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#compile the model. stochastic gradient decent with nesterov accelerated gradient - gives good results
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('bakebot_model.tf', hist)

print("model created")