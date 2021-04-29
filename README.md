# bakerybot
A keras tensorflow chatbot that has learnt to give responses when asked questions about a bakery.

Pacakages that need to be installed:

-BakerBot.py

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
  
 -chatbotapp.py
 
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
  
  -chatbotGUI.py
  
  import tkinter
  from tkinter import *
  from chatbotapp import chatbot_response

If using a terminal or shell, it is best use these installs;
  pickle-mixin
  python -m nltk.downloader all
  
The python files must be run in this order:
  1. BakerBot.py
  2. chatbotapp.py
  3. chatbotGUI.py

Currently, the chatbot is not able to register long sentences. To get responses, please refer to the intents file as this is what the bot was trained with.
