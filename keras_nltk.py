# nltk module
import nltk
from nltk.stem.lancaster import LancasterStemmer

# module we need for Tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import pandas as pd
import pickle
import random
import json

# create a stemmer
stemmer = LancasterStemmer()

# load intents
# import our chat-bot intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)


words = []
classes = []
documents = []
ignore_words = []
