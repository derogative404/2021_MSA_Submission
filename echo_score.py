# Load libraries 
import json5 as json
import numpy as np
from sklearn.linear_model import Ridge
import joblib
import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Requried init function
def init():
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  # Create a global variable for loading the model
  global model
  model_filename = 'model.pkl'
  model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), model_filename)
  model = joblib.load(model_path)

# 2. Requried run function
def run(request):
  raw_data = json.loads(request)
  data = np.array(raw_data["data"])
  processed_data = list(map(lambda x: preprocess(x), data))
  test = vec_me(processed_data)
  predictions = model.predict(test)
  return predictions.tolist()

def preprocess(sentence):
  """
  This function tokenises the sentences into words, removes unnecesary puntuation 
  that may not be useful to the ML model, removes stop words from each paragraph
  and also lemmantises it, dropping similar words.

  Parameters
  ----------
  sentence (string) - the sentence to be pre-processed

  Returns
  ----------
  String containing the pre-processed words

  Notes
  ----------
  References: 
  https://gist.github.com/ameyavilankar/10347201, 
  https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
  """
  
  # Making all sentences lower case
  sentence = sentence.lower()
  
  # tokenising the sentences and removing the punctuation
  tokeniser = RegexpTokenizer(r'\w+')
  tokens = tokeniser.tokenize(sentence)

  # filtering the tokenised words by removing the stopwords
  filtered_words = filter(lambda token: token not in set(stopwords.words('english')), tokens)

  # setting up  WordNetLemmatizer to reduce words to their stems
  wnl = WordNetLemmatizer()
  
  # setting up a set for lemmatised words
  lemmatised_words = set()

  # lemmatize words
  for word in filtered_words:
    if word.isalpha() == True:
      # Only if the words are alphabets, lemmatize the words
      lemmatised_words.add(wnl.lemmatize(word))

  return ' '.join(lemmatised_words)

def vec_me(data):
  train_df = pd.read_csv("https://raw.githubusercontent.com/stho382/2021_MSA_Submission/main/training.csv")

  # Removing any characters within the text that may not contain alphabelts only
  vectorizer = TfidfVectorizer(stop_words="english")

  #setting up X to give as independent input to the ML model
  X = vectorizer.fit_transform(train_df['processed_text'].values)
  new_data = vectorizer.transform(data)
  return new_data