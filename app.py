import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# importing required libraries--modified
import numpy as np 
import pandas as pd

from river import feature_extraction
from river import linear_model
from river import metrics
from river import preprocessing
from river import stats

#import text_processing


app = Flask(__name__)
model = pickle.load(open('PA_model2.pkl', 'rb'))

def text_processing(dataset, Y=None):
    def count_punct(text):
        try:
            count = sum([1 for char in text if char in string.punctuation])
            return round(count/(len(text) - text.count(" ")), 3)*100
        except:
            return 0
        
    dataset['punct%'] = count_punct(dataset['title'])
    
    #Text Cleaning
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', dataset['content'])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    dataset['content']  =  review
    #del dataset['title']
    
    #Similarity = (A.B) / (||A||.||B||) where A and B are vectors.
    def similarity(A,B):
        # tokenization 
        X_list = word_tokenize(A) 
        Y_list = word_tokenize(B) 

        # list of words 
        l1 =[];l2 =[] 

        # remove stop words from the string 
        X_set = {w for w in X_list if not w in stopwords.words('english')} 
        Y_set = {w for w in Y_list if not w in stopwords.words('english')} 

        # form a set containing keywords of both strings 
        rvector = X_set.union(Y_set) 
        for w in rvector: 
            if w in X_set: 
                l1.append(1) # create a vector 
            else: 
                l1.append(0) 
            if w in Y_set:
                l2.append(1) 
            else: 
                l2.append(0) 
        
        c = 0
        # cosine formula 
        for i in range(len(rvector)):
            c+= l1[i]*l2[i]
        try:
            cosine = c / float((sum(l1)*sum(l2))**0.5)
        except:
            cosine = 0
        return cosine
    
    dataset['similarity']=similarity(dataset['title'],dataset['text'])
    
    return dataset


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features = {'text':features[1],'title':features[0]}
    final_features['content']=final_features['title']+final_features['text']
    final_features=text_processing(final_features)
    del final_features['title']
    del final_features['text']
    final_features['length'] = len(str(final_features['content']))
    
    
    prediction = model.predict_one(final_features)
    if prediction==1:
      result='Fake'
    else:
      result='Real'

    #output = round(prediction[0], 2)

    return render_template('index.html',prediction_text='The News is {}'.format(result))


if __name__ == "__main__":
    app.run(debug=True)
