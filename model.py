# importing required libraries--modified
import numpy as np 
import pandas as pd
from river import feature_extraction
from river import linear_model
from river import metrics
from river import preprocessing

import nltk
import re
import string
nltk.download("stopwords")
nltk.download("punkt")
stopwords = nltk.corpus.stopwords.words('english')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from river import compose
from river import ensemble
from river import evaluate
from river import linear_model

from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as acs

df=pd.read_csv('train.csv')
df=df.dropna()
df.reset_index(inplace=True, drop=True)

## Create New feature
df['content']=df['title']+df['text']

length = []
[length.append(len(str(text))) for text in df['content']]
df['length'] = length

## drop unwanted features
df.drop(['id','author'],inplace=True,axis=1)

# seperate the independent and target variables
df_X = df.drop(columns=['label'])
df_y = df['label']

## Divide the dataset into Train and Test--shuffle=True, stratify=df_y
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=0,shuffle=True, stratify=df_y)

X_train.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)
y_train.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)


X_train_dict=X_train.to_dict('records')
X_test_dict=X_test.to_dict('records')

train_tuple=[]
for i in range(len(X_train)):
    train_tuple.append((X_train_dict[i],y_train[i]))

test_tuple=[]
for i in range(len(X_test)):
    test_tuple.append((X_test_dict[i],y_test[i]))

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

train=train_tuple[:]

test=test_tuple[:]

#Passive Aggressive Classifier
PA_model = compose.Pipeline(('features', compose.TransformerUnion(('pipe1',
    compose.Pipeline(('select_numeric_features', compose.Select('length','punct%','similarity')), ('scale', preprocessing.MinMaxScaler()))),('pipe2',
    compose.Pipeline(('select_text_features', compose.Select('content')), 
                     ('tfidf', feature_extraction.TFIDF(on='content')))))),('modeling', linear_model.PAClassifier()))

metric = metrics.ROCAUC()
train1=train[:]
PA_score1=[]
y_pred_l1=[]
y_l1=[]
for x, y in train1:
    x=text_processing(x)
    y_pred = PA_model.predict_one(x)
    y_pred_l1.append(y_pred)
    y_l1.append(y)
    PA_model.learn_one(x, y)
    metric.update(y, y_pred)
    PA_score1.append(float(str(metric).split(':')[1]))
print("Training_",metric)


import matplotlib.pyplot as plt
plt.plot(PA_score1, label='Passive Aggressive Classifier')
plt.title('ROCAUC-Passive Aggressive Classifier-Training')
plt.xlabel('Sample Size')
plt.ylabel('ROCAUC')
plt.legend()
plt.show()

precision, recall, fscore, train_support = score(y_l1, y_pred_l1, pos_label=1, average='binary')
print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(y_l1,y_pred_l1), 3)))
import seaborn as sns
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_l1, y_pred_l1)
class_label = [1,0]
df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
sns.heatmap(df_cm, annot=True, fmt='d')
plt.title("Confusion Matrix-Passive Aggressive Classifier-Trainig")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# save
with open('PA_model1.pkl', 'wb') as f:
    pickle.dump(PA_model, f)


test2=test[:]
PA_score3=[]
y_pred_l3=[]
y_l3=[]
for x, y in test2:
    x=text_processing(x)
    y_pred = PA_model.predict_one(x)
    y_pred_l3.append(y_pred)
    y_l3.append(y)
    PA_model.learn_one(x, y)
    metric.update(y, y_pred)
    PA_score3.append(float(str(metric).split(':')[1]))

print("Validation_With_Learning_",metric)

import matplotlib.pyplot as plt
plt.plot(PA_score3, label='Passive Aggressive Classifier')
plt.title('ROCAUC-Passive Aggressive Classifier-Validation')
plt.xlabel('Sample Size')
plt.ylabel('ROCAUC')
plt.legend()
plt.show()

precision, recall, fscore, train_support = score(y_l3, y_pred_l3, pos_label=1, average='binary')
print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(y_l3,y_pred_l3), 3)))
import seaborn as sns
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_l3, y_pred_l3)
class_label = [1,0]
df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
sns.heatmap(df_cm, annot=True, fmt='d')
plt.title("Confusion Matrix-Passive Aggressive Classifier-Validation")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# save
with open('PA_model2.pkl', 'wb') as f:
    pickle.dump(PA_model, f)


model = pickle.load(open('PA_model2.pkl','rb'))
z={'text': 'WASHINGTON (Reuters) - The special counsel investigation of links between Russia and President Trump’s 2016 election campaign should continue without interference in 2018, despite calls from some Trump administration allies and Republican lawmakers to shut it down, a prominent Republican senator said on Sunday. Lindsey Graham, who serves on the Senate armed forces and judiciary committees, said Department of Justice Special Counsel Robert Mueller needs to carry on with his Russia investigation without political interference. “This investigation will go forward. It will be an investigation conducted without political influence,” Graham said on CBS’s Face the Nation news program. “And we all need to let Mr. Mueller do his job. I think he’s the right guy at the right time.”  The question of how Russia may have interfered in the election, and how Trump’s campaign may have had links with or co-ordinated any such effort, has loomed over the White House since Trump took office in January. It shows no sign of receding as Trump prepares for his second year in power, despite intensified rhetoric from some Trump allies in recent weeks accusing Mueller’s team of bias against the Republican president. Trump himself seemed to undercut his supporters in an interview last week with the New York Times in which he said he expected Mueller was “going to be fair.”    Russia’s role in the election and the question of possible links to the Trump campaign are the focus of multiple inquiries in Washington. Three committees of the Senate and the House of Representatives are investigating, as well as Mueller, whose team in May took over an earlier probe launched by the U.S. Federal Bureau of Investigation (FBI). Several members of the Trump campaign and administration have been convicted or indicted in the investigation.  Trump and his allies deny any collusion with Russia during the campaign, and the Kremlin has denied meddling in the election. Graham said he still wants an examination of the FBI’s use of a dossier on links between Trump and Russia that was compiled by a former British spy, Christopher Steele, which prompted Trump allies and some Republicans to question Mueller’s inquiry.   On Saturday, the New York Times reported that it was not that dossier that triggered an early FBI probe, but a tip from former Trump campaign foreign policy adviser George Papadopoulos to an Australian diplomat that Russia had damaging information about former Trump rival Hillary Clinton.  “I want somebody to look at the way the Department of Justice used this dossier. It bothers me greatly the way they used it, and I want somebody to look at it,” Graham said. But he said the Russia investigation must continue. “As a matter of fact, it would hurt us if we ignored it,” he said. ',
 'title': "Senior U.S. Republican senator: 'Let Mr. Mueller do his job'"}
z=text_processing(z)
print(z)
print(model.predict_one(z))
