
# coding: utf-8

# # Textklassifikation mit Vektorisierung in gensim und MLPClassifier 
# Labels (Theme und Disziplinen) sind nicht reduziert (all_labels)
# 
# Autorin: Maria Hartmann

# In[23]:


# Imports
import os
import csv
import time
import numpy as np
import pandas as pd
import scipy.sparse
import multiprocessing # module for multiprocessing 
from sklearn.base import BaseEstimator
import gensim # module for Doc2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from gensim.models import KeyedVectors
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.sklearn_api import D2VTransformer
from sklearn.preprocessing import MultiLabelBinarizer # module to one-hot-encode the labels
from sklearn.pipeline import Pipeline # assemples transormers 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer # module to transform a count matrix to a normalized tf-idf representation
from sklearn.neural_network import MLPClassifier # MultiLayerPerceptron classifier 
from sklearn.model_selection import RandomizedSearchCV # module for paramter optimization

np.random.seed(7) # fix random seed for reproducibility


# Einlesen des Trainings- und Testdatensatzes

# In[2]:


trainset = '../Datasets/all_labels_trainset.csv' 
testset = '../Datasets/all_labels_testset.csv' 

trainset_csv = pd.read_csv(trainset, delimiter=';')
X_train = trainset_csv['text'].values
y_train = trainset_csv['classes'].values
z_train = trainset_csv['filename'].values

testset_csv = pd.read_csv(testset, delimiter=';')
X_test = testset_csv['text'].values
y_test = testset_csv['classes'].values
z_test = testset_csv['filename'].values

# Splitten der Labels pro Blogbeitrag
y_train = [e.split(', ') for e in y_train]
y_test = [e.split(', ') for e in y_test]

# Splitten der Texte in Wörter
X_train = [e.split(' ') for e in X_train]
X_test = [e.split(' ') for e in X_test]


# In[3]:


print(z_train[0])
print(y_train[0])
print(X_train[0])


# Stoppwortfilterung

# In[4]:


def remove_stopwords(X_train):
    #stopwords = open('../Preprocessing/filtered_words_MLP.txt', 'r', encoding='utf-8').read().splitlines()
    stopwords = open('../Preprocessing/filtered_words.txt', 'r', encoding='utf-8').read().splitlines()
    #stopwords = open('../Preprocessing/german_stopwords_plain.txt', 'r', encoding='utf-8').read().splitlines()
    clean_textlist = []
    for text in X_train:
        clean_text = []
        for word in text:
            if word in stopwords:
            #if word in stopwords[9:]: #Die ersten Zeilen enthalten eine Beschreibung
                continue
            else:
                clean_text.append(word)
        clean_textlist.append(clean_text)
    #print(clean_textlist)
    return clean_textlist

X_train = remove_stopwords(X_train)


# k-hot-Kodierung der Labels

# In[5]:


# k-hot-encode labels mit MultiLabelBinarizer
label_encoder = MultiLabelBinarizer()
encoded_y_train = label_encoder.fit_transform(y_train)
encoded_y_test = label_encoder.transform(y_test)
print(encoded_y_train[0])


# In[6]:


print(len(label_encoder.classes_))
for i, element in enumerate(label_encoder.classes_):
    print(i, element)


# Klassifikation der Daten mit gensim

# In[40]:


vectorizer = D2VTransformer(dm=0, window=10, iter=20, size=100, min_count=4, sample=0)


# In[41]:


text_clf = Pipeline([('vect', vectorizer),
                     ('clf', MLPClassifier(hidden_layer_sizes=(4096,1024), validation_fraction=0.1, early_stopping=True, verbose=True, random_state=1))
                    ])


# In[42]:


# train
start = time.time()
text_clf = text_clf.fit(X_train, encoded_y_train)
processing_time = (time.time() - start) / 60


# In[43]:


clf_params = text_clf.get_params()
print(clf_params)


# In[44]:


print(processing_time)


# In[45]:


# predict
predicted = text_clf.predict(X_test)
#predicted_proba = text_clf.predict_proba(X_test)


# In[46]:


# precision is a measure of result relevancy
from sklearn.metrics import precision_score
precision = precision_score(encoded_y_test, predicted, average='samples')
print(precision)


# In[47]:


# recall is a measure of how many truly relevant results are returned
from sklearn.metrics import recall_score
recall = recall_score(encoded_y_test, predicted, average='samples')  
print(recall)


# In[48]:


# F1 score is a weighted average of the precision and recall
from sklearn.metrics import f1_score
f1 = f1_score(encoded_y_test, predicted, average='samples') 
print(f1)


# In[49]:


output = '../MLP/gensim_klein'
if not os.path.exists(output):
    os.makedirs(output)


# In[50]:


# write parameters and scores to file

with open(output+'/MLP_gensim_all_labels_params.txt',"a", encoding="utf8") as params:
    params.write("\n*********************************************************************************************")
    params.write("\nParameters for classification with MLP and vectorization in gensim (all labels):")
    params.write("\n*********************************************************************************************")
    params.write("\n%s" % text_clf.named_steps.vect)
    params.write("\n%s" % text_clf.named_steps.clf)
    #for key, value in clf_params.items():
        #params.write("\n%s: %s" % (key, value))
    params.write("\nclasses: %s" % text_clf.named_steps.clf.n_outputs_)
    params.write("\nlayers: %s" % text_clf.named_steps.clf.n_layers_)
    params.write("\nactivation function output layer: %s" % text_clf.named_steps.clf.out_activation_) 
    params.write("\nepochs: %s" % text_clf.named_steps.clf.n_iter_)
    params.write("\nprocessing time: %s" % processing_time)
    params.write("\nSCORES:")
    params.write("\nprecision: %s" % precision)
    params.write("\nrecall: %s" % recall)
    params.write("\nf1-score: %s" % f1)
    params.write("\n")


# Speicherung der vektrorisierten Daten

# In[51]:


z_train = [e.replace('.txt', '') for e in z_train]
z_test = [e.replace('.txt', '') for e in z_test]
ident_train = [e.replace('_', '.hypotheses.org/') for e in z_train]
ident_test = [e.replace('_', '.hypotheses.org/') for e in z_test]

print(len(ident_train))
print(ident_train[0])


# In[52]:


# vectorize textdata
train_vect = vectorizer.transform(X_train)
test_vect = vectorizer.transform(X_test)

print(train_vect.shape)
print(type(train_vect))


# In[53]:


# convert vectorized textdata to sparse matrix
train_matrix = sparse.csr_matrix(train_vect)
test_matrix = sparse.csr_matrix(test_vect)

train_matrix


# In[54]:


# save filename, classes, textvectors in csv file
# trainset
# speichert vektorisierten Text
output_file_train = 'Datasets/all_labels_train_gensim_sparse_matrix.npz'
scipy.sparse.save_npz('../'+output_file_train, train_matrix)

# speichert filenames und classes
with open('../Datasets/all_labels_gensim_train_idents_labels.csv', 'w', newline='', encoding="utf-8") as traincsv:
    train = csv.writer(traincsv, delimiter = ";")
    train.writerow(["url", "classes", "filename"])
    
    for ident, labels in zip(ident_train, y_train):
        labellist = ", ".join(labels)
        train.writerow([ident, labellist, output_file_train])

# testset
# speichert vektorisierten Text
output_file_test = 'Datasets/all_labels_test_gensim_sparse_matrix.npz'
scipy.sparse.save_npz('../'+output_file_test, test_matrix)

# speichert filenames und classes
with open('../Datasets/all_labels_gensim_test_idents_labels.csv', 'w', newline='', encoding="utf-8") as testcsv:
    test = csv.writer(testcsv, delimiter = ";")
    test.writerow(["url", "classes", "filename"])
    
    for ident, labels in zip(ident_test, y_test):
        labellist = ", ".join(labels)
        test.writerow([ident, labellist, output_file_test])

