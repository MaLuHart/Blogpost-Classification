
# coding: utf-8

# # Textklassifikation mit Vektorisierung in scikit-learn und MLPClassifier 
# Labels sind auf Disziplinen reduziert (disciplines_only)
# 
# Autorin: Maria Hartmann

# In[1]:


# Imports
import os
import time
import csv
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.preprocessing import MultiLabelBinarizer # module to one-hot-encode the labels
from sklearn.pipeline import Pipeline # assemples transormers 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer # module to transform a count matrix to a normalized tf-idf representation
from sklearn.neural_network import MLPClassifier # MultiLayerPerceptron classifier 
from sklearn.model_selection import RandomizedSearchCV # module for paramter optimization
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

np.random.seed(7) # fix random seed for reproducibility


# Einlesen des Trainings- und Testdatensatzes

# In[2]:


trainset = '../Datasets/disciplines_only_trainset.csv' 
testset = '../Datasets/disciplines_only_testset.csv' 

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


# In[3]:


print(z_train[0])
print(y_train[0])
print(X_train[0])


# k-hot-Kodierung der Labels

# In[4]:


# k-hot-encode labels mit MultiLabelBinarizer
label_encoder = MultiLabelBinarizer()
encoded_y_train = label_encoder.fit_transform(y_train)
encoded_y_test = label_encoder.transform(y_test)
print(encoded_y_train[0])


# In[5]:


print(len(label_encoder.classes_))
for i, element in enumerate(label_encoder.classes_):
    print(i, element)


# Vektorisierung und Klassifikation der Daten mit scikit-learn

# In[6]:


max_features = 10000
stopwords = open('../Preprocessing/filtered_words.txt','r', encoding='utf-8').read().splitlines()
vectorizer = CountVectorizer(ngram_range=(1,1),  max_features=max_features, stop_words=stopwords)
tfidf_transformer = TfidfTransformer(use_idf=True)


# In[67]:


"""# first try with best params for vect and tfidf from kNN classification
text_clf = Pipeline([#('vect', CountVectorizer(ngram_range=(1,4), max_df=0.9, min_df=0.01)),#min_df=0.0 auf min_df=0.01 geändert
                     ('vect', CountVectorizer(ngram_range=(1,4), max_features=max_features)),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', MLPClassifier(hidden_layer_sizes=(1024,512), max_iter=500, validation_fraction=0.1, early_stopping=True, verbose=True, random_state=1)),
                    ])"""


# In[17]:


text_clf = Pipeline([('vect', vectorizer),
                     ('tfidf', tfidf_transformer),
                     ('clf', MLPClassifier(hidden_layer_sizes=(2048, 512), tol=0.0001, early_stopping=True, validation_fraction=0.1, verbose=True, random_state=1))
                    ])


# In[18]:


# train
start = time.time()
text_clf = text_clf.fit(X_train, encoded_y_train)
processing_time = (time.time() - start) / 60


# In[19]:


clf_params = text_clf.get_params()
print(clf_params)


# In[20]:


# predict
predicted = text_clf.predict(X_test)
#predicted_proba = text_clf.predict_proba(X_test)


# In[21]:


# precision is a measure of result relevancy
precision = precision_score(encoded_y_test, predicted, average='samples')
print(precision)


# In[22]:


# recall is a measure of how many truly relevant results are returned
recall = recall_score(encoded_y_test, predicted, average='samples')  
print(recall)


# In[23]:


# F1 score is a weighted average of the precision and recall
f1 = f1_score(encoded_y_test, predicted, average='samples') 
print(f1)


# In[24]:


output = '../MLP'
if not os.path.exists(output):
    os.makedirs(output)
    
timestamp = time.strftime('%Y-%m-%d_%H.%M')


# In[25]:


"""# write first parameters and scores to file

#with open(output+'/MLP_disciplines_only_first_params.txt',"w+", encoding="utf8") as params:
with open(output+'/MLP_disciplines_only_first_params_max_features.txt',"w+", encoding="utf8") as params:
    params.write("First parameters for classification with MLP (disciplines only):")
    params.write("\nprocessing_time: %s" % processing_time)
    for key, value in clf_params.items():
        params.write("\n%s: %s" % (key, value))
    params.write("\nactivation function output layer: %s" %  text_clf.named_steps.clf.out_activation_)    
    params.write("\nprecision: %s" % precision)
    params.write("\nrecall: %s" % recall)
    params.write("\nf1-score: %s" % f1)"""


# In[26]:


# write parameters and scores to file

with open(output+'/MLP_disciplines_only_params.txt',"a", encoding="utf8") as params:
    params.write("\n*********************************************************************************************")
    params.write("\nParameters for classification with MLP (disciplines only):")
    params.write("\n*********************************************************************************************")
    params.write("\n%s" % text_clf.named_steps.vect)
    params.write("\n%s" % text_clf.named_steps.tfidf)
    params.write("\n%s" % text_clf.named_steps.clf)
    #for key, value in clf_params.items():
        #params.write("\n%s: %s" % (key, value))
    params.write("\nclasses: %s" % text_clf.named_steps.clf.n_outputs_)
    params.write("\nlayers: %s" % text_clf.named_steps.clf.n_layers_)
    params.write("\nactivation function output layer: %s" %  text_clf.named_steps.clf.out_activation_) 
    params.write("\nepochs: %s" % text_clf.named_steps.clf.n_iter_)
    params.write("\nprocessing time: %s" % processing_time)
    params.write("\nSCORES:")
    params.write("\nprecision: %s" % precision)
    params.write("\nrecall: %s" % recall)
    params.write("\nf1-score: %s" % f1)
    params.write("\n")


# In[27]:


# write real labels and predictions to file

inverse_prediction = label_encoder.inverse_transform(predicted)
print('PREDICTED:')
print(inverse_prediction[0])
print('TRUE:')
print(y_test[0])

with open(output+'/MLP_disciplines_only_predictions.txt',"w+", encoding="utf8") as preds:
    preds.write("Predictions from classification with Multi-Layer-Perzeptron and vectorization in scikit-learn (disciplines only):\n\n")
    for ident, label, pred in zip(z_test, y_test, inverse_prediction):
        label = sorted(label)
        pred = sorted(pred)
        preds.write(ident)
        preds.write('\n')
        preds.write('TRUE: ')
        for element in label:
            preds.write('%s, ' % element)
        preds.write('\n')
        preds.write('PRED: ')
        for element in pred:
            preds.write('%s, ' % element)
        preds.write('\n')
        preds.write('\n*********************\n')
    


# Speicherung der vektorisierten Textdaten

# In[28]:


z_train = [e.replace('.txt', '') for e in z_train]
z_test = [e.replace('.txt', '') for e in z_test]
ident_train = [e.replace('_', '.hypotheses.org/') for e in z_train]
ident_test = [e.replace('_', '.hypotheses.org/') for e in z_test]

print(len(ident_train))
print(ident_train[0])


# In[29]:


# vectorize textdata
train_vect = vectorizer.transform(X_train)
train_tfidf = tfidf_transformer.transform(train_vect)
print(train_tfidf.shape)

test_vect = vectorizer.transform(X_test)
test_tfidf = tfidf_transformer.transform(test_vect)


# In[30]:


print(type(test_tfidf))
train_tfidf


# In[31]:


# save filename, classes, textvectors in csv file
# trainset
# speichert vektorisierten Text
output_file_train = 'Datasets/disciplines_only_train_scikit-learn_sparse_matrix.npz'
scipy.sparse.save_npz('../'+output_file_train, train_tfidf)

# speichert filenames und classes
with open('../Datasets/disciplines_only_train_idents_labels.csv', 'w', newline='', encoding="utf-8") as traincsv:
    train = csv.writer(traincsv, delimiter = ";")
    train.writerow(["url", "classes", "filename"])
    
    for ident, labels in zip(ident_train, y_train):
        labellist = ", ".join(labels)
        train.writerow([ident, labellist, output_file_train])

# testset
# speichert vektorisierten Text
output_file_test = 'Datasets/disciplines_only_test_scikit-learn_sparse_matrix.npz'
scipy.sparse.save_npz('../'+output_file_test, test_tfidf)

# speichert filenames und classes
with open('../Datasets/disciplines_only_test_idents_labels.csv', 'w', newline='', encoding="utf-8") as testcsv:
    test = csv.writer(testcsv, delimiter = ";")
    test.writerow(["url", "classes", "filename"])
    
    for ident, labels in zip(ident_test, y_test):
        labellist = ", ".join(labels)
        test.writerow([ident, labellist, output_file_test])


# Parameteroptimierung mit Rastersuche (RandomizedSearch)

# In[65]:


clf = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MLPClassifier(validation_fraction=0.1, early_stopping=True, verbose=True, random_state=1)),
                ])


# In[66]:


# parameter tuning with RandomSearch
stopwords = open('../Preprocessing/filtered_words.txt', 'r', encoding='utf-8').read().splitlines()
rs_parameters = {'vect__ngram_range': [(1,1),(1,2),(1,3),(1,4)], 
                 #'vect__max_df' : (0.7, 0.8, 0.85, 0.9, 0.95), #1.0 
                 #'vect__min_df' : (0.01, 0.025, 0.05, 0.075, 0.1, 0.2), #0.0
                 'vect__max_features': (100000,7500,50000,25000,10000,5000,2500,1000,500,300,100), 
                 'tfidf__use_idf': (True, False),
                 'clf__hidden_layer_sizes': ((2048,1024),(2048,512),(1024,512),(512,128),(4096,1024),(4096,512),(2048,1024,512),(1024,512,128))
                }


# In[68]:


# train
rs_clf = RandomizedSearchCV(clf, rs_parameters, cv=2, n_iter=10, n_jobs=1, verbose=10, random_state=1)
start = time.time()
rs_clf = rs_clf.fit(X_train, encoded_y_train)
rs_processing_time = (time.time() - start) / 60


# In[69]:


best_score = rs_clf.best_score_
print(best_score)


# In[70]:


best_params = rs_clf.best_params_
print(best_params)


# In[71]:


rs_clf_params = rs_clf.get_params()
print(rs_clf_params)


# In[72]:


# predict 
rs_predicted = rs_clf.predict(X_test)
#print(predicted)


# In[74]:


# precision is a measure of result relevancy
rs_precision = precision_score(encoded_y_test, rs_predicted, average='samples')
print(rs_precision)


# In[75]:


# recall is a measure of how many truly relevant results are returned
rs_recall = recall_score(encoded_y_test, rs_predicted, average='samples')  
print(rs_recall)


# In[76]:


# F1 score is a weighted average of the precision and recall
rs_f1 = f1_score(encoded_y_test, rs_predicted, average='samples') 
print(rs_f1)
#49,46


# In[77]:


print(classification_report(encoded_y_test, rs_predicted))


# Ergebnisse in Dateien speichern

# In[78]:


output = '../MLP'
if not os.path.exists(output):
    os.makedirs(output)
    
timestamp = time.strftime('%Y-%m-%d_%H.%M')


# In[79]:


# write real labels and predictions to file

inverse_prediction = label_encoder.inverse_transform(rs_predicted)
print('PREDICTED:')
print(inverse_prediction[0])
print('TRUE:')
print(y_test[0])

with open(output+'/MLP_disciplines_only_rs_predictions_%s.txt' % timestamp,"w+", encoding="utf8") as preds:
    preds.write("Predictions from classification with Multi-Layer-Perzeptron and vectorization in scikit-learn (disciplines only):\n\n")
    for ident, label, pred in zip(z_test, y_test, inverse_prediction):
        label = sorted(label)
        pred = sorted(pred)
        preds.write(ident)
        preds.write('\n')
        preds.write('TRUE: ')
        for element in label:
            preds.write('%s, ' % element)
        preds.write('\n')
        preds.write('PRED: ')
        for element in pred:
            preds.write('%s, ' % element)
        preds.write('\n')
        preds.write('\n*********************\n')
    


# In[80]:


# write parameters and scores to file

with open(output+'/MLP_disciplines_only_rs_params_%s.txt' % timestamp,"w+", encoding="utf8") as params:
    params.write("Parameters for classification with Multi-Layer-Perceptron and vectorization in scikit-learn from randomized search (disciplines only):")
    params.write("\nprocessing_time: %s" % rs_processing_time)
    params.write("\nparams:")
    for key, value in rs_clf_params.items():
        params.write("\n%s: %s" % (key, value))
    params.write("\nbest params:")
    for key, value in best_params.items():
        params.write("\n%s: %s" % (key, value))
    params.write("\nbest_score: %s" % best_score)
    params.write("\nprecision: %s" % rs_precision)
    params.write("\nrecall: %s" % rs_recall)
    params.write("\nf1-score: %s" % rs_f1)


# In[81]:


results = rs_clf.cv_results_
df = pd.DataFrame(data=results)
print(df)
df.to_csv(output+'/MLP_disciplines_only_rs_results_%s.csv' % timestamp, encoding='utf-8')

