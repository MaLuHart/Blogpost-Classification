
# coding: utf-8

# # Textklassifikation mit Bag-of-Words-Vektorisierung in tf-idf-Repräsentation und kNN-Algorithmus 
# Labels sind auf die Themen reduziert
# 
# Autorin: Maria Hartmann

# In[1]:


# Imports
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer # module to one-hot-encode the labels
from sklearn.pipeline import Pipeline # assemples transormers 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer # module to transform a count matrix to a normalized tf-idf representation
from sklearn.neighbors import KNeighborsClassifier # k-nearest neighbors classifier (supports multi-label classification)
from sklearn.model_selection import RandomizedSearchCV # module for paramter optimization

np.random.seed(7) # fix random seed for reproducibility


# Einlesen des Trainings- und Testdatensatzes

# In[2]:


trainset = '../Datasets/themes_only_trainset.csv' 
testset = '../Datasets/themes_only_testset.csv' 

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

# In[19]:


# best params from randomized search
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,4), max_df=0.9, min_df=0.0)),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', KNeighborsClassifier(n_neighbors=6, weights='distance')),
                   ])


# In[20]:


# train
start = time.time()
text_clf = text_clf.fit(X_train, encoded_y_train)
processing_time = (time.time() - start) / 60


# In[21]:


# predict
predicted = text_clf.predict(X_test)
#predicted_proba = text_clf.predict_proba(X_test)


# In[22]:


clf_params = text_clf.get_params()
print(clf_params)


# In[23]:


# precision is a measure of result relevancy
from sklearn.metrics import precision_score
precision = precision_score(encoded_y_test, predicted, average='samples')
print(precision)


# In[24]:


# recall is a measure of how many truly relevant results are returned
from sklearn.metrics import recall_score
recall = recall_score(encoded_y_test, predicted, average='samples')  
print(recall)


# In[25]:


# F1 score is a weighted average of the precision and recall
from sklearn.metrics import f1_score
f1 = f1_score(encoded_y_test, predicted, average='samples') 
print(f1)


# Parameteroptimierung mit Rastersuche (RandomizedSearch)

# In[6]:


clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', KNeighborsClassifier()),
                    ])


# In[7]:


# parameter tuning with RandomSearch
parameters = {'vect__ngram_range': [(1,1),(1,2),(1,3),(1,4)], 
              'vect__max_df' : (0.7, 0.8, 0.9, 1.0), 
              'vect__min_df' : (0.0, 0.01, 0.05, 0.1),
              'tfidf__use_idf': (True, False),
              'clf__n_neighbors': list(range(1,10,1)),
              'clf__weights' : ('distance', 'uniform')
             }


# In[8]:


# train
rs_clf = RandomizedSearchCV(clf, parameters, n_jobs=1, verbose=10, random_state=1, return_train_score=True, cv=3, n_iter=50)
start = time.time()
rs_clf = rs_clf.fit(X_train, encoded_y_train)
rs_processing_time = (time.time() - start) / 60


# In[9]:


best_score = rs_clf.best_score_
print(best_score)


# In[10]:


best_params = rs_clf.best_params_
print(best_params)


# In[11]:


# predict 
rs_predicted = rs_clf.predict(X_test)
#print(predicted)


# In[12]:


# precision is a measure of result relevancy
from sklearn.metrics import precision_score
rs_precision = precision_score(encoded_y_test, rs_predicted, average='samples')
print(rs_precision)


# In[13]:


# recall is a measure of how many truly relevant results are returned
from sklearn.metrics import recall_score
rs_recall = recall_score(encoded_y_test, rs_predicted, average='samples')  
print(rs_recall)


# In[14]:


# F1 score is a weighted average of the precision and recall
from sklearn.metrics import f1_score
rs_f1 = f1_score(encoded_y_test, rs_predicted, average='samples') 
print(rs_f1)


# Ergebnisse in Dateien speichern

# In[15]:


output = '../kNN'
if not os.path.exists(output):
    os.makedirs(output)
    
timestamp = time.strftime('%Y-%m-%d_%H.%M')


# In[16]:


# write real labels and predictions to file

inverse_prediction = label_encoder.inverse_transform(rs_predicted)
print('PREDICTED:')
print(inverse_prediction[0])
print('TRUE:')
print(y_test[0])

with open(output+'/kNN_themes_only_predictions_%s.txt' % timestamp,"w+", encoding="utf8") as preds:
    preds.write("Predictions from classification with k-nearest-neighbors (themes only):\n\n")
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
    


# In[17]:


# write parameters and scores to file

with open(output+'/kNN_themes_only_params_%s.txt' % timestamp,"w+", encoding="utf8") as params:
    params.write("Parameters for classification with k-nearest-neighbors from randomized search (themes only):")
    params.write("\nprocessing_time: %s" % rs_processing_time)
    for key, value in best_params.items():
        params.write("\n%s: %s" % (key, value))
    params.write("\nbest_score: %s" % best_score)
    params.write("\nprecision: %s" % rs_precision)
    params.write("\nrecall: %s" % rs_recall)
    params.write("\nf1-score: %s" % rs_f1)


# In[18]:


results = rs_clf.cv_results_
df = pd.DataFrame(data=results)
print(df)
df.to_csv(output+'/kNN_themes_only_rs_results_%s.csv' % timestamp, encoding='utf-8')

