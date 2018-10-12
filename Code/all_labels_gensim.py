
# coding: utf-8

# # Klassifikation der in gensim vektorisierten Daten (all_labels) 
# 
# Autorin: Maria Hartmann

# In[46]:


# Imports
import os
import time
import pandas as pd
import numpy as np
import scipy.sparse
from matplotlib import pyplot
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.neural_network import MLPClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard


# Einlesen der vektorisierten Daten

# In[2]:


trainset_labels = '../Datasets/all_labels_gensim_train_idents_labels.csv' 
testset_labels = '../Datasets/all_labels_gensim_test_idents_labels.csv' 

trainset_csv = pd.read_csv(trainset_labels, delimiter=';')
y_train = trainset_csv['classes'].values
z_train = trainset_csv['url'].values
train_vectors = trainset_csv['filename'].values

testset_csv = pd.read_csv(testset_labels, delimiter=';')
y_test = testset_csv['classes'].values
z_test = testset_csv['url'].values
test_vectors = testset_csv['filename'].values

# Splitten der Labels pro Blogbeitrag
y_train = [e.split(', ') for e in y_train]
y_test = [e.split(', ') for e in y_test]


# In[3]:


if len(set(train_vectors)) == 1:
    X_train = scipy.sparse.load_npz('../%s' % train_vectors[0])
else:
    print("Error with len(set(train_vectors))")
if len(set(test_vectors)) == 1:
    X_test = scipy.sparse.load_npz('../%s' % test_vectors[0])
else:
    print("Error with len(set(test_vectors))")


# In[4]:


print(z_train[0])
print(y_train[0])
print(test_vectors[0])
print(X_train.shape)
X_train


# In[5]:


dim = X_train.shape[1]
print(dim)


# In[6]:


output_dir = "../all_labels_gensim" 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# k-hot-Kodierung der Labels

# In[7]:


# k-hot-encode labels mit MultiLabelBinarizer
label_encoder = MultiLabelBinarizer()
encoded_y_train = label_encoder.fit_transform(y_train)
encoded_y_test = label_encoder.transform(y_test)
print(encoded_y_train[0])


# In[8]:


print(len(label_encoder.classes_))
for i, element in enumerate(label_encoder.classes_):
    print(i, element)


# # Klassifikation mit MLPClassifier

# In[9]:


mlp_clf = MLPClassifier(hidden_layer_sizes=(4096,1024), validation_fraction=0.1, early_stopping=True, verbose=True, random_state=1)
mlp_start = time.time()
mlp_clf = mlp_clf.fit(X_train, encoded_y_train)
mlp_time = (time.time() - mlp_start)/60


# In[10]:


print(mlp_time)
mlp_predicted = mlp_clf.predict(X_test)


# In[11]:


mlp_precision = precision_score(encoded_y_test, mlp_predicted.round(), average='samples')
print(mlp_precision)
mlp_recall = recall_score(encoded_y_test, mlp_predicted.round(), average='samples')
print(mlp_recall)
mlp_f1 = f1_score(encoded_y_test, mlp_predicted.round(), average='samples')
print(mlp_f1)


# In[13]:


loss_values = mlp_clf.loss_curve_
print(loss_values)
pyplot.title('Loss on training data (all_labels)')
pyplot.xlabel('epochs')
pyplot.ylabel('loss')
pyplot.plot(loss_values)
pyplot.savefig('%s/all_labels_gensim_mlp_plot_loss.png' % output_dir)
pyplot.show()


# In[14]:


validation_scores = mlp_clf.validation_scores_
print(validation_scores)
pyplot.title('Accuracy on validation data (all_labels)')
pyplot.xlabel('epochs')
pyplot.ylabel('acc')
pyplot.plot(validation_scores)
pyplot.savefig('%s/all_labels_gensim_mlp_plot_val_acc.png' % output_dir)
pyplot.show()


# Vergleich zwischen der Vektorisierung in scikit-learn und gensim (Klassifikation mit dem MLPClassifier von all_labels)

# In[18]:


loss_values_scikit_learn = [19.21366700374499, 9.467989909749473, 5.495542813416394, 2.980149101333112, 1.6234586937668352, 0.9450190023217871, 0.6229070227073472, 0.4846545875629756, 0.4053494018358509]
validation_scores_scikit_learn = [0.38457042665108127, 0.45879602571595557, 0.5137346580946814, 0.5406195207481005, 0.5429573348918761, 0.5441262419637639, 0.5417884278199883, 0.5388661601402689, 0.5423728813559322]


# In[19]:


pyplot.title('Loss on training data (all_labels)')
pyplot.xlabel('epochs')
pyplot.ylabel('loss')
pyplot.plot(loss_values, label='gensim')
pyplot.plot(loss_values_scikit_learn, label='scikit-learn')
pyplot.legend()
pyplot.savefig('%s/all_labels_mlp_plot_loss_comparison.png' % output_dir)
pyplot.show()


pyplot.title('Accuracy on validation data (all_labels)')
pyplot.xlabel('epochs')
pyplot.ylabel('acc')
pyplot.plot(validation_scores, label='gensim')
pyplot.plot(validation_scores_scikit_learn, label='scikit-learn')
pyplot.legend()
pyplot.savefig('%s/all_labels_mlp_plot_val_acc_comparison.png' % output_dir)
pyplot.show()


# # Klassifikation mit Dense-Layer in Keras

# In[31]:


count=0


# In[33]:


# MLP in Keras  

dense_model = Sequential()
dense_model.add(Dense(4096, input_dim=dim, activation="relu"))
#dense_model.add(Dropout(0.4))
#dense_model.add(Dropout(0.4))
#dense_model.add(Dropout(0.4))
dense_model.add(Dense(1024, activation="relu"))
#dense_model.add(Dropout(0.4))
#dense_model.add(Dropout(0.4))
#dense_model.add(Dropout(0.4))
dense_model.add(Dense(len(label_encoder.classes_), activation="sigmoid"))

summary = dense_model.summary()
print("\n", summary)
config = dense_model.get_config()
print("\n", config)

# compile keras model
lossfunction = 'binary_crossentropy'
optimizer = "adam"
metrics = ['accuracy']

dense_model.compile(loss=lossfunction, 
              optimizer=optimizer,
              metrics=metrics)

callbacks_list = [EarlyStopping(monitor='val_loss', patience=7, verbose=10),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=10),
                  TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
                 ]

# train keras model
batch_size = 32
epochs = 100
dense_start = time.time()
dense_estimator = dense_model.fit(X_train,
                            encoded_y_train, 
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=callbacks_list,
                            verbose=10, 
                            validation_split=0.1,
                            shuffle=True
                           )

dense_time = (time.time() - dense_start)/60
print("Laufzeit in Minuten:", dense_time)
count+=1


# In[34]:


# visualize the train and validate loss and accuracy

# plot history for accuracy
pyplot.plot(dense_estimator.history['acc'], label='train')
pyplot.plot(dense_estimator.history['val_acc'], label='test')
pyplot.legend()
pyplot.savefig('%s/all_labels_gensim_dense_plot_acc_%s.png' % (output_dir, count))
pyplot.show()
pyplot.close()

# plot history for loss
pyplot.plot(dense_estimator.history['loss'], label='train')
pyplot.plot(dense_estimator.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('%s/all_labels_gensim_dense_plot_loss_%s.png' % (output_dir, count))
pyplot.show()
pyplot.close()


# In[35]:


print(dense_time)
dense_predicted = dense_model.predict_proba(X_test)


# In[36]:


print(dense_predicted[0].round())
print(encoded_y_test[0])


# In[38]:


dense_precision = precision_score(encoded_y_test, dense_predicted.round(), average='samples')
print(dense_precision)
dense_recall = recall_score(encoded_y_test, dense_predicted.round(), average='samples')
print(dense_recall)
dense_f1 = f1_score(encoded_y_test, dense_predicted.round(), average='samples')
print(dense_f1)


# Ohne Dropout:
# 
# Precision: 0.8368304699300492
# Recall: 0.7853174771828347
# F1-Score: 0.7967917446616334
# 
# Mit einer Dropout-Schicht (0,4)
# 
# Precision: 0.8364704953765262
# Recall: 0.8097225342316505
# F1-Score: 0.8126913833496906
# 
# Mit drei Dropout-Schichten (0,4):
# 
# Precision: 0.817464881490127
# Recall: 0.764179040791944
# F1-Score: 0.771810034224439

# # Klassifikation mit LSTM-Layer in Keras

# In[47]:


# LSTM in Keras
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=dim, output_dim=114))
lstm_model.add(LSTM(114))#, dropout=0.4, recurrent_dropout=0.4))
lstm_model.add(Dense(len(label_encoder.classes_), activation="sigmoid"))

summary = lstm_model.summary()
print("\n", summary)
config = lstm_model.get_config()
print("\n", config)

# compile keras model
lossfunction = 'binary_crossentropy'
optimizer = "adam"
metrics = ['accuracy']

lstm_model.compile(loss=lossfunction,
                   optimizer=optimizer,
                   metrics=metrics)

callbacks_list = [EarlyStopping(monitor='val_loss', patience=7, verbose=10),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=10),
                  TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
                 ]

# train keras model
batch_size = 32
epochs = 100
lstm_start = time.time()
lstm_estimator = lstm_model.fit(X_train,
                            encoded_y_train, 
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=callbacks_list,
                            verbose=10, 
                            validation_split=0.1,
                            shuffle=True
                           )

lstm_time = (time.time() - lstm_start)/60
print("Laufzeit in Minuten:", lstm_time)


# In[ ]:


# visualize the train and validate loss and accuracy

# plot history for accuracy
pyplot.plot(lstm_estimator.history['acc'], label='train')
pyplot.plot(lstm_estimator.history['val_acc'], label='test')
pyplot.legend()
pyplot.savefig('%s/all_labels_gesim_lstm_plot_acc.png' % output_dir)
pyplot.show()
pyplot.close()

# plot history for loss
pyplot.plot(lstm_estimator.history['loss'], label='train')
pyplot.plot(lstm_estimator.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('%s/all_labels_gensim_lstm_plot_loss.png' % output_dir)
pyplot.show()
pyplot.close()


# In[ ]:


print(lstm_time)
lstm_predicted = lstm_clf.predict_proba(X_test)


# In[ ]:


lstm_precision = precision_score(encoded_y_test, lstm_predicted.round(), average='samples')
print(lstm_precision)
lstm_recall = recall_score(encoded_y_test, lstm_predicted.round(), average='samples')
print(lstm_recall)
lstm_f1 = f1_score(encoded_y_test, lstm_predicted.round(), average='samples')
print(lstm_f1)

