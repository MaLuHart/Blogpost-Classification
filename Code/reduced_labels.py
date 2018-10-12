
# coding: utf-8

# # Klassifikation der in scikit-learn vektorisierten Daten (reduced_labels)
# 
# Autorin: Maria Hartmann

# In[1]:


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
from keras.layers import Embedding, Dense, LSTM, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# Einlesen der vektorisierten Daten

# In[2]:


trainset_labels = '../Datasets/reduced_labels_train_idents_labels.csv' 
testset_labels = '../Datasets/reduced_labels_test_idents_labels.csv' 

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


output_dir = "../reduced_labels" 
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


# In[ ]:


loss_values = mlp_clf.loss_curve_
print(loss_values)
pyplot.title('Loss on training data (reduced_labels)')
pyplot.xlabel('epochs')
pyplot.ylabel('loss')
pyplot.plot(loss_values)
pyplot.savefig('%s/all_labels_mlp_plot_loss.png' % output_dir)
pyplot.show()


# In[ ]:


validation_scores = mlp_clf.validation_scores_
print(validation_scores)
pyplot.title('Accuracy on validation data (reduced_labels)')
pyplot.xlabel('epochs')
pyplot.ylabel('acc')
pyplot.plot(validation_scores)
pyplot.savefig('%s/all_labels_mlp_plot_val_acc.png' % output_dir)
pyplot.show()


# # Klassifikation mit Dense-Layer in Keras

# In[24]:


# Keras model 

dense_model = Sequential()
dense_model.add(Dense(4096, input_dim=dim, activation="relu"))
dense_model.add(Dropout(0.4))
dense_model.add(Dropout(0.4))
dense_model.add(Dropout(0.4))
dense_model.add(Dense(1024, activation="relu"))
dense_model.add(Dropout(0.4))
dense_model.add(Dropout(0.4))
dense_model.add(Dropout(0.4))
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
                 ]

# train keras model
batch_size = 32
epochs = 100
dense_start = time.time()
dense_estimator = dense_model.fit(X_train,
                            np.array(encoded_y_train), 
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=callbacks_list,
                            verbose=10, 
                            validation_split=0.1,
                            shuffle=True
                           )

dense_time = (time.time() - dense_start)/60
print("Laufzeit in Minuten:", dense_time)


# In[25]:


# visualize the train and validate loss and accuracy

# plot history for accuracy
pyplot.plot(dense_estimator.history['acc'], label='train')
pyplot.plot(dense_estimator.history['val_acc'], label='test')
pyplot.legend()
pyplot.savefig('%s/reduced_labels_dense_plot_acc.png' % output_dir)
pyplot.show()
pyplot.close()

# plot history for loss
pyplot.plot(dense_estimator.history['loss'], label='train')
pyplot.plot(dense_estimator.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('%s/reduced_labels_dense_plot_loss.png' % output_dir)
pyplot.show()
pyplot.close()


# In[26]:


print(dense_time)
dense_predicted = dense_model.predict_proba(X_test)


# In[27]:


print(dense_predicted[0].round())
print(encoded_y_test[0])


# In[28]:


dense_precision = precision_score(encoded_y_test, dense_predicted.round(), average='samples')
print(dense_precision)
dense_recall = recall_score(encoded_y_test, dense_predicted.round(), average='samples')
print(dense_recall)
dense_f1 = f1_score(encoded_y_test, dense_predicted.round(), average='samples')
print(dense_f1)


# Ohne Dropout:
# 
# Precision: 0.8664668625748569
# Recall: 0.8243970628612984
# F1-Score: 0.8343708633948713
# 
# Mit einer Dropout-Schicht (0,4):
# 
# Precision: 0.868548016058535
# Recall: 0.8362136104246903
# F1-Score: 0.8428356320640372
# 
# Mit drei Dropout-Schichten (0,4):
# 
# Precision: 0.8637551043055952
# Recall: 0.8624904457653405
# F1-Score: 0.8561092500582034

# # Klassifikation mit LSTM-Layer in Keras

# In[37]:


# LSTM in Keras
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=dim, output_dim=128))
lstm_model.add(LSTM(64, activation="relu"))
lstm_model.add(Dropout(0.4))
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

callbacks_list = [EarlyStopping(monitor='val_loss', patience=4, verbose=10),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=10),
                 ]

# train keras model
batch_size = 32
epochs = 100
lstm_start = time.time()
lstm_estimator = lstm_model.fit(X_train,
                            np.array(encoded_y_train), 
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=callbacks_list,
                            verbose=10, 
                            validation_split=0.1,
                            shuffle=True
                           )

lstm_time = (time.time() - lstm_start)/60
print("Laufzeit in Minuten:", lstm_time)


# In[30]:


# visualize the train and validate loss and accuracy

# plot history for accuracy
pyplot.plot(lstm_estimator.history['acc'], label='train')
pyplot.plot(lstm_estimator.history['val_acc'], label='test')
pyplot.legend()
pyplot.savefig('%s/reduced_labels_lstm_plot_acc.png' % output_dir)
pyplot.show()
pyplot.close()

# plot history for loss
pyplot.plot(lstm_estimator.history['loss'], label='train')
pyplot.plot(lstm_estimator.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('%s/reduced_labels_lstm_plot_loss.png' % output_dir)
pyplot.show()
pyplot.close()


# In[33]:


print(lstm_time)
lstm_predicted = lstm_model.predict_proba(X_test)


# In[36]:


print(lstm_predicted[0])


# In[35]:


lstm_precision = precision_score(encoded_y_test, lstm_predicted.round(), average='samples')
print(lstm_precision)
lstm_recall = recall_score(encoded_y_test, lstm_predicted.round(), average='samples')
print(lstm_recall)
lstm_f1 = f1_score(encoded_y_test, lstm_predicted.round(), average='samples')
print(lstm_f1)

