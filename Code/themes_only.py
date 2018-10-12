
# coding: utf-8

# # Klassifikation der in scikit-learn vektorisierten Daten (themes_only)
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


trainset_labels = '../Datasets/themes_only_train_idents_labels.csv' 
testset_labels = '../Datasets/themes_only_test_idents_labels.csv' 

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


output_dir = "../themes_only" 
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


mlp_clf = MLPClassifier(hidden_layer_sizes=(2048, 512), validation_fraction=0.1, early_stopping=True, verbose=True, random_state=1)
mlp_start = time.time()
mlp_clf = mlp_clf.fit(X_train, encoded_y_train)
mlp_time = (time.time() - mlp_start)/60


# In[ ]:


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
pyplot.title('Loss on training data (themes_only)')
pyplot.xlabel('epochs')
pyplot.ylabel('loss')
pyplot.plot(loss_values)
pyplot.savefig('%s/themes_only_mlp_plot_loss.png' % output_dir)
pyplot.show()


# In[ ]:


validation_scores = mlp_clf.validation_scores_
print(validation_scores)
pyplot.title('Accuracy on validation data (themes_only)')
pyplot.xlabel('epochs')
pyplot.ylabel('acc')
pyplot.plot(validation_scores)
pyplot.savefig('%s/themes_only_mlp_plot_val_acc.png' % output_dir)
pyplot.show()


# # Klassifikation mit Dense-Layer in Keras

# In[25]:


# Keras model 

dense_model = Sequential()
dense_model.add(Dense(2048, input_dim=dim, activation="relu"))
dense_model.add(Dropout(0.4))
dense_model.add(Dropout(0.4))
dense_model.add(Dropout(0.4))
dense_model.add(Dense(512, activation="relu"))
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


# In[26]:


# visualize the train and validate loss and accuracy

# plot history for accuracy
pyplot.plot(dense_estimator.history['acc'], label='train')
pyplot.plot(dense_estimator.history['val_acc'], label='test')
pyplot.legend()
pyplot.savefig('%s/themes_only_dense_plot_acc.png' % output_dir)
pyplot.show()
pyplot.close()

# plot history for loss
pyplot.plot(dense_estimator.history['loss'], label='train')
pyplot.plot(dense_estimator.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('%s/themes_only_dense_plot_loss.png' % output_dir)
pyplot.show()
pyplot.close()


# In[27]:


print(dense_time)
dense_predicted = dense_model.predict_proba(X_test)


# In[28]:


print(dense_predicted[0].round())
print(encoded_y_test[0])


# In[29]:


dense_precision = precision_score(encoded_y_test, dense_predicted.round(), average='samples')
print(dense_precision)
dense_recall = recall_score(encoded_y_test, dense_predicted.round(), average='samples')
print(dense_recall)
dense_f1 = f1_score(encoded_y_test, dense_predicted.round(), average='samples')
print(dense_f1)


# Ohne Dropout:
# 
# Precision: 0.857538569424965
# Recall: 0.8122954651706404
# F1-Score: 0.8195001507764481
# 
# Mit einer Dropout-Schicht (0,4):
# 
# Precision: 0.8632226897304036
# Recall: 0.829153031011376
# F1-Score: 0.8331697223352202
# 
# Mit drei Dropout-Schichten (0,4):
# 
# Precision: 0.8671770297646876
# Recall: 0.8522907900888267
# F1-Score: 0.8490243043538975

# # Klassifikation mit LSTM-Layer in Keras

# In[9]:


# reshape input data from 2D to 3D for MLP input layer
def reshape_input(text_vectors):
    text_vectors = scipy.sparse.csr_matrix.toarray(text_vectors)
    data = np.expand_dims(text_vectors, axis=2)
    return data

# reshape vectorized trainingdata (X_train)
X_train_3d = reshape_input(X_train)
# reshape vectorized testdata (X_test)
X_test_3d = reshape_input(X_test)

print("shape X_train:", X_train.shape)
print("newshape X_train:", X_train_3d.shape)
print("newshape X_test:", X_test_3d.shape)


# In[18]:


print(type(X_train_3d))
print(len(X_train_3d))
print(type(X_train_3d[0]))
print(len(X_train_3d[0]))
print(type(X_train_3d[0][0]))
print(len(X_train_3d[0][0]))
print(type(X_train_3d[0][1]))
print(len(X_train_3d[0][1]))
print(type(X_train_3d[0][2]))


# In[23]:


# LSTM in Keras
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=dim, output_dim=64))
lstm_model.add(LSTM(32, activation="relu"))
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


# In[24]:


# visualize the train and validate loss and accuracy

# plot history for accuracy
pyplot.plot(lstm_estimator.history['acc'], label='train')
pyplot.plot(lstm_estimator.history['val_acc'], label='test')
pyplot.legend()
pyplot.savefig('%s/themes_only_lstm_plot_acc.png' % output_dir)
pyplot.show()
pyplot.close()

# plot history for loss
pyplot.plot(lstm_estimator.history['loss'], label='train')
pyplot.plot(lstm_estimator.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('%s/themes_only_lstm_plot_loss.png' % output_dir)
pyplot.show()
pyplot.close()


# In[18]:


#print(lstm_time)
lstm_predicted = lstm_model.predict_proba(X_test)


# In[19]:


lstm_precision = precision_score(encoded_y_test, lstm_predicted.round(), average='samples')
print(lstm_precision)
lstm_recall = recall_score(encoded_y_test, lstm_predicted.round(), average='samples')
print(lstm_recall)
lstm_f1 = f1_score(encoded_y_test, lstm_predicted.round(), average='samples')
print(lstm_f1)

