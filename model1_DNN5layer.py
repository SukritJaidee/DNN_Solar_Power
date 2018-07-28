
# coding: utf-8

# # EE_chula Model_1
# Implementation
# <br>DNN = 5 layer<br>
# hidden layer = 4 layer<br>
# layer1 = 200 unit, activation = relu<br>
# layer2 = 100 unit, activation = relu<br>
# layer3 = 50 unit, activation = relu<br>
# layer4 = 25 unit, activation = relu<br>
# loss=mean_squared_error(MSE) = 0.0033954655623835163
# # 7/28/2018

# # import library

# In[55]:


import os
import keras
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# # load data for train

# In[57]:


train = pd.read_csv('mydata/dataset_rev4_train.csv')
#buffer datetime
buffer_datetime_train = train.datetime
#remove object
train = train.select_dtypes(exclude=['object'])
#replace misssing value
train.fillna(0,inplace=True)


# # load data for test

# In[59]:


test = pd.read_csv('mydata/dataset_rev4_test.csv')
#buffer datetime
buffer_datetime_test = test.datetime
#remove object
test = test.select_dtypes(exclude=['object'])
#replace misssing value
test.fillna(0,inplace=True)


# In[60]:


print('dimension of train:', train.shape)
print('dimension of test:', test.shape)


# In[61]:


print("features:",list(train.columns))


# # remove outlier

# In[63]:


from sklearn.ensemble import IsolationForest
clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(train)
y_noano = clf.predict(train)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
y_noano[y_noano['Top'] == 1].index.values

train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
train.reset_index(drop = True, inplace = True)
print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
print("Number of rows without outliers:", train.shape[0])


# # Normalize

# In[66]:


import warnings
warnings.filterwarnings('ignore')

col_train = list(train.columns)
col_train_bis = list(train.columns)

col_train_bis.remove('P')

mat_train = np.matrix(train)
mat_test  = np.matrix(test)

mat_new = np.matrix(train.drop('P',axis = 1))
mat_y = np.array(train.P).reshape((118427,1))

prepro_y = MinMaxScaler()
prepro_y.fit(mat_y)

prepro = MinMaxScaler()
prepro.fit(mat_train)

prepro_test = MinMaxScaler()
prepro_test.fit(mat_new)

train = pd.DataFrame(prepro.transform(mat_train),columns = col_train)
test  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)


# # create training_set and prediction_set

# In[68]:


# List of features
COLUMNS = col_train #column train (x train)
FEATURES = col_train_bis  #column train-label (x test)
LABEL = "P"

# Columns
feature_cols = FEATURES #(x test)

# Training set and Prediction set with the features to predict
training_set = train[COLUMNS] #column train (x train)
prediction_set = train.P # column P


# In[73]:


print(type(training_set))
print(type(prediction_set))


# # create x_train and Test 

# In[70]:


x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES] , prediction_set, test_size=0.33, random_state=42)


# In[71]:


print(type(x_train))
print(type(x_test))
print(type(y_train))
print(type(y_test))


# In[72]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # create training_set

# In[78]:


y_train = pd.DataFrame(y_train, columns = [LABEL])
training_set = pd.DataFrame(x_train, columns = FEATURES).merge(y_train, left_index = True, right_index = True)
training_sub = training_set[col_train]


# # create testing_set

# In[80]:


y_test = pd.DataFrame(y_test, columns = [LABEL])
testing_set = pd.DataFrame(x_test, columns = FEATURES).merge(y_test, left_index = True, right_index = True)


# # create model

# In[84]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

seed = 7
np.random.seed(seed)

# Model
model = Sequential()
model.add(Dense(200, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(25, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

# Compile model
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta())

feature_cols = training_set[FEATURES]
labels = training_set[LABEL].values

model.fit(np.array(feature_cols), np.array(labels), epochs=100, batch_size=10)


# # Evaluation on the test set created by train_test_split

# In[85]:


score = model.evaluate(np.array(feature_cols), np.array(labels))


# In[87]:


print(model.summary())


# In[92]:


score


# # Predictions

# In[93]:


feature_cols_test = testing_set[FEATURES]
labels_test = testing_set[LABEL].values

y = model.predict(np.array(feature_cols_test))
predictions = list(itertools.islice(y, testing_set.shape[0]))


# # inverse_transform

# In[97]:


predictions = prepro_y.inverse_transform(np.array(predictions).reshape(39081,1))


# In[98]:


reality = pd.DataFrame(prepro.inverse_transform(testing_set), columns = [COLUMNS]).P


# # Plot real vs predict

# In[99]:


matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 

fig, ax = plt.subplots(figsize=(50, 40))

plt.style.use('ggplot')
plt.plot(predictions, reality.values, 'ro')
plt.xlabel('Predictions', fontsize = 30)
plt.ylabel('Reality', fontsize = 30)
plt.title('Predictions x Reality on dataset Test', fontsize = 30)
ax.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()


# # Predict on unseen data

# In[102]:


y_predict = model.predict(np.array(test))

def to_submit(pred_y,name_out):
    y_predict = list(itertools.islice(pred_y, test.shape[0]))
    y_predict = pd.DataFrame(prepro_y.inverse_transform(np.array(y_predict).reshape(len(y_predict),1)), columns = ['P'])
    y_predict = y_predict.join(buffer_datetime_test)
    y_predict.to_csv(name_out + '.csv',index=False)
    
to_submit(y_predict, "Predict_model1")

