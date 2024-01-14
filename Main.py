#!/usr/bin/env python
# coding: utf-8

# # Do we need more bikes? Project in ML
# 
# 
# Capital Bikeshare is a 24-hour public bicycle-sharing system that serves Washington, D.C., and which offers transportation for thousands of people throughout the city. The problem that arises is that there are certain occasions when, due to various circumstances, there are not as many bikes available as there are demands. In the long term, this situation will result in more people taking the car instead of the bicycle, increasing CO2 emissions in the city. To tackle this situation, the District Department of Transportation in the city wants to know if at certain hours an increase in the number of bikes available will be necessary.
# The goal of the project is to predict whether an increase in the number of bikes is necessary or not based on various temporal and meteorological data. You are expected to use all the knowledge that you have acquired in the course about classification algorithms, to come up with one model that you think is suited for this problem and which you decide to put ‘in production’. This model will then be tested against a test set made available after peer review.
# 
# Method to be implemented
# (i) Logistic regression
# (ii) Discriminant analysis: LDA, QDA
# (iii) K-nearest neighbor
# (iv) Tree-based methods: classification trees, random forests, bagging
# (v) Boosting
# 
# 
# In this project we will present a logistic regressor, LDA,K-nearest neighbor, random forests and a deep neural network
# 
# 
# Authors:
# 
# Anton Blaho Mildton, Nir Teyar, Axel Östfeldt, Jennifer Underdal
# 
# 

# In[211]:


#Install libaries
#Funkar i python 3.9

#!pip install pydot
#!pip install tensorflow
#!pip install prettytable
#!pip install mlxtend
#!pip install seaborn


# In[212]:


import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow.random
from keras.callbacks import TensorBoard
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.callbacks import EarlyStopping

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from IPython.display import Image, display
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pydot

from prettytable import PrettyTable
from mlxtend.plotting import plot_decision_regions
import warnings
import seaborn as sns




# In[213]:


# Ignore all warnings
warnings.filterwarnings("ignore")


# In[214]:


image_filename = 'Datafeatures.png'

# Display the image
display(Image(filename=image_filename))


# # Data preprocessing

# In[215]:


df = pd.read_csv('training_data.csv')
df


# In[216]:


column_names = df.columns
column_names


# In[217]:


# Plot histograms for numerical features
df.hist(figsize=(12, 10))
plt.suptitle("Histograms of Features")
plt.show()


# We can see that snop is constant for all datapoints (0), which will therefore be dropped due to that there is nothing for a ML modell to learn from a constant value

# In[218]:


#drop snowdepth coloumn
dftrain = df.drop('snow', axis=1)
dftrain


# Feature building to categorize bad and good weather condition. This is in general good to do, due to that the ML model get more data

# In[219]:


# Good weather =1 bad weather =0
featurebuild = []

for i in range(1600):
    if dftrain['temp'][i] > 10 and dftrain['windspeed'][i] < 10 and dftrain['cloudcover'][i] < 50:
        featurebuild.append(1)
    else:
        featurebuild.append(0)


count_of_ones = featurebuild.count(1)
print(count_of_ones)


# In[220]:


position = 1
dftrain.insert(position, 'featurebuild', featurebuild)


# In[221]:


# Create the target vector
# 0 - low_bike_demand
# 1 - high_bike_demand

target_vect = dftrain.loc[:,'increase_stock'].to_numpy()


# One hot encoder for the data
enc = OneHotEncoder()
target_names = ['low_bike_demand','high_bike_demand']
target = enc.fit_transform(target_vect[:, np.newaxis]).toarray()


# In[222]:


Target=pd.DataFrame(target)
Target #First coloumn is the correct and used one


# In[300]:


# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(
    dftrain.iloc[:,0:df.shape[1]-1], Target.iloc[:,0], test_size=0.1,random_state=2) #Test


# In[393]:


dftrain.iloc[:,0:df.shape[1]-1]


# In[302]:


#Concating our target 

df_concatenated = pd.concat([df, Target[0]], axis=1)
df_concatenated.rename(columns={Target[0].name: 'Target'}, inplace=True)  # Rename the column to 'Target'

df_concatenated = df_concatenated.drop(columns=['increase_stock'])
#Calculate Correlation
correlation_matrix = df_concatenated.corr()

# Create Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[303]:


df_concatenated


# # Gaussian Naive Bayes

# In[304]:


# Build a Gaussian Classifier
modelNaive = GaussianNB()

# Model training
modelNaive.fit(X_train, Y_train)

# Predict Output
predicted = modelNaive.predict(X_test)


# In[305]:


y_pred = modelNaive.predict(X_test)
accurayNaive = accuracy_score(y_pred, Y_test)
f1_scoreNaive=f1_score(Y_test, y_pred, average="binary")
print("Accuracy:", accurayNaive)
print("F1score:", f1_scoreNaive)


# In[306]:


print("Classification Report:")
print(classification_report(y_pred, Y_test))


# # KNN (K nearest neighbours)
# 
# 
# 
# 

# In[387]:


def average_list(lst):
    sum = sum(lst)
    len = len(lst)
    return sum/len


# In[388]:


#Algorithm parameters
K = 50
test_cases = 10


# In[389]:


X = dftrain.iloc[:,0:df.shape[1]-1]
Y = Target.iloc[:,0] # Extract the target column


# In[390]:


X = np.array(X)
Y = np.array(Y)


# In[391]:


all_k_values = []
plt.figure()
plt.ylabel('K-values for different random states')

for states in range(test_cases):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=states)
    k_list = []

    for k in range(1,K+1):
        #Implement algorithm
        KNN = KNeighborsClassifier(n_neighbors=k) #Initiate the algorithm method
        KNN.fit(Xtrain, Ytrain) #Fit the algorithm to our training set
        Ypred = KNN.predict(Xtest) # Based on our KNN we now make predctions for our test set
        #Evaluate results 
        k_list.append(accuracy_score(Ytest, Ypred))
        #print('K Value: ', k ,'\nAccuracy Score: ', accuracy_score(Ytest, Ypred), '\nClassifications Report:\n', classification_report(Ytest, Ypred), '\nConfusion matrix: \n', confusion_matrix(Ytest,Ypred), '\n\n\n')

    plt.plot(range(1,K+1), k_list)

    #Find maximum k value 
    max_k = max(k_list)
    #Find the index of the maximum value
    max_k_index = k_list.index(max_k)
    print('In random state ', states+ 1, 'the best K value was ', max_k_index, ' at a ', max_k, ' accuracy score.')
    all_k_values.append(k_list)
plt.legend()
mean_k = []

for i in range(K):
    temp = 0
    for j in range(states):
        temp += all_k_values[j][i]
    mean_k.append(temp/states)
    
plt.figure()
plt.plot(range(1,K+1), mean_k, label='Average K over multiple states')
plt.legend()


# In[392]:


#Correct prediction by the modell as a procentage
accuracyKNN = accuracy_score(Ytest, Ypred)
f1_scoreKNN=f1_score(Ytest, Ypred, average="binary")

print(f"Accuracy: {accuracyKNN}")
print("F1score:", f1_scoreKNN)


# In[313]:


print("Classification Report:")
print(classification_report(Ytest, Ypred))


# In[314]:


conf_mat = confusion_matrix(Ytest, Ypred)
displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
displ.plot()


# ## Random forest (Classification tree)
# 
# 
# 

# In[315]:


"""
Input parameters for RandomForestClassifier

n_estimators: The number of trees in the forest.

criterion: The function used to measure the quality of a split. It can be either 'gini' for Gini impurity or 'entropy' for information gain.

max_depth: The maximum depth of the tree. If None, the nodes are expanded until all leaves are pure or until they contain less than the minimum samples split.

min_samples_split: The minimum number of samples required to split an internal node.

min_samples_leaf: The minimum number of samples required to be at a leaf node.

min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.

max_features: The number of features to consider when looking for the best split. It can be an int, float, 'auto', 'sqrt', 'log2', or None.

max_leaf_nodes: Grow trees with max_leaf_nodes in best-first fashion.

min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

min_impurity_split: Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold; otherwise, it is a leaf.

bootstrap: Whether bootstrap samples are used when building trees.

oob_score: Whether to use out-of-bag samples to estimate the generalization accuracy.

n_jobs: The number of jobs to run in parallel for both fit and predict. -1 means using all processors.

random_state: If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.


"""
Froet = RandomForestClassifier()

Parameters = {
    #'n_estimators': [200, 400, 600, 800, 1000],
    #'max_features': ['auto', 'sqrt', 'log2'],
    #'max_depth' : [4,5,6,7,8],
    #'min_samples_split' : [2, 5, 10],
    #'min_samples_leaf': [1, 2, 4],
    #'criterion' :['gini', 'entropy'],
    'n_estimators': [600],
    'max_features': ['auto'],
    'max_depth' : [8],
    'min_samples_split' : [2],
    'min_samples_leaf': [4],
    'criterion' :['entropy'],
    'n_jobs' : [-1]
}


Parameters = {
    'n_estimators': [600],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [8],
    'min_samples_split': [2],
    'min_samples_leaf': [4],
    'criterion': ['entropy'],
    'n_jobs': [-1]
}


Skogen = GridSearchCV(Froet, Parameters, scoring='accuracy', refit=True)
Skogen.fit(X_train, Y_train)


model_prediction = Skogen.predict(X_test)


# In[316]:


best_params = Skogen.best_params_

print("Best Hyperparameters:", best_params)


# In[318]:


#Correct prediction by the modell as a procentage
accuracyRandomF = accuracy_score(Y_test, model_prediction)
f1_scoreRandomF=f1_score(Y_test, model_prediction, average="binary")

print(f"Accuracy: {accuracyRandomF}")
print("F1score:", f1_scoreRandomF)


# In[319]:


print("Classification Report:")
print(classification_report(Y_test, model_prediction))


# In[320]:


conf_mat = confusion_matrix(Y_test, model_prediction)
displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
displ.plot()


# In[243]:


n_estimators_value = best_params['n_estimators']
max_features_value = best_params['max_features']
max_depth_value = best_params['max_depth']
min_samples_split_value = best_params['min_samples_split']
min_samples_leaf_value = best_params['min_samples_leaf']
criterion_value = best_params['criterion']

Ek_froet = RandomForestClassifier(
    n_estimators=n_estimators_value,
    max_features=max_features_value,
    max_depth=max_depth_value,
    min_samples_split=min_samples_split_value,
    min_samples_leaf=min_samples_leaf_value,
    criterion=criterion_value
)



Ek_froet.fit(X_train, Y_train)

Eken = Ek_froet.estimators_[1]

#export_graphviz(Eken, out_file = 'ett_trad_main.dot', feature_names = list(column_names), rounded = True, precision = 3)

#(graph, ) = pydot.graph_from_dot_file('ett_trad_main.dot')

#graph.write_png('ett_trad_main.png')


# ## Logistic regression

# In[244]:


Parameters={
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    'max_iter': [1, 10, 25, 50, 100, 1000],
}

# Specify the logistic regression model
logistic_model = LogisticRegression()

# Initialize the grid search with logistic regression and parameter grid
logicgrid = GridSearchCV(logistic_model, Parameters, scoring='accuracy', refit=True)

# Train the model on the training data
logicgrid.fit(X_train, Y_train)

# Get the best parameters from the grid search
best_params = logicgrid.best_params_
print("Best Parameters:", best_params)

# Use the best model to make predictions
y_pred = logicgrid.predict(X_test)



# In[245]:


#Correct prediction by the modell as a procentage
accuracyLOG = accuracy_score(Y_test, y_pred)
f1_scoreLOG=f1_score(Y_test, y_pred, average="binary")

print(f"Accuracy: {accuracyLOG}")
print("F1score:", f1_scoreLOG)


# In[246]:


conf_mat = confusion_matrix(Y_test, y_pred)
displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
displ.plot()


# In[247]:


print("Classification Report:")
print(classification_report(Y_test, y_pred))


# # Discriminant analysis
# 

# In[248]:


modeldisc = LinearDiscriminantAnalysis()


# In[394]:


Parameters={
    'solver': ['svd', 'lsqr', 'eigen'],
    'priors': [[0.4, 0.6], [0.6, 0.4], [0.5, 0.5]],
    'n_components': [1,2.5,10,15]  # Depending on the number of features in your dataset
}


# In[395]:


clf = GridSearchCV(modeldisc, Parameters, scoring='accuracy', refit=True)


# In[396]:


clf.fit(X_train, Y_train)


# In[397]:


best_params =clf.best_params_

print("Best Hyperparameters:", best_params)


# In[253]:


# fit model
clf.fit(X_train, Y_train)
# make a prediction
yhat = clf.predict(X_test)


# In[254]:


# Evaluate the model
accuracyDisc = accuracy_score(Y_test, yhat)
f1_scoreDisc=f1_score(Y_test, yhat, average='binary')

conf_matrix = confusion_matrix(Y_test, yhat)
classification_rep = classification_report(Y_test, yhat)
print('Accuracy:',accuracyDisc)
print("F1score:", f1_scoreDisc)


# In[255]:


print("Classification Report:")
print(classification_report(Y_test, model_prediction))


# In[256]:


conf_mat = confusion_matrix(Y_test, yhat)
displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
displ.plot()


# # Deep learning NN

# In[257]:


modeldeep = keras.Sequential([
    layers.InputLayer(input_shape=(15,)),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


# In[258]:


X_train


# In[259]:


modeldeep.summary() 


# In[260]:


modeldeep.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[336]:


early_stopping_callback = EarlyStopping(monitor='val_loss', patience=30)


# In[337]:


epoch=150
history_callback = modeldeep.fit(X_train, Y_train,
                                 batch_size=64,
                                 epochs=epoch,
                                 verbose=1,
                                 validation_data=(X_test, Y_test),
                                 callbacks=[early_stopping_callback])


# In[343]:


def plot_accuracy(fit, epochs):
    iterations = np.arange(1, epochs+1)
    plt.plot(iterations, fit.history['accuracy'], label='accuracy')
    plt.plot(iterations, fit.history['val_accuracy'],
             label='validation accuracy')
    plt.legend()
    plt.xticks(iterations[::epochs//10])
    plt.xlabel('Epochs')
    plt.show()

num_epochs = 57
57 #change to number of epochs
plot_accuracy(history_callback, num_epochs) #Might change, Depending on number of epochs it runs, previous cell


# In[344]:


def plot_loss(fit, epochs):
    iterations = np.arange(1, epochs+1)
    plt.plot(iterations, fit.history['loss'], label='loss')
    plt.plot(iterations, fit.history['val_loss'], label='validation loss')
    plt.legend()
    plt.xticks(iterations[::epochs//10])
    plt.xlabel('Epochs')
    plt.show()


plot_loss(history_callback, num_epochs)


# In[345]:


pred = modeldeep.predict(X_test)


# In[346]:


predicted = tensorflow.squeeze(pd.DataFrame(pred))
predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
actual = np.array(Y_test)
conf_mat = confusion_matrix(actual, predicted)
displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
displ.plot()


# In[347]:


# Evaluate the model
accuracydeep = accuracy_score(Y_test, predicted)
f1_scoredeep=f1_score(Y_test, predicted, average='binary')

conf_matrix = confusion_matrix(Y_test, predicted)
classification_rep = classification_report(Y_test, predicted)
print(accuracydeep)
print("F1score:", f1_scoredeep)


# In[348]:


print("Classification Report:")
print(classification_report(Y_test, predicted))


# # Final result

# In[335]:


table = PrettyTable()

table.field_names = ["Model", "Accuracy", "F1 score"]

table.add_row(["Naive Bayes", accurayNaive, f1_scoreNaive])
table.add_row(["KNN",accuracyKNN, f1_scoreKNN])
table.add_row(["Random forest", accuracyRandomF, f1_scoreRandomF])
table.add_row(["Logistic Regression", accuracyLOG, f1_scoreLOG])
table.add_row(["Discriminant analysis", accuracyDisc, f1_scoreDisc])
table.add_row(["Deep learning", accuracydeep, f1_scoredeep])

print(table)


# # Back to real names

# In[271]:


#Result=[] #Back to real names 
#for i in predicted:
#    if i==0:
#        Result.append('low_bike_demand')
#    else:
#        Result.append('high_bike_demand')


# ## Test data test, for second submission
# 
# Main idea here is to use all the training data now (no split of training and testing data so can get as much data
# as possible for a better accuracy on the new test data)
# 
# 
# First part is to read in and manipulate the test data to the right format and features

# In[272]:


test_data=pd.read_csv('test_data.csv')
test_data


# In[273]:


#drop snowdepth coloumn
test_data = test_data.drop('snow', axis=1)
test_data


# In[274]:


# Good weather =1 bad weather =0
featurebuild = []

for i in range(400):
    if test_data['temp'][i] > 10 and test_data['windspeed'][i] < 10 and test_data['cloudcover'][i] < 50:
        featurebuild.append(1)
    else:
        featurebuild.append(0)


count_of_ones = featurebuild.count(1)
print(count_of_ones)


# In[275]:


position = 1
test_data.insert(position, 'featurebuild', featurebuild)


# In[276]:


test_data


# In[352]:


# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(
    dftrain.iloc[:,0:df.shape[1]-1], Target.iloc[:,0], test_size=0.000001,random_state=2) #Test


# In[355]:


"""
Input parameters for RandomForestClassifier

n_estimators: The number of trees in the forest.

criterion: The function used to measure the quality of a split. It can be either 'gini' for Gini impurity or 'entropy' for information gain.

max_depth: The maximum depth of the tree. If None, the nodes are expanded until all leaves are pure or until they contain less than the minimum samples split.

min_samples_split: The minimum number of samples required to split an internal node.

min_samples_leaf: The minimum number of samples required to be at a leaf node.

min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.

max_features: The number of features to consider when looking for the best split. It can be an int, float, 'auto', 'sqrt', 'log2', or None.

max_leaf_nodes: Grow trees with max_leaf_nodes in best-first fashion.

min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

min_impurity_split: Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold; otherwise, it is a leaf.

bootstrap: Whether bootstrap samples are used when building trees.

oob_score: Whether to use out-of-bag samples to estimate the generalization accuracy.

n_jobs: The number of jobs to run in parallel for both fit and predict. -1 means using all processors.

random_state: If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.


"""
Froet = RandomForestClassifier()

Parameters = {
    #'n_estimators': [200, 400, 600, 800, 1000],
    #'max_features': ['auto', 'sqrt', 'log2'],
    #'max_depth' : [4,5,6,7,8],
    #'min_samples_split' : [2, 5, 10],
    #'min_samples_leaf': [1, 2, 4],
    #'criterion' :['gini', 'entropy'],
    'n_estimators': [600],
    'max_features': ['auto'],
    'max_depth' : [8],
    'min_samples_split' : [2],
    'min_samples_leaf': [4],
    'criterion' :['entropy'],
    'n_jobs' : [-1]
}


Parameters = {
    'n_estimators': [600],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [8],
    'min_samples_split': [2],
    'min_samples_leaf': [4],
    'criterion': ['entropy'],
    'n_jobs': [-1]
}


Skogen = GridSearchCV(Froet, Parameters, scoring='accuracy', refit=True)
Skogen.fit(X_train, Y_train)




# In[360]:


testpredrandom = Skogen.predict(test_data)
testpredrandom


# In[366]:


epoch=150
history_callback = modeldeep.fit(X_train, Y_train,
                                 batch_size=64,
                                 epochs=epoch,
                                 verbose=1,
                                 #validation_data=(X_test, Y_test), 
                                 callbacks=[early_stopping_callback])


# In[367]:


predtestdeep = modeldeep.predict(test_data) #95 procent in accuracy here


# In[368]:


predicted = tensorflow.squeeze(pd.DataFrame(predtestdeep))
predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
predicted


# In[380]:


# Element-wise comparison
comparison_result = np.equal(testpredrandom, predicted)

# Check if all elements are equal
arrays_equal = np.all(comparison_result)

# Print the result
print(f"Array 1: {testpredrandom}")
print(f"Array 2: {predicted}")
print(f"Element-wise comparison result: {comparison_result}")
print(f"Arrays are equal: {arrays_equal}")


# In[382]:


# Find indices where the arrays differ
differ_indices = np.where(testpredrandom != predicted)

# Print differences
print("Differences:")
for index in differ_indices[0]:
    print(f"Index {index}: Array 1 value = {testpredrandom[index]}, Array 2 value = {predicted[index]}")


# In[383]:


csv_file_name = "predictionrandom.csv"

# Round the values to integers
rounded_values = np.round(testpredrandom).astype(int)

# Writing a single rounded row to CSV file
np.savetxt(csv_file_name, [rounded_values], delimiter=',', fmt='%d', comments='')

print(f"CSV file '{csv_file_name}' created successfully.")


# 

# In[384]:


csv_file_name = "predictiondeep.csv"

# Round the values to integers
rounded_values = np.round(predicted).astype(int)

# Writing a single rounded row to CSV file
np.savetxt(csv_file_name, [rounded_values], delimiter=',', fmt='%d', comments='')

print(f"CSV file '{csv_file_name}' created successfully.")

