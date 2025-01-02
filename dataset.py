import matplotlib.pyplot as plt
import tsfel
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
#@title Data Preparation

# Load data
x_train_sig = list(np.loadtxt('E:\\Topici\Dream\\UCI HAR Dataset\\train\\Inertial Signals\\total_acc_x_train.txt', dtype='float32'))
x_test_sig = list(np.loadtxt('E:\\Topici\Dream\\UCI HAR Dataset\\test\\Inertial Signals\\total_acc_x_test.txt', dtype='float32'))
y_test = np.loadtxt('E:\\Topici\\Dream\\UCI HAR Dataset\\test\\y_test.txt')
y_train = np.loadtxt('E:\\Topici\\Dream\\UCI HAR Dataset\\train\\y_train.txt')
activity_labels = np.array(pd.read_csv('E:\\Topici\\Dream\\UCI HAR Dataset\\activity_labels.txt', header=None, delimiter=' '))[:,1]

# dataset sampling frequency
fs = 100


#@title Signal Preview
# %matplotlib inline
plt.figure()
plt_size = 10
plt.plot(np.concatenate(x_train_sig[0:plt_size],axis=0))
plt.xlabel("time (s)")
plt.ylabel("Acceleration (m/sÂ²)")
plt.title("Accelerometer Signal")
plt.legend('x axis')
plt.show()

#@title Feature Extraction
cfg_file = tsfel.get_features_by_domain()                # All features 

# Get features
X_train = tsfel.time_series_features_extractor(cfg_file, x_train_sig, fs=fs)
X_test = tsfel.time_series_features_extractor(cfg_file, x_test_sig, fs=fs)

# Highly correlated features are removed
corr_features = tsfel.correlated_features(X_train)
X_train.drop(corr_features, axis=1, inplace=True)
X_test.drop(corr_features, axis=1, inplace=True)

# Remove low variance features
selector = VarianceThreshold()
X_train = selector.fit_transform(X_train)
X_test = selector.transform(X_test)

# Normalising Features
scaler = preprocessing.StandardScaler()
nX_train = scaler.fit_transform(X_train)
nX_test = scaler.transform(X_test)

classifier = RandomForestClassifier()
# Train the classifier
classifier.fit(nX_train, y_train.ravel())

# Predict test data
y_test_predict = classifier.predict(nX_test)

# Get the classification report
accuracy = accuracy_score(y_test, y_test_predict) * 100
print(classification_report(y_test, y_test_predict, target_names=activity_labels))
print("Accuracy: " + str(accuracy) + '%')