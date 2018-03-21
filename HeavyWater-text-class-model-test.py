import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

with open('test.pkl', 'rb') as file:
    test = pd.read_pickle(file)
    test_labels = pd.read_pickle(file)
with open('pipeline.pkl', 'rb') as file:
    pipeline = joblib.load(file)
test_vector = pipeline.transform(test)
with open('HeavyWater-text-class-model.pkl', 'rb') as file:
    clf = joblib.load(file)
predicted = clf.predict(test_vector)
cm = confusion_matrix(test_labels, predicted)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cr = classification_report(test_labels, predicted)
print(cr)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion-matrix.png')
