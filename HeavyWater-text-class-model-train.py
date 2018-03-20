import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

with open('train.pkl', 'rb') as file:
    train = pd.read_pickle(file)
    train_labels = pd.read_pickle(file)
with open('pipeline.pkl', 'rb') as file:
    pipeline = joblib.load(file)
train_vector = pipeline.transform(train)
clf = OneVsRestClassifier(LinearSVC(random_state=42,
    class_weight='balanced'))
cclf = CalibratedClassifierCV(clf).fit(train_vector, train_labels)
with open('HeavyWater-text-class-model.pkl', 'wb') as file:
    joblib.dump(cclf, file)
cv = cross_val_score(cclf, train_vector, train_labels)
print(cv)
