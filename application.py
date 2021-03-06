import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, redirect, request
from werkzeug.utils import secure_filename

APP_ROOT = os.path.dirname(os.path.realpath(__file__))

application = Flask(__name__)

#Index
@application.route('/', methods=['GET'])
def index(result=None):
    if request.args.get('words', None):
        result = classify_words(request.args['words'])
    return render_template('index.html', result=result)

def classify_words(words):
    with open(os.path.join(APP_ROOT, 'id_to_label.pkl'),
        'rb') as file:
        id_to_label = joblib.load(file)
    with open(os.path.join(APP_ROOT, 'pipeline.pkl'), 'rb') as file:
        pipeline = joblib.load(file)
    with open(os.path.join(APP_ROOT,
        'HeavyWater-text-class-model.pkl'), 'rb') as file:
        clf = joblib.load(file)
    words = [words]
    words_vector = pipeline.transform(words)
    predicted = clf.predict(words_vector)
    confidence = clf.predict_proba(words_vector)
    confidence = confidence[0][predicted[0]] * 100
    predicted = id_to_label[predicted[0]]
    return("Predicted label: {0} with {1:.2f}% confidence".format(predicted,
        confidence))

if __name__ == "__main__":
    application.run(debug=True)
