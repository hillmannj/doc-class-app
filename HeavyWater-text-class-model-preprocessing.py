import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

df = pd.read_csv('shuffled-full-set-hashed.csv', header=None,
    names=['label','data'])
df = df[pd.notnull(df['data'])]
df['label_id'] = df['label'].factorize()[0]

label_id_df = df[['label', 'label_id']].drop_duplicates().sort_values('label_id')
label_to_id = dict(label_id_df.values)
id_to_label = dict(label_id_df[['label_id', 'label']].values)
with open('label_to_id.pkl', 'wb') as file:
    joblib.dump(label_to_id, file)
with open('id_to_label.pkl', 'wb') as file:
    joblib.dump(id_to_label, file)
train, test, train_labels, test_labels = train_test_split(df['data'],
    df['label_id'], test_size=0.4, random_state=42, stratify=df['label_id'])
with open('test.pkl', 'wb') as file:
    test.to_pickle(file)
    pd.to_pickle(test_labels, file)
with open('train.pkl', 'wb') as file:
    pd.to_pickle(train, file)
    pd.to_pickle(train_labels, file)

count_vec = CountVectorizer()
sample = df['data'][:10000]
sample_counts = count_vec.fit_transform(sample)
sample_sum = sample_counts.sum(axis=0)
sample_freq = [(word, sample_sum[0, i]) for word, i in count_vec.vocabulary_.items()]
sample_freq = sorted(sample_freq, key=lambda x: x[1], reverse=True)
top5 = [sample_freq[i][0] for i in range(5)]

pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer(analyzer='word', stop_words=top5,
        min_df=5, ngram_range=(1,2))),
    ('tfidf_vectorizer', TfidfTransformer(sublinear_tf=True, norm='l2'))
])
pipeline.fit(train)
with open('pipeline.pkl', 'wb') as file:
    joblib.dump(pipeline, file)

#train_vector_res, train_res_labels = SMOTE(random_state=42).fit_sample(train_vector,
#    train_labels)
