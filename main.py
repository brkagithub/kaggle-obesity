import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

TEST_DATASET = "test.csv"
TRAIN_DATASET = "train.csv"
SENTIMENT_LABELS = ['s1', 's2', 's3', 's4', 's5']
WHEN_LABELS = ['w1', 'w2', 'w3', 'w4']
KIND_LABELS = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15']

# load data
df = pd.read_csv('train.csv')

# preprocessing of data

def preprocess_tweet(tweet):
    # make tweet lowercase
    tweet = tweet.lower()

    # remove mocked links
    tweet = tweet.replace('{link}', '')

    # remove mentions and retweets
    tweet = tweet.replace('rt', '')
    tweet = re.sub(r'@\w+', '', tweet)

    # remove most alphanumerical characters
    tweet = re.sub(r'[^\w\s#°]', '', tweet)

    return tweet

# use one column for sentiment and when variables
df['highest_confidence_sentiment'] = df[SENTIMENT_LABELS].idxmax(axis=1)
df['highest_confidence_when'] = df[WHEN_LABELS].idxmax(axis=1)

# mapping sentiment and when labels to numerical values
sentiment_mapping = {label: i+1 for i, label in enumerate(SENTIMENT_LABELS)}
df['highest_confidence_sentiment'] = df['highest_confidence_sentiment'].map(sentiment_mapping)
when_mapping = {label: i+1 for i, label in enumerate(WHEN_LABELS)}
df['highest_confidence_when'] = df['highest_confidence_when'].map(when_mapping)

# thresholding for kind categories
threshold = 0.3
df[KIND_LABELS] = df[KIND_LABELS].applymap(lambda x: x if x >= threshold else 0)

df.drop(SENTIMENT_LABELS + WHEN_LABELS,  axis=1, inplace=True)

# extract useful information from the tweet
df['tweet'] = df['tweet'].apply(preprocess_tweet)
print(df.head())

# split data to training and test
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['highest_confidence_sentiment'], test_size=0.2, random_state=42)

# create a pipeline with Vectorizer and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('logreg', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
])

# train model
pipeline.fit(X_train, y_train)

# predict test dataset
predictions = pipeline.predict(X_test)

# evaluate model
print(classification_report(y_test, predictions))