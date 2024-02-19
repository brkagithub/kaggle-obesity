import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

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

# drop unneeded labels for sentiment and when
df.drop(SENTIMENT_LABELS + WHEN_LABELS,  axis=1, inplace=True)

# thresholding for kind categories
threshold = 0.3
df[KIND_LABELS] = df[KIND_LABELS].applymap(lambda x: x if x >= threshold else 0)

# extract useful information from the tweet
df['tweet'] = df['tweet'].apply(preprocess_tweet)

# combine tweet, state, location and drop them
df['combined_tweet'] = df['tweet'].astype(str) + ", " + df['state'].astype(str) + ", " + df['location'].astype(str)
df.drop(["tweet", "state", "location"], axis=1, inplace=True)
print(df.head())

# split data to training and test
clf_targets = ['highest_confidence_sentiment', 'highest_confidence_when']
x_train, x_test, y_train_clf, y_test_clf = train_test_split(df['combined_tweet'], df[clf_targets], test_size=0.2, random_state=1)

# create a pipeline with Vectorizer and MultiOutput Regression
pipeline_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultiOutputClassifier(LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000)))
])

# train model
pipeline_clf.fit(x_train, y_train_clf)

# predict test dataset
predictions_clf = pipeline_clf.predict(x_test)

# evaluate model
for i in range(y_test_clf.shape[1]):  # y_test_clf.shape[1] gives the number of target columns
    print(f"Classification Report for {clf_targets[i]}:")
    print(classification_report(y_test_clf.iloc[:, i], predictions_clf[:, i])) # compare true values to predicted values

from sklearn.model_selection import train_test_split

# regression for kinds
X = df['combined_tweet']  # feature
y = df[['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15']]  # targets

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('regressor', MultiOutputRegressor(Ridge()))
])

pipeline.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

predictions = pipeline.predict(X_test)

# evaluate each kind label
for i, kind in enumerate(y.columns):
    mse = mean_squared_error(y_test.iloc[:, i], predictions[:, i])
    r2 = r2_score(y_test.iloc[:, i], predictions[:, i])
    print(f"{kind} - MSE: {mse}, R-squared: {r2}")

# notes: try Bag of Words, feature correlation