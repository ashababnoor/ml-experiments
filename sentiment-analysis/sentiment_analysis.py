import nltk
import random
from nltk.corpus import movie_reviews
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

# Download NLTK resources
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')

# Create a list of tuples where each tuple contains words from a review and its category (pos or neg)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
random.shuffle(documents)

# Get all words from the movie reviews
all_words = [word.lower() for word in movie_reviews.words()]

# Create a frequency distribution of words
all_words = FreqDist(all_words)

# Use the 2000 most common words as features
word_features = list(all_words.keys())[:2000]

# Define a function to create a dictionary of features for each review
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# Create the feature sets using the feature extraction function
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# Split the dataset into training and testing sets
train_set = featuresets[:1500]
test_set = featuresets[1500:]

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Define a function to perform sentiment analysis on a given text
def sentiment_analysis(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word.lower() not in punctuation]
    feats = find_features(words)
    return classifier.classify(feats)

# Test the classifier
test_text = "This movie was amazing!"
print(sentiment_analysis(test_text))  # Output: pos

test_text = "I didn't like the plot."
print(sentiment_analysis(test_text))  # Output: neg
