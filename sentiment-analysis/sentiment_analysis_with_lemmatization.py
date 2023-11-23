import nltk
import random
from nltk.corpus import movie_reviews
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
from string import punctuation

# Download NLTK resources
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Use lemmatization for better word representation
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    words = [word for word in words if word not in stopwords.words('english')]
    return words

# Create a list of tuples where each tuple contains words from a review and its category (pos or neg)
documents = [(preprocess_text(list(movie_reviews.words(fileid))), category)
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

# Update find_features function to use preprocessed text
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
    words = preprocess_text(text)
    feats = find_features(words)
    return classifier.classify(feats)

# Test the classifier with preprocessed text
test_text = "This movie was amazingly good!"
print(sentiment_analysis(test_text))  # Output: pos

test_text = "I didn't like the plot."
print(sentiment_analysis(test_text))  # Output: neg
