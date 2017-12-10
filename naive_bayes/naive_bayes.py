import nltk.classify.util
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas


def get_useful_word(words):
    words = nltk.word_tokenize(words)
    words = ({word: True for word in words if word not in stopwords.words(
        'english') or len(word) == 1})
    return words


def read_tweets():
    positive_tweets = []
    negative_tweets = []
    with open('data/data.txt', encoding='latin-1') as tweets, open('data/data_labels.txt', encoding='latin-1') as labels:
        for tweet, label in (tweets, labels):
            useful_words = get_useful_word(tweet)
            if(int(label[0]) == 1):
                positive_tweets.append((useful_words, 'negative'))
            else:
                negative_tweets.append((useful_words, 'positive'))

            return positive_tweets, negative_tweets


if __name__ == "__main__":
    positive_tweets, negative_tweets = read_tweets()

    train_set = negative_tweets[:int(
        (.8) * len(negative_tweets))] + positive_tweets[:int((.8) * len(positive_tweets))]
    test_set = negative_tweets[int(
        (.8) * len(negative_tweets)):] + positive_tweets[int((.8) * len(positive_tweets)):]

    classifier = NaiveBayesClassifier.train(train_set)
    accuracy = nltk.classify.util.accuracy(classifier, test_set)
    print(accuracy * 100)
    classifier.show_most_informative_features()
    print('END')
