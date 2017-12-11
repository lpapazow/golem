import nltk.classify.util
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


def get_useful_word(words, stemmer):
    words = nltk.word_tokenize(words)
    # TODO: only one for-loop
    useful_words = [word for word in words if word not in stopwords.words("english")]
    useful_words = [word for word in words if len(word) != 1]
    useful_words = [stemmer.stem(word) for word in useful_words]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict


def read_tweets():
    positive_tweets = []
    negative_tweets = []
    stemmer = PorterStemmer()
    with open('data/data.txt', encoding='latin-1') as tweets, open('data/data_labels.txt', encoding='latin-1') as labels:
        for tweet, label in zip(tweets, labels):
            useful_words = get_useful_word(tweet, stemmer)
            if(int(label[0]) == 1):
                positive_tweets.append((useful_words, 'positive'))
            else:
                negative_tweets.append((useful_words, 'negative'))

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
