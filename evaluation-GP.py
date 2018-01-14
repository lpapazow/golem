from golem.parsers import csvtojson
import nltk
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


def run_model(X_train, y_train, X_dev, y_dev, clf):
    """General helper function for returning predicted classes on train
    and dev set.

    Args:
        X_train (array-like): Features array. Shape (n_samples, n_features)
        y_train (array-like): Labels array. Shape (n_samples, n_features)
        X_train (array-like): Features array. Shape (n_samples, n_features)
        y_train (array-like): Labels array. Shape (n_samples, n_features)
        clf: Classifier to use. If None, default Log reg is used.

    Returns:
        predicted classes on train, test
    """


    fitted = clf.fit(X_train, y_train)
    probas_train = clf.predict_proba(X_train)
    probas_dev = clf.predict_proba(X_dev)

    # print("Training set score: {:.3f}".format(fitted.score(X_train, y_train)))
    # print("dev set score: {:.3f}".format(fitted.score(X_dev, y_dev)))
    return np.argmax(probas_train, axis=1), np.argmax(probas_dev, axis=1)

def jaccard(df):
    """Calculates the Jaccard similarity
    """
    unions = np.sum(np.logical_or(df.iloc[:, 2:13].values, df.iloc[:, 13:24]).values, axis=1)
    intersections = np.sum(np.logical_and(df.iloc[:, 2:13].values, df.iloc[:, 13:24]).values, axis=1)
    jaccard = intersections / unions
    jaccard[np.isnan(jaccard)] = 1
    return sum(jaccard) / len(jaccard)

def evaluate_model(data_train, data_dev, bag_of_words_train, bag_of_words_dev, clf=None):
    """
    Runs the 11 models and returns the calculated jaccard score on train and test
    """
    if clf is None:
        clf = LogisticRegression()

    for emotion in emotions:
        # print(emotion)
        data_train[emotion + "-pred"], data_dev[emotion + "-pred"] = run_model(bag_of_words_train, data_train[emotion].values.ravel(),
                                                                                bag_of_words_dev, data_dev[emotion].values.ravel(),                                                                              clf)

    return jaccard(data_train), jaccard(data_dev)

def main():
    # vocab = count_vectorizer.get_feature_names()
    a = 1

if __name__ == '__main__':
    parser_train = csvtojson.CsvToJsonParser('data/data_csv/2018-E-c-En-train.txt')
    df = parser_train.get_pandas_data_frame()

    parser_dev = csvtojson.CsvToJsonParser('data/data_csv/2018-E-c-En-dev.txt')
    data_dev = parser_dev.get_pandas_data_frame()

    data_merged=pd.concat([df, data_dev])

    count_vectorizer = CountVectorizer(
        analyzer="word", tokenizer=nltk.word_tokenize,
        preprocessor=None, stop_words='english', max_features=None)

    bag_of_words = count_vectorizer.fit_transform(data_merged['Tweet'])

    emotions = ["anger", "anticipation", "disgust", "fear", "joy",
                "love", "optimism", "pessimism", "sadness", "surprise", "trust"]

    print("Simple logit")
    score_train, score_dev = evaluate_model(df, data_dev, bag_of_words[0:6838], bag_of_words[6838:7724])
    print("Score train: " + str(score_train)) # 0.843431663394
    print("Score dev: " + str(score_dev)) # 0.421218961625

    print("Logit with regularization 5 - this value gives best dev set performance")
    score_train, score_dev = evaluate_model(df, data_dev, bag_of_words[0:6838], bag_of_words[6838:7724], LogisticRegression(C=5))
    print("Score train: " + str(score_train)) # 0.946175782393
    print("Score dev: " + str(score_dev)) # 0.446089970977



# logistic regression without cross val

    main()



