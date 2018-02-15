from golem.parsers import csvtojson
import nltk
from nltk.wsd import lesk
import pandas as pd
import numpy as np
import codecs
import nltk.stem


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer


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

    tknzr = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=True)
################################ Stemming ################################
    stemmer = nltk.stem.LancasterStemmer() # may use snowball stemmer, for log regressioon lancaster works better

    class StemmedCountVectorizer(CountVectorizer):
        def build_analyzer(self):
            analyzer = super(StemmedCountVectorizer, self).build_analyzer()
            return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

    ################################ Stemming ################################

    count_vectorizer = StemmedCountVectorizer(
        analyzer="word", tokenizer=tknzr.tokenize,
        preprocessor=None, stop_words='english', max_features=None,ngram_range =(1, 3), min_df=10)

    bag_of_words = count_vectorizer.fit_transform(data_merged['Tweet'])

    emotions = ["anger", "anticipation", "disgust", "fear", "joy",
                "love", "optimism", "pessimism", "sadness", "surprise", "trust"]

    print("Simple logit")
    score_train, score_dev = evaluate_model(df, data_dev, bag_of_words[0:6838], bag_of_words[6838:7724])
    print("Score train: " + str(score_train)) # 0.843431663394
    print("Score dev: " + str(score_dev)) # 0.421218961625

    regularizations = [2]
    max_reg = 0
    max_score_dev = 0
    max_score_train = 0
    for reg in regularizations:
        score_train, score_dev = evaluate_model(df, data_dev, bag_of_words[0:6838], bag_of_words[6838:7724], LogisticRegression(C=reg))
        if score_dev > max_score_dev:
            max_score_dev = score_dev
            max_reg = reg
            max_score_train = score_train

    print("Logit with regularization " + str(max_reg))
    print("Score train: " + str(max_score_train))
    print("Score dev: " + str(max_score_dev))

    emotions_predicted = ["anger-pred", "anticipation-pred", "disgust-pred", "fear-pred", "joy-pred",
                "love-pred", "optimism-pred", "pessimism-pred", "sadness-pred", "surprise-pred", "trust-pred"]

    data_dev.to_csv('data/data_csv/2018-E-c-En-dev-logit-predictions.txt', columns=emotions_predicted, sep = "\t",
                    index = False)
    df.to_csv('data/data_csv/2018-E-c-En-train-logit-predictions.txt', columns=emotions_predicted, sep="\t",
                    index=False)

    main()



