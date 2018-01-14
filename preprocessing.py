from golem.parsers import csvtojson
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def evaluate_features(X, y, clf=None):
    """General helper function for evaluating effectiveness of passed features in ML model

    Prints out Log loss, accuracy, and confusion matrix with 3-fold stratified cross-validation

    Args:
        X (array-like): Features array. Shape (n_samples, n_features)

        y (array-like): Labels array. Shape (n_samples,)

        clf: Classifier to use. If None, default Log reg is use.
    """
    if clf is None:
        clf = LogisticRegression()

    probas = cross_val_predict(clf, X, y, cv=StratifiedKFold(random_state=8),
                               n_jobs=-1, method='predict_proba', verbose=2)
    pred_indices = np.argmax(probas, axis=1)
    classes = np.unique(y)
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(y, probas)))
    print('Accuracy: {}'.format(accuracy_score(y, preds)))


def main():
    parser = csvtojson.CsvToJsonParser('data/data_csv/2018-E-c-En-train.txt')

    data = parser.get_pandas_data_frame()

    count_vectorizer = CountVectorizer(
        analyzer="word", tokenizer=nltk.word_tokenize,
        preprocessor=None, stop_words='english', max_features=None)

    bag_of_words = count_vectorizer.fit_transform(data['Tweet'])

    vocab = count_vectorizer.get_feature_names()
    # print("Anger")
    # evaluate_features(bag_of_words, data['anger'].values.ravel())
    # print("Joy")
    # evaluate_features(bag_of_words, data['joy'].values.ravel())
    print("trust")
    evaluate_features(bag_of_words, data['trust'].values.ravel())

if __name__ == '__main__':
    main()



