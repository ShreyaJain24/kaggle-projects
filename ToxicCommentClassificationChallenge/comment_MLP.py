from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import *

from sklearn.neural_network import MLPClassifier

stop_words = stopwords.words('english') + list(punctuation)
col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']



# glovefile = "/Users/shreyajain/Downloads/datasets/glove.840B.300d.txt"
#
# print "Loading Glove Model"
# f = open(glovefile, 'r')
# model = {}
# count = 0
# for line in f:
#     splitline = line.split()
#     word = splitline[0]
#     embedding = [float(val) for val in splitline[1:]]
#     model[word] = embedding
#     count += 1
# print "Model Loaded. No of Keys: " + str(count)
#
#
# def findaveragevec(xvec):
#     # print "len(xvec) ", len(xvec)
#     finalvec = []
#     try :
#         for i in range(len(xvec[0])):
#             num = 0.0
#             for each in xvec:
#                 # print "each ", each
#                 num += each[i]
#             finalvec.append(num/len(xvec))
#     except:
#         pass
#     return finalvec
#
# def find_vectors(words):
#     xvec = []
#
#     for each in words:
#         num = []
#         try:
#             xvec.append(model[each.lower()])
#         except:
#             pass
#
#     x_av_vec = findaveragevec(xvec)
#     return x_av_vec
#
#
# def createWordVectors():
#     traindata = pd.read_csv("/Users/shreyajain/Downloads/Kaggle/Comment/input/train.csv")
#     testdata = pd.read_csv("/Users/shreyajain/Downloads/Kaggle/Comment/input/test.csv")
#     X_test = np.array(list(testdata["comment_text"]))
#
#     X_train_temp = (list(traindata["comment_text"]))
#
#     X_train = []
#     for each in X_train_temp:
#         X_train.append(tokenize(each))
#
#     X_train = np.array(X_train)
#     X_train_vector = []
#     X_test_vector = []
#
#     for each in X_train:
#         wordlist = each.split(" ")
#         xvec = find_vectors(wordlist)
#         X_train_vector.append(xvec)
#
#     for each in X_test:
#         wordlist = each.split(" ")
#         xvec = find_vectors(wordlist)
#         X_test_vector.append(xvec)
#
#     X_val_vector = X_train_vector[130000:-1]
#
#     return  X_train_vector, X_val_vector, X_test_vector

def savetoFile(predictions):
    subm = pd.read_csv('/Users/shreyajain/Downloads/Kaggle/Comment/input/sample_submission.csv')
    submid = pd.DataFrame({'id': subm["id"]})
    submission = pd.concat([submid, pd.DataFrame(predictions, columns = col)], axis=1)
    submission.to_csv('mlp.csv', index=False)


def tokenize(text):
    text = text.decode("utf-8")
    # print "text ", text
    words = word_tokenize(text)
    words = [w.lower() for w in words]
    listwords = [w for w in words if w not in stop_words and not w.isdigit()]
    finaltext = ""
    for each in listwords:
        finaltext += each + " "
    return finaltext.encode("utf-8")


def createTfIdfInputMatrix():
    # traindata = pd.read_csv("s3://virginia.all.us.datascience.adhoc.internal.zeotap.com/shreya/train.csv")
    traindata = pd.read_csv("/Users/shreyajain/Downloads/Kaggle/Comment/input/train.csv")
    testdata = pd.read_csv("/Users/shreyajain/Downloads/Kaggle/Comment/input/test.csv")
    X_test = np.array(list(testdata["comment_text"]))

    X_train_temp = (list(traindata["comment_text"]))

    X_train = []
    for each in X_train_temp:
        X_train.append(tokenize(each))

    X_train = np.array(X_train)

    tfidf_vectorizer = TfidfVectorizer(min_df=1)
    # ADD X_test
    # X_combined = np.array(X_train) + np.array(X_test)
    tfidf_vectorizer.fit(list(X_train) + list(X_test))
    tfidf_matrix = tfidf_vectorizer.transform(X_train)
    X_train_matrix = tfidf_matrix[0:-1]
    X_val_matrix = tfidf_matrix[130000:-1]
    X_test_matrix = tfidf_vectorizer.transform(X_test)

    return X_train_matrix,X_val_matrix, X_test_matrix


def findYvalues():
    traindata = pd.read_csv("/Users/shreyajain/Downloads/Kaggle/Comment/input/train.csv")
    data = traindata[col]
    return data[0:-1], data[130000:-1]


submission_binary = pd.read_csv('/Users/shreyajain/Downloads/Kaggle/Comment/input/sample_submission.csv')

mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10), verbose=10, learning_rate_init=0.01, max_iter=500)


def add_feature(X, feature_to_add):
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


def MLP(X_train_matrix, Y_train_matrix, X_val_matrix, Y_val_matrix,X_test_matrix, label):
    mlp.fit(X_train_matrix, Y_train_matrix)
    # print "Accuracy for MLP ", mlp.score(X_val_matrix, Y_val_matrix)
    test_y = mlp.predict(X_test_matrix)
    predictions = mlp.predict_proba(X_test_matrix)[:,1]
    # X_train = add_feature(X_train_matrix, Y_train_matrix)
    # X_test = add_feature(X_test_matrix, test_y)

    submission_binary[label] = predictions
    # return X_train, X_test


if __name__ == "__main__":
    X_train, X_val, X_test = createTfIdfInputMatrix()
    # X_train, X_val, X_test = createWordVectors()
    Y_train, Y_val = findYvalues()

    try :
        print "X_train_matrix ", len(X_train)
        print "X_val_matrix ", len(X_val)
        print "Y_train_matrix ", len(Y_train)
        print "Y_val_matrix ", len(Y_val)
        print "X_test_matrix ", len(X_test)
    except:
        pass

    for label in col:
        print "for label ", label
        MLP(X_train, Y_train[label], X_val, Y_val[label], X_test, label)
        # X_train, X_test = MLP(X_train, Y_train[label], X_val, Y_val[label],X_test, label)

    # savetoFile(submission_binary)
    print "Completed"

submission_binary.to_csv('submission_mlp_chain.csv',index=False)


# tensorflow, rnn, cnn