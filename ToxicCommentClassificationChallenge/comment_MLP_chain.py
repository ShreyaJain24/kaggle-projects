from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import *
import re
from sklearn.neural_network import MLPClassifier

stop_words = stopwords.words('english') + list(punctuation)
col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

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


def add_feature(X, feature_to_add):
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# Read data
train_df = pd.read_csv("/Users/shreyajain/Downloads/Kaggle/Comment/input/train.csv")
test_df = pd.read_csv("/Users/shreyajain/Downloads/Kaggle/Comment/input/test.csv")

cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']

train_df['char_length'] = train_df['comment_text'].apply(lambda x: len(str(x)))
data = train_df[cols_target]

test_df['char_length'] = test_df['comment_text'].apply(lambda x: len(str(x)))


cleaned_train_comment = []
for i in range(0,len(train_df)):
    cleaned_comment = tokenize(train_df['comment_text'][i])
    cleaned_train_comment.append(cleaned_comment)
train_df['comment_text'] = pd.Series(cleaned_train_comment).astype(str)


cleaned_test_comment = []
for i in range(0,len(test_df)):
    cleaned_comment = tokenize(test_df['comment_text'][i])
    cleaned_test_comment.append(cleaned_comment)
test_df['comment_text'] = pd.Series(cleaned_test_comment).astype(str)


train_df = train_df.drop('char_length',axis=1)

X = train_df.comment_text
test_X = test_df.comment_text

vect = TfidfVectorizer(max_features=5000,stop_words='english')
vect

X_dtm = vect.fit_transform(X)
# examine the document-term matrix created from X_train
X_dtm

test_X_dtm = vect.transform(test_X)
# examine the document-term matrix from X_test
test_X_dtm


mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10), verbose=10, learning_rate_init=0.01, max_iter=500)

mlp_train = pd.read_csv("/Users/shreyajain/Downloads/Kaggle/Comment/input/train.csv")
mlp_test = pd.read_csv("/Users/shreyajain/Downloads/Kaggle/Comment/input/test.csv")


for label in cols_target:
    print('... Processing {}'.format(label))
    y = train_df[label]
    # train the model using X_dtm & y
    mlp.fit(X_dtm, y)

    # compute the training accuracy
    y_pred_X = mlp.predict(X_dtm)
    print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
    print "X_dtm ", X_dtm.shape
    # compute the predicted probabilities for X_test_dtm
    train_y_prob = mlp.predict_proba(X_dtm)[:,1]
    test_y_prob = mlp.predict_proba(test_X_dtm)[:,1]
    mlp_train[label] = train_y_prob
    mlp_test[label] = test_y_prob


print "mlp_train ", len(mlp_train)
print "mlp_test ", len(mlp_test)

mlp_train.to_csv("mlp_chain_train.csv",index=False)
mlp_test.to_csv("mlp_chain_test.csv",index=False)

submission_chains = pd.read_csv('/Users/shreyajain/Downloads/Kaggle/Comment/input/sample_submission.csv')


for label in cols_target:
    print('... Processing {}'.format(label))
    y = train_df[label]
    # train the model using X_dtm & y
    # X_new = dataLabelTrain(label)

    for each in col:
        if each != label:
            X_dtm = add_feature(X_dtm, mlp_train[each])

    print "X_dtm training ", X_dtm.shape
    print "y ", len(y)
    mlp.fit(X_dtm, y)

    # make predictions from test_X
    for each in col:
        if each != label:
            test_X_dtm = add_feature(test_X_dtm, mlp_test[each])

    print "test_X_dtm training ", test_X_dtm.shape
    test_y = mlp.predict(test_X_dtm)
    test_y_prob = mlp.predict_proba(test_X_dtm)[:, 1]
    submission_chains[label] = test_y_prob
    # chain current label to X_dtm
    print "X_dtm ", X_dtm.shape
    print "y ", len(y)

submission_chains.to_csv("submission_mlp_chain.csv",index=False)

"""
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


X_train, X_val, X_test = createTfIdfInputMatrix()
Y_train, Y_val = findYvalues()


def add_feature(X, feature_to_add):
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


mlp_chain_test_labels_prob = pd.read_csv('/Users/shreyajain/Downloads/Kaggle/Comment/input/sample_submission.csv')
mlp_chain_train_labels_prob = pd.read_csv('/Users/shreyajain/Downloads/Kaggle/Comment/input/train.csv')
submission_chains = pd.read_csv('/Users/shreyajain/Downloads/Kaggle/Comment/input/sample_submission.csv')
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10), verbose=10, learning_rate_init=0.01, max_iter=10)

try:
    print "X_train_matrix ", len(X_train)
    print "X_val_matrix ", len(X_val)
    print "Y_train_matrix ", len(Y_train)
    print "Y_val_matrix ", len(Y_val)
    print "X_test_matrix ", len(X_test)
except:
    pass

mlp_chain_train_labels_prob = mlp_chain_train_labels_prob["comment_text"]
print "mlp_chain_train_labels_prob ", len(mlp_chain_train_labels_prob)

for label in col:
    print "for label ", label
    print('... Processing {}'.format(label))
    y = Y_train[label]
    print "Y_train ", Y_train.shape
    print "y ", len(y)
    print "X_train ", X_train.shape
    # train the model using X_dtm & y
    mlp.fit(X_train, y)
    # compute the training accuracy
    y_pred_X = mlp.predict(X_train)
    train_y_prob = mlp.predict_proba(X_train)[:,1]
    print "train_y_prob ", len(train_y_prob)
    # mlp_chain_train_labels_prob[label] = train_y_prob
    mlp_chain_train_labels_prob = add_feature(mlp_chain_train_labels_prob, train_y_prob)
    print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
    # compute the predicted probabilities for X_test_dtm
    test_y_prob = mlp.predict_proba(X_test)[:, 1]
    mlp_chain_test_labels_prob[label] = test_y_prob

# mlp_chain_test_labels_prob.to_csv('mlp_chain_test_labels_prob.csv',index=False)
# mlp_chain_train_labels_prob.to_csv('mlp_chain_train_labels_prob',index=False)


def dataLabelTrain(label):
    labels = []
    for each in col:
        if each != label:
            labels.append(each)
    X_label = add_feature(X_train, mlp_chain_train_labels_prob[labels])
    return X_label

def dataLabelTest(label):
    labels = []
    for each in col:
        if each != label:
            labels.append(each)
    X_label = add_feature(X_test, mlp_chain_test_labels_prob[labels])
    return X_label


for label in col:
    print('... Processing {}'.format(label))
    y = Y_train[label]
    # train the model using X_dtm & y
    x_train_label = dataLabelTrain(label)
    mlp.fit(x_train_label)

    # compute the training accuracy
    y_pred_X = mlp.predict(x_train_label)
    print('Training Accuracy is {}'.format(accuracy_score(y, y_pred_X)))
    # make predictions from test_X
    x_test_label = dataLabelTest(label)
    # test_y = mlp.predict(list(X_test) + list(x_test_label))
    test_y_prob = mlp.predict_proba(x_test_label)[:, 1]
    submission_chains[label] = test_y_prob
    # # chain current label to X_dtm
    # X_train = add_feature(X_train, y)
    # print('Shape of X_dtm is now {}'.format(X_train.shape))
    # # chain current label predictions to test_X_dtm
    # test_X_dtm = add_feature(X_test, test_y)
    # print('Shape of test_X_dtm is now {}'.format(X_test.shape))

print "Completed"

submission_chains.to_csv('submission_mlp_chain.csv',index=False)

"""
# tensorflow, rnn, cnn