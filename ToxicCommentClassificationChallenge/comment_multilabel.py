import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re


# Read data
train_df = pd.read_csv("/Users/shreyajain/Downloads/Kaggle/Comment/input/train.csv")
test_df = pd.read_csv("/Users/shreyajain/Downloads/Kaggle/Comment/input/test.csv")

cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']

no_comment = train_df[train_df['comment_text'].isnull()]
len(no_comment)

train_df['char_length'] = train_df['comment_text'].apply(lambda x: len(str(x)))
data = train_df[cols_target]

test_df['char_length'] = test_df['comment_text'].apply(lambda x: len(str(x)))


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


cleaned_train_comment = []
for i in range(0,len(train_df)):
    cleaned_comment = clean_text(train_df['comment_text'][i])
    cleaned_train_comment.append(cleaned_comment)
train_df['comment_text'] = pd.Series(cleaned_train_comment).astype(str)


cleaned_test_comment = []
for i in range(0,len(test_df)):
    cleaned_comment = clean_text(test_df['comment_text'][i])
    cleaned_test_comment.append(cleaned_comment)
test_df['comment_text'] = pd.Series(cleaned_test_comment).astype(str)

train_df = train_df.drop('char_length',axis=1)


X = train_df.comment_text
test_X = test_df.comment_text


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(max_features=5000,stop_words='english')
vect

X_dtm = vect.fit_transform(X)
# examine the document-term matrix created from X_train
X_dtm

test_X_dtm = vect.transform(test_X)
# examine the document-term matrix from X_test
test_X_dtm


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression(C=12.0)

# create submission file
submission_binary = pd.read_csv('/Users/shreyajain/Downloads/Kaggle/Comment/input/sample_submission.csv')

for label in cols_target:
    print('... Processing {}'.format(label))
    y = train_df[label]
    # train the model using X_dtm & y
    logreg.fit(X_dtm, y)
    # compute the training accuracy
    y_pred_X = logreg.predict(X_dtm)
    print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
    # compute the predicted probabilities for X_test_dtm
    test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]
    submission_binary[label] = test_y_prob


submission_binary.to_csv('submission_binary.csv',index=False)


submission_chains = pd.read_csv('/Users/shreyajain/Downloads/Kaggle/Comment/input/sample_submission.csv')

# create a function to add features
def add_feature(X, feature_to_add):
    print "Feature to add"
    print "X ", X.shape
    print "feature_to_add ", len(feature_to_add)
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')



for label in cols_target:
    print('... Processing {}'.format(label))
    y = train_df[label]
    # train the model using X_dtm & y
    logreg.fit(X_dtm,y)
    # compute the training accuracy
    y_pred_X = logreg.predict(X_dtm)
    print('Training Accuracy is {}'.format(accuracy_score(y,y_pred_X)))
    # make predictions from test_X
    test_y = logreg.predict(test_X_dtm)
    test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]
    submission_chains[label] = test_y_prob
    # chain current label to X_dtm
    print "X_dtm ", X_dtm.shape
    print "y ", len(y)
    X_dtm = add_feature(X_dtm, y)
    print('Shape of X_dtm is now {}'.format(X_dtm.shape))
    # chain current label predictions to test_X_dtm
    test_X_dtm = add_feature(test_X_dtm, test_y)
    print('Shape of test_X_dtm is now {}'.format(test_X_dtm.shape))


submission_chains.to_csv('submission_chains.csv', index=False)
submission_combined = pd.read_csv('/Users/shreyajain/Downloads/Kaggle/Comment/input/sample_submission.csv')
for label in cols_target:
    submission_combined[label] = 0.5*(submission_chains[label]+submission_binary[label])


submission_combined.to_csv('submission_combined.csv', index=False)
