import pandas as pd
import gc
import xgboost as xgb
import numpy as np
import scipy

from subprocess import check_output
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from nltk import word_tokenize

stop_words = stopwords.words('english') + list(punctuation)

# Right format of data
from subprocess import check_output
print(check_output(["ls", "/Users/shreyajain/Downloads/Kaggle/Comment/input"]).decode("utf8"))


# Read data
train = pd.read_csv("/Users/shreyajain/Downloads/Kaggle/Comment/input/train.csv")
test = pd.read_csv("/Users/shreyajain/Downloads/Kaggle/Comment/input/test.csv")


# Check if missing data
train = train.fillna("unknown")
test = test.fillna("unknown")

# Train test split
train_mes, valid_mes, train_l, valid_l = train_test_split(train['comment_text'],train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']], test_size=0.2, random_state=2)


# Tokenise , remove punctuation
def tokenize(text):
    words = word_tokenize(text)
    words = [w.lower() for w in words]
    listwords = [w for w in words if w not in stop_words and not w.isdigit()]
    finaltext = ""
    for each in listwords:
        finaltext += each + " "
    # return finaltext.encode("utf-8")
    return finaltext


transform_com = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                                min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                                smooth_idf=1, sublinear_tf=1).fit(train['comment_text'])
'''comments_train = transform_com.transform(train['comment_text'])'''
comments_train = transform_com.transform(train_mes)
comments_valid = transform_com.transform(valid_mes)
comments_test = transform_com.transform(test['comment_text'])
gc.collect()

train_mes = pd.DataFrame(train_mes)
valid_mes = pd.DataFrame(valid_mes)

print "train_mes ", train_mes[0]
data = [train_mes, valid_mes, test]

for element in data:
    element['total_length'] = element['comment_text'].apply(len)
    element['capitals'] = element['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    element['caps_vs_length'] = element.apply(lambda row: float(row['capitals']) / float(row['total_length']),
                                              axis=1)
    element['num_exclamation_marks'] = element['comment_text'].apply(lambda comment: comment.count('!'))
    element['num_question_marks'] = element['comment_text'].apply(lambda comment: comment.count('?'))
    element['num_punctuation'] = element['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
    element['num_symbols'] = element['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
    element['num_words'] = element['comment_text'].apply(lambda comment: len(comment.split()))
    element['num_unique_words'] = element['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))
    element['words_vs_unique'] = element['num_unique_words'] / element['num_words']
    element['num_smilies'] = element['comment_text'].apply(
        lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))

col = ['total_length', 'capitals', 'caps_vs_length',
       'num_exclamation_marks', 'num_question_marks', 'num_punctuation',
       'num_symbols', 'num_words', 'num_unique_words', 'words_vs_unique',
       'num_smilies']

train_mes = scipy.sparse.csr_matrix(train_mes[col].values)
valid_mes = scipy.sparse.csr_matrix(valid_mes[col].values)
test = scipy.sparse.csr_matrix(test[col].values)

comments_train = scipy.sparse.hstack([train_mes.tocsr(), comments_train.tocsr()])
comments_valid = scipy.sparse.hstack([valid_mes, comments_valid])
comments_test = scipy.sparse.hstack([test, comments_test])

import xgboost as xgb


def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=2017, num_rounds=500):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['eval_metric'] = 'auc'
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain, 'train'), (xgtest, 'test')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    return model


col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
preds = np.zeros((test.shape[0], len(col)))

for i, j in enumerate(col):
    print('fit ' + j)
    model = runXGB(comments_train, train_l[j], comments_valid, valid_l[j])
    preds[:, i] = model.predict(xgb.DMatrix(comments_test), ntree_limit=model.best_ntree_limit)
    gc.collect()

subm = pd.read_csv('../input/sample_submission.csv')
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns=col)], axis=1)
submission.to_csv('xgb.csv', index=False)

# word encoding, skip bigrams, check context improvement, average vector
# tensorflow
# Character encoding, skip bigrams
# CNN - text classification
# GRU , LSTM
# Phrase vectors

# Word + char n gram logistic regression