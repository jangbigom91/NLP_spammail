import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# NLP모델을 활용해서 스팸메일 적발시스템 구현
# 나이브 베이즈 분류기는 이메일 필터링과 관련하여 널리 쓰이는 통계기법
df = pd.read_csv('spam.csv', encoding='latin-1')

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df['label'] = df['v1'].map({'ham': 0, 'spam': 1})

X = df['v2']
y = df['label']
cv = CountVectorizer()

X = cv.fit_transform(X) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Naive Bayes Classifier
clf = MultinomialNB()

clf.fit(X_train, y_train)
clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))

# pickle 파일로 모델 저장
import joblib

joblib.dump(clf, 'NB_spam_model.pkl')

# 저장된 모델을 로드할 때는
# NB_spam_model = open('NB_spam_model.pkl', 'rb')
# clf = joblib.load(NB_spam_model)
