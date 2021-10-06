import pandas as pd
import pickle
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    df = pd.read_csv('spam.csv', encoding='latin-1')
    
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    df['label'] = df['v1'].map({'ham': 0, 'spam': 1})

    X = df['v2']
    y = df['label']

    cv = CountVectorizer()

    X = cv.fit_transform(X) # Fit the Data

    # train, test data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Naive Bayes Classifier
    clf = MultinomialNB()

    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

    return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)