import numpy as np
from sklearn.externals import joblib

vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')


def _get_profane_prob(prob):
    return prob[1]


def predict(texts):
    return model.predict(vectorizer.transform(texts)),\
           np.apply_along_axis(_get_profane_prob, 1, model.predict_proba(vectorizer.transform(texts)))


if __name__ == '__main__':
    """."""

    t, x = predict([input('Enter text (Press enter to stop): ')])
    print('Positive' if t[0] == 0 else 'Negative', x[0])
