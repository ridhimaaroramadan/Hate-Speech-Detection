import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


# Read in data
data = pd.read_csv(Cleaned file name goes here)
texts = data['text'].astype(str)
y = data['is_offensive']
X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2)

# Vectorize the text
vectorizer = CountVectorizer(stop_words='english', min_df=0.0001)
X = vectorizer.fit_transform(X_train)

# Train the model
model = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=1e5)
cclf = CalibratedClassifierCV(base_estimator=model)
cclf.fit(X, y_train)


predictions = cclf.score(vectorizer.transform(X_test), y_test)
print("The accuracy of the model ",predictions)

Save the model
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(cclf, 'model.joblib')
