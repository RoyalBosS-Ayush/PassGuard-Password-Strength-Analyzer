from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("data.csv", on_bad_lines="skip")

# print(data.head())
# print(data["strength"].unique())
# print(data.isna().sum())
# print(data[data["password"].isnull()])
data.dropna(inplace=True)
# print(data.isnull().sum())

x = data["password"]
y = data["strength"]


def word_divide_char(inputs):
    return [i for i in inputs]


# print(word_divide_char("kzde5577"))


vectorizer = TfidfVectorizer(tokenizer=word_divide_char)
X = vectorizer.fit_transform(x)

first_document_vector = X[0]
# print(first_document_vector)
# print(first_document_vector.T.todense())
# print(vectorizer.get_feature_names_out())


df = pd.DataFrame(
    first_document_vector.T.todense(),
    index=vectorizer.get_feature_names_out(),
    columns=["TF-IDF"],
)
# print(df.sort_values(by=["TF-IDF"], ascending=False))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


clf = LogisticRegression(random_state=0, multi_class="multinomial").fit(X_train, y_train)


# y_pred = clf.predict(X_test)
# print(y_pred)
# print(confusion_matrix(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


while True:
    password = input("Enter Password: ")
    dt = np.array([password])
    vec = vectorizer.transform(dt)
    pred = clf.predict(vec)[0]
    print("Strength:", pred, "/ 2")
