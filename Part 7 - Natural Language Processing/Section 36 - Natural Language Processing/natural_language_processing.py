# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# quoting 3 to ignore " "
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []


# in range (len(dataset))
for i in range(0, 1000):  
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()     
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  # set() is for faster execution
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
# each column to each word and the number of times it appears in the review 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)  # max features to choose the most frequent words, we can see how many words we have and choose a number based on it
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set also trees can be used and works well
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)