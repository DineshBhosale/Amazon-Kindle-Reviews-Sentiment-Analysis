import re
import gensim
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pd.options.mode.chained_assignment = None  # default='warn'

'''
PROJECT FLOW
1. FEATURE ENGINEERING (Text Cleaning; Stop Words Removal; Remove URL's, HTML; LEMMATIZATION)
2. TRAIN TEST SPLIT
3. CONVERT TEXT TO VECTORS USING BAG OF WORDS/TF-IDF/WORD2VEC
4. TRAIN MODELS

Note: If doing BOW/IDF, do train test split before. if doing Word2Vec, do train test split after. 
'''

def lemmatize(review):
    lem = WordNetLemmatizer()
    return ' '.join([lem.lemmatize(word) for word in review.split()])

def get_avgword2vec(review):
   return np.mean([w2v_model.wv[word] for word in review.split() if word in w2v_model.wv.index_to_key], axis = 0)

df = pd.read_csv("./all_kindle_review.csv")
print(df.head())

# extract only reviews and ratings
data = df[["reviewText", "rating"]]

print(data.head())
print("Shape of Data", data.shape)

# check for nans
print(data.isnull().sum())

# check for unique ratings to generate labels
print(data['rating'].unique())

# check for number of data points for each unique label
print(data['rating'].value_counts())

'''
FEATURE ENGINEERING (Text Cleaning; Stop Words Removal; Remove URL's, HTML; Lemmatization)
'''
# ratings below 3 are 0, else 1
data["rating"] = data["rating"].apply(lambda x: "positive" if x > 3 else ("neutral" if x == 3 else "negative"))
print(data.head())
print(data['rating'].value_counts())

# lower all reviews
data['reviewText'] = data['reviewText'].str.lower()
# remove special characters
data['reviewText'] = data['reviewText'].apply(lambda review: re.sub('[^a-z A-Z 0-9-]+', ' ', review))
# remove stop words 
data['reviewText'] = data['reviewText'].apply(lambda review: ' '.join([word for word in review.split() if word not in stopwords.words('english')]))
# remove all url's
data['reviewText'] = data['reviewText'].apply(lambda review: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', ' ', str(review)))
# remove all html tags using Beautiful Soup Library
data['reviewText'] = data['reviewText'].apply(lambda review: BeautifulSoup(review, 'lxml').get_text())
# remove additional spaces
data['reviewText'] = data['reviewText'].apply(lambda review: ' '.join(review.split()))
# perform lemmatization
data['reviewText'] = data['reviewText'].apply(lambda review: lemmatize(review))

'''
CONVERT WORDS TO VECTORS USING WORD2VEC
'''

# word2vec feature engineering
w2v_model = gensim.models.Word2Vec(data['reviewText'], vector_size = 200, epochs = 100)
print(" Vocabulary Length: {}".format(len(w2v_model.wv.index_to_key)))
data['reviewText'] = data['reviewText'].apply(lambda review: get_avgword2vec(review))

print(data.head())
# count for NAN reviews
print(data.isnull().sum())
# drop all NAN reviews
data.dropna(inplace=True, ignore_index = True)
# check again for NAN reviews
print(data.isnull().sum())

'''
TRAIN-TEST-SPLIT
'''

# train test split
X_train, X_test, y_train, y_test = train_test_split(data[['reviewText']], data[['rating']], test_size=0.3, stratify=data['rating'], random_state=42)

# extract all train and test data into a data frame
X_train_df = pd.DataFrame()
for index, row in X_train.iterrows():
    X_train_df = X_train_df._append(pd.DataFrame(row['reviewText'].reshape(1, -1)), ignore_index = True)

X_test_df = pd.DataFrame()
for index, row in X_test.iterrows():
    X_test_df = X_test_df._append(pd.DataFrame(row['reviewText'].reshape(1, -1)), ignore_index = True)
print(X_train_df.shape)

# count for NAN reviews
print(X_train_df.isnull().sum())

'''
MODEL TRAINING AND PREDICTING
'''

classifier = RandomForestClassifier()
classifier.fit(X_train_df, y_train)
y_pred = classifier.predict(X_test_df)

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, normalize='true')

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=classifier.classes_, yticklabels=classifier.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()