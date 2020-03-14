import nltk
#nltk.download()
import re
nltk.download('stpwords')
import pandas as pd
var=pd.read_csv('C://Users//Gopi//Desktop//Data Science//NLP//sample.csv')

var.head()
print(var.isnull().sum())
var.columns  = ['message','output','sample']
messages = var[['message']]
print(messages.shape)
messages.dropna(inplace=True)
print(messages.shape)
print(messages.isnull().sum())

print(var.shape)
print(messages.shape)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps=PorterStemmer()
corpus = []


for i in range(0,len(messages)):
	review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
	review = review.lower()
	review = review.split()
	
	#review = [ps.stem(word) for word in review if word not in stopwords.set(stopwords.words('english'))]
	review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
	review = ' '.join(review)
	corpus.append(review)

#Creating Bag of Words
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

#Making Label encoding to output variable
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

var["output"]=encoder.fit_transform(var['output'])
var['output'].head()
y=var['output']
print(len(X),type(X))
print(len(y),type(y))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0)

#Train the model using bt Navie Bayes Classifier
from sklearn.naive_bayes import MultinominalNB
model = MultinominalNB.fit(X_train, y_train)

y_pred=model.predict(X_test)

from sklearn.mtrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac)