import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


dataset=pd.read_csv("data.csv",delimiter=',')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

data=[]
for i in range(0,1000):
    t1=dataset["text"][i]
    t1=re.sub('[^a-zA-Z]'," ",t1)
    t1=t1.lower()
    t1=t1.split()
    t1=[ps.stem(word) for word in t1 if not word in set(stopwords.words("english"))]
    t1=" ".join(t1)
    data.append(t1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset["author"]=le.fit_transform(dataset["author"]) 
   
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=10000)
x=cv.fit_transform(data).toarray()

with open('CountVectorizer','wb') as file:
    pickle.dump(cv,file)

y=dataset.iloc[:1000,2:3].values    
from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder()
z=one.fit_transform(y[:,0:1]).toarray()
y=np.delete(y,0,axis=1)
y=np.concatenate((z,y),axis=1)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(units= 4281 ,activation="relu" ,init="uniform"))
model.add(Dense(units= 8000 ,activation="relu" ,init="uniform"))
model.add(Dense(units= 2000 ,activation="relu" ,init="uniform"))
model.add(Dense(units= 3 ,activation="softmax" ,init="uniform"))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=5,batch_size=32)

y_pred=model.predict(x_test)
y_pred

index=["EAP","HPL","MWS"]

y_p=model.predict_classes(cv.transform(["But a Glance will show the fallacy of this idea"]))

y_p


print("predicted class",index[y_p[0]])

model.save("author.h5")

