# EX 10 Implementation of SVM For Spam Mail Detection
## DATE:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Convert emails into numerical features using tokenization, lowercasing, and TF-IDF or Bag of Words.
2. Transform the processed text into feature vectors for SVM input.
3. Train an SVM classifier (with a linear or other kernel) on labeled data to distinguish between spam and not spam emails.
4. Use the trained SVM model to predict whether new emails are spam and evaluate performance using metrics like accuracy and precision.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection.
Developed by:PREM PRASANTH.J 
RegisterNumber:2305001028  
*/
```
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
df=pd.read_csv('/content/spamEX10.csv',encoding='ISO-8859-1')
df.head()
vectorizer = CountVectorizer()
x=vectorizer.fit_transform(df['v2'])
y=df['v1']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=42)
model=svm.SVC(kernel='linear')
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test,predictions))
def predict_message(message):
  message_vec=vectorizer.transform([message])
  prediction=model.predict(message_vec)
  return prediction[0]
new_message="Congratulations!"
result=predict_message(new_message)
print(f"The message: '(new_message)' is classified as: {result}")
```

## Output:
![image](https://github.com/user-attachments/assets/aaefc283-1640-4e04-8a20-8d1eb0bd2021)
![image](https://github.com/user-attachments/assets/048c9297-f59a-441c-af4f-f55fab0e3261)
![image](https://github.com/user-attachments/assets/23574d88-3cfb-4d82-8284-0125944c282a)
![image](https://github.com/user-attachments/assets/0532c919-c29c-46c2-823e-61852de80529)
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
