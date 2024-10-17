# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student
## DATE:

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and perform any necessary preprocessing, such as handling missing values
   and encoding categorical variables.
2. Initialize the logistic regression model and train it using the training data.
3. Use the trained model to predict the placement status for the test set.
4. Evaluate the model using accuracy and confusion matrix.
   

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:MOULISHWAR 
RegisterNumber:2305001020  
*/
import pandas as pd
data=pd.read_csv("/content/ex45Placement_Data.csv")
data.head
data1=data.copy()
data1.head()data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred,x_test
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()

```

## Output:
![Screenshot 2024-10-17 110154](https://github.com/user-attachments/assets/4267cc76-3040-4d71-a8a3-525a19231989)
![Screenshot 2024-10-17 110229](https://github.com/user-attachments/assets/52446176-64fb-47dc-9d6e-6078870e80e6)
![Screenshot 2024-10-17 110253](https://github.com/user-attachments/assets/75154f7c-3f7a-4664-a771-af2a3c1b5199)
![Screenshot 2024-10-17 110318](https://github.com/user-attachments/assets/0f93b22d-9948-4c09-bc99-6c63ac7f79b6)
![Screenshot 2024-10-17 110334](https://github.com/user-attachments/assets/3003e523-1c42-4ff8-9dc0-5a8c67aba7d1)
![Screenshot 2024-10-17 110357](https://github.com/user-attachments/assets/b3c5a761-17ec-41a0-b16c-c13bf568fa92)
![Screenshot 2024-10-17 110506](https://github.com/user-attachments/assets/fb1490f0-2757-4c34-b2aa-a77e6d65deb5)
![Screenshot 2024-10-17 110530](https://github.com/user-attachments/assets/2263e4b8-8d98-458d-82d7-d78c3174c2ca)
![Screenshot 2024-10-17 110552](https://github.com/user-attachments/assets/b86fccaa-5cfd-4a10-adc1-2eb9f2bea147)
![Screenshot 2024-10-17 110603](https://github.com/user-attachments/assets/1ffa2267-ac89-46c2-9f41-d7c527ae0b62)
![Screenshot 2024-10-17 110617](https://github.com/user-attachments/assets/59103eae-5bfe-4ede-b229-72e99baafca2)














## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
