# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. x is the feature matirix ,and y is the target variable
2. train test split splits the data
3. logisticregression  builds the model
4. accuracy score evaluates performance

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: moulishwar
RegisterNumber:  2305001020
import pandas as pd
data=pd.read_csv("/content/ex45Placement_Data.csv")
data.head()
data1=data.copy()
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
print("\nClassification report:\n",cr)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()

*/
```

## Output:

![Screenshot 2024-10-03 084138](https://github.com/user-attachments/assets/f9b06ac8-0ec3-47c8-ac25-a56f80b7f9a3)
![Screenshot 2024-10-03 084035](https://github.com/user-attachments/assets/66bb8c64-8a71-4de9-8290-b8fbdfa9666a)
![Screenshot 2024-10-03 084021](https://github.com/user-attachments/assets/090c068c-3157-4625-b7df-dc094bb493ea)
![Screenshot 2024-10-03 084010](https://github.com/user-attachments/assets/b0b0d196-2a31-4828-ab39-8ba673dc399f)
![Screenshot 2024-10-03 083956](https://github.com/user-attachments/assets/551eb49a-901c-48b5-a56d-60f1e23aca59)
![Screenshot 2024-10-03 083947](https://github.com/user-attachments/assets/a0f74320-1640-4061-af0c-5f9c74b03d7a)
![Screenshot 2024-10-03 083938](https://github.com/user-attachments/assets/55097ff8-574d-455d-b650-3aec1d1005d9)
![Screenshot 2024-10-03 083931](https://github.com/user-attachments/assets/2096b886-e176-415d-9987-a39c12d2e649)
![Screenshot 2024-10-03 083919](https://github.com/user-attachments/assets/f1ad1700-1d7e-412b-a20b-578b8b476b24)
![Screenshot 2024-10-03 083904](https://github.com/user-attachments/assets/d1057c3a-493e-41a1-b4e6-992a13b82273)


![Screenshot 2024-10-03 083854](https://github.com/user-attachments/assets/6fd0efa7-81a2-439a-8b5f-c95d98e3384f)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
