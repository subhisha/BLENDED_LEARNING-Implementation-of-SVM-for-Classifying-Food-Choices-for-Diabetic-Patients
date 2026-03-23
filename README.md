# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required Python libraries and load the food dataset for diabetic classification.

2.Load the dataset containing food items and their nutritional information.

3.Train the SVM classifier using the training dataset and tune the hyperparameters to improve performance.

4.Test the model using the test dataset and display the classification results and accuracy.
```
## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
import seaborn as sns

data=pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features=['Calories','Total Fat', 'Saturated Fat', 'Sugars','Dietary Fiber','Protein' ]
target='class'

X=data[features]
y=data[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

svm=SVC()
param_grid={
    'C':[0.1,1,10,100],
    'kernel':['linear','rbf'],
    'gamma':['scale','auto']}
grid_search=GridSearchCV(svm,param_grid,cv=5,scoring='accuracy')
grid_search.fit(X_train,y_train)
best_model=grid_search.best_estimator_

print("Name: SUBHISHA P")
print("Register Number:212225040431")
print("Best Parameters:",grid_search.best_params_)

y_pred=best_model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Name: SUBHISHA P")
print("Register Number:212225040431")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))

conf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="1043" height="707" alt="Screenshot 2026-03-23 090501" src="https://github.com/user-attachments/assets/d5058494-bbdd-49cc-acba-6f2cba9a13f2" />
<img width="984" height="584" alt="Screenshot 2026-03-23 090520" src="https://github.com/user-attachments/assets/07bc1798-4b2d-4414-bbea-594ea285eadb" />



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
