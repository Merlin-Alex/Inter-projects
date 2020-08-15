#PREDICTION OF HEART DISEASES
#DATA COLLECTION
import pandas as pd
heart = pd.read_csv('heart.csv')

#DATA INTERPRETATION
#Information about dataset
heart.info()
#Statistical information about dataset
heart.describe()

#info=["age","1: male, 0: female","chest pain type, 1: typical angina,
#      2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"
#      ," serum cholestoral in mg/dl","fasting blood  sugar > 120 mg/dl"
#      ,"resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved"
#      ,"exercise induced angina","oldpeak = ST depression induced by exercise 
#      relative to rest","the slope of the peak exercise ST segment","number of major vessels
#     (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



#DATA CLEANING
#Delete the useless data from dataset, here 'Id'
#not required here

#DATA ANALYSIS

#COUNTPLOT
#Library: seaborn
#Analys.: countplot
import seaborn as sb
sb.countplot(x='TARGET',data=heart)

#SCATTER MATRIX
#Helps to see feature-feature relation
#Library: pandas
#Class  : plotting
#Funct. : scatter_matrix
pd.plotting.scatter_matrix(heart,c='g',alpha=0.6,figsize=(10,10),marker='+')


#REPLACE SPECIES OBJECT DATA TO NUMERICAL
classes={'Not sick':0,'sick':1}
heart.replace({'TARGET':classes},inplace=True)

#ARRAY CREATION
#X=Features, Y=Target
x=heart.iloc[:,:-1].values  #data[:,0:4]
y=heart.iloc[:,-1].values   #data[:,4]

#SPLIT UNIVERSAL DATA TO TRAIN vs TEST
#Library: sklearn
#Module : model_selection
#Class  : train_test_split
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2,random_state=45) 
#LOGISTIC REGRESSION
#Library: sklearn
#Module : linear_model
#Class  : LogisticRegression
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
#TRAINING step (Give data to model to learn)
#Function: fit(arrays*), *training set of x and y
logreg.fit(x_train,y_train)
#TESTING step (Give new data and test algo efficiency)
#1. Accuracy function: score(arrays*), *test arrays
#2. Predict function : predict(array*),*new features
logregacc=logreg.score(x_test,y_test)
logregpred=logreg.predict(x_test)

#Compare right vs wrong predictions
#CONFUSION MATRIX
#Library: sklearn
#Module : metrics
#Class  : confusion_matrix
from sklearn.metrics import confusion_matrix
conmat=confusion_matrix(y_test,logregpred)
