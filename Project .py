#!/usr/bin/env python
# coding: utf-8

# ##  FIRST OBJECTIVE: WHAT TYPE OF EMPLOYEES ARE LEAVING?
# 
# EXPLAIN WHAT TYPE OF EMPLOYEE ARE PRONE TO LEAVE

# In[1]:


## Load libaries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load the datasets for Employee who has left.
left = pd.read_excel("C:/Users/hp/Desktop/Hash Analytics  Internship Study Kit/Assignments/Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx",sheet_name=2)
left.head(4)


# ## Exploratory Data Analysis
# 
# #### Descriptive Analysis of the Quantitative features

# In[3]:


### features that are Quantitative
quantitative =  ['satisfaction_level', 'last_evaluation', 'number_project',       'average_montly_hours', 'time_spend_company']

## Descriptive Analysis- Average Value
left[quantitative].describe()


# The average satisfaction level for employee who have left is 0.440098,last evaluation on an average was 0.718113, the average number of project is 3.855503, the  monthly hour spent on an average  is 207.419210, and  the average time spent in the company is 3.876505(approximately 4)
# 
# 
# ### Qualitative Features 
# #### a)Number of Project Completed 
# 

# In[4]:


print(left['number_project'].value_counts())

## Chart
plt.figure(figsize=(10,5))
plt.bar(left['number_project'].value_counts().index,left['number_project'].value_counts().values)
plt.ylabel("Number of Employee who have left")
plt.xlabel("number_project")
plt.title("Number of projects completed")
plt.grid()
plt.savefig("proj_completed.png")
plt.show()


# #### b) Time spent at the company

# In[5]:



print(left['time_spend_company'].value_counts())

## Chart
plt.figure(figsize=(10,5))
plt.bar(left['time_spend_company'].value_counts().index,left['time_spend_company'].value_counts().values)
plt.ylabel("Number of Employee who have left")
plt.xlabel("time_spend_company")
plt.title("Time spent at the company")
plt.grid()
plt.show()


# #### c) Whether they have had a work accident

# In[6]:


print(left['Work_accident'].value_counts())

## Chart
plt.figure(figsize=(5,5))
plt.bar(["0","1"],left['Work_accident'].value_counts().values,label=["No/0","Yes/1"])
plt.ylabel("Number of Employee who have left")
plt.xlabel("Work Accident")
plt.title("Whether they have had a work accident")
plt.legend()
plt.grid()
plt.savefig("work_accident.png",dpi = 100)
plt.show()


# The majority of employee leaving have no work accidents
# 
# #### d) Whether they have had a promotion in the last 5 years

# In[7]:


print(left['promotion_last_5years'].value_counts())

## Chart
plt.figure(figsize=(5,5))
plt.bar(["0","1"],left['promotion_last_5years'].value_counts().values,label=["No/0","Yes/1"])
plt.ylabel("Number of Employee who have left")
plt.xlabel("promotion_last_5years")
plt.title("Whether they have had a promotion in the last 5 years")
plt.legend()
plt.grid()
plt.savefig("Promotion",dp1=100)
plt.show()


# The majority of employee leaving have not been promoted in the last five years
# 
# #### e) Departments

# In[14]:


### Number of Employee who have left in Each department
print(left['dept'].value_counts())

## Chart
plt.figure(figsize=(12,5))
plt.bar(left['dept'].value_counts().index,left['dept'].value_counts().values)
plt.ylabel("Number of Employee who have left")
plt.xlabel("Departments")
plt.title("What departments are they from?")
plt.grid()
plt.savefig("dept",dpi=100)
plt.show()


# The sales department has the highest number of employee who have left, followed by the technical department, Support Department and the IT department.
# 
# The Mangement department has the lowest number of employee who had left the organization
# 
# #### f)Salary Class

# In[15]:


### Number of Employee who have left for each salary class
print(left['salary'].value_counts())

### Bar Chart
plt.figure(figsize=(8,5))
plt.bar(left['salary'].value_counts().index,left['salary'].value_counts().values)
plt.title("What categories is thier salary?")
plt.xlabel('Salary')
plt.ylabel("Number of Employee who have left")
plt.grid()
plt.savefig("salary",dpi=100)
plt.show()


# In[10]:


print(left.groupby('salary')["Work_accident"].value_counts())
print("\n\n")
print(left.groupby('salary')["promotion_last_5years"].value_counts())


# ### <b> conclusion</b>
# 
# The Employee with Low to Medium salarie tends to leave, This is is because marjority aof them have no work accidnet and have not been promoted in last five years.

# ## SECOND OBJECTIVE: DETERMINE WHICH EMPLOYEES ARE PRONE TO LEAVE NEXT
# 
# PREDICT THE FUTURE EMPLOYEE WHO WOULD TEND TO LEAVE THE COMPANY.
# 
# 

# In[8]:


#load the datasets for Employee who is still existing.
existing = pd.read_excel("C:/Users/hp/Desktop/Hash Analytics  Internship Study Kit/Assignments/Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx",sheet_name=1)


# In[9]:


## Add the atrribute Churn to Existing Employeee dataset
existing['Churn']= 'No'
existing.head(2)


# In[10]:


## Add the attribute churn to Employee who has left dataset
left['Churn']='Yes'
left.head(2)


# In[11]:


## Combining left and existing Dataframes together to create a single dataframes.
employee_attrition =  pd.concat([left, existing], ignore_index=True)
employee_attrition.head(10)


# In[12]:


##  Aggregate Average using the Churn Group
employee_attrition.groupby('Churn').mean()


# In[44]:


## Percentage Of Employee who have left and are still existing in the Employee Attrition Dataset\
print(employee_attrition['Churn'].value_counts(),'\n')

print("The percentage of existing employee is {}, while that of the employee who had left is {}".format(   employee_attrition[employee_attrition['Churn'] == "No"].shape[0]/ employee_attrition.shape[0],   employee_attrition[employee_attrition['Churn'] == "Yes"].shape[0]/ employee_attrition.shape[0]))
print("An imbalanced data")


# In[13]:


plt.bar(["No","Yes"],employee_attrition['Churn'].value_counts().values)


# ## Data Preprocessing
# 
# ### a) Checking For Missing Values

# In[14]:


# Removing Redindant Variables
employee_attrition.drop('Emp ID', axis=1, inplace=True)
# Checking For Missing Values
employee_attrition.isnull().sum()


# ### b) Enconding All Categorical Variables

# In[15]:


## Encoding The  Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le= LabelEncoder()


# In[16]:



# Label Encoding will be used for columns with 2 or less unique values
le_count = 0
for col in employee_attrition.columns:
    if employee_attrition[col].dtype == 'object':
        le.fit(employee_attrition[col])
        employee_attrition[col] = le.transform(employee_attrition[col])
        le_count += 1


print('{} columns were label encoded.'.format(le_count))


# In[17]:


employee_attrition.head(3)


# ### c) Checking For the Distribution of the Variables 

# In[18]:


employee_attrition.hist(figsize=(15,10))


# 
# ### d) Checking for Correlation with the target variable

# In[19]:


correlation= employee_attrition.corrwith(employee_attrition['Churn'])
correlation


# - Looking at the distribution of the features, all of the  features are not normally distirbuted, no clear assumption         regarding the distribution of the features can be made.
# 
# - All of the independent features has a low correlation with  the target varaibles.
# 
# - Therefore the Non parametric Machine learning will be used for training and testing the data on the model.
# 
# We will consider
#  - K nearest Neighbours
#  - Decision Tree Algorithm 
#  - Support Vector Machine and
#  - Random Forest Classifiers
# 

# In[20]:


## Selceting the Independent and Dependent Variables
X = employee_attrition.iloc[:,[0,1,2,3,4,5,6,7,8]].values ### Matrix of Independent faetures
y = employee_attrition.iloc[:,9].values    ### Vector of target varible 

np.random.seed(100)


# In[21]:


### Splitting the data into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30, random_state= 1000)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# ##   Training and Testing the Model

# In[22]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,cohen_kappa_score


# #### a)  Decsion Tree algorithm

# In[23]:


from sklearn.tree import DecisionTreeClassifier

## Training the Model
dtree=  DecisionTreeClassifier()
dtree.fit(X_train, y_train)

## Testing the Model
y_pred_dtree =  dtree.predict(X_test)


# In[32]:


## Evaluation
print(accuracy_score(y_test,y_pred_dtree))
cf_dtree = confusion_matrix(y_test,y_pred_dtree)
sns.heatmap(cf_dtree, annot=True, cmap='Blues', fmt='g')
plt.title("CONFUSION MATRIX",pad=20)
plt.xlabel("TRUE VALUE")
plt.ylabel("PREDICTED VALUE")
plt.savefig("decision_tree",dpi=100)
plt.show()
print(classification_report(y_test,y_pred_dtree))
cohen_kappa_score(y_test,y_pred_dtree)


# #### b) Random forest Classifier

# In[33]:


from sklearn.ensemble import RandomForestClassifier

## Trainign the Model
rforest = RandomForestClassifier()
rforest.fit(X_train,y_train)

## Testing the Model
y_pred_rforest = rforest.predict(X_test)


# In[34]:


## Evaluation
print(accuracy_score(y_test,y_pred_rforest))
cf_rforest = confusion_matrix(y_test,y_pred_rforest)
sns.heatmap(cf_rforest, annot=True, cmap='Blues', fmt='g')
plt.title("CONFUSION MATRIX",pad=20)
plt.xlabel("TRUE VALUE")
plt.ylabel("PREDICTED VALUE")
plt.savefig("random_forest",dpi=100)
plt.show()
print(classification_report(y_test,y_pred_rforest))
cohen_kappa_score(y_test,y_pred_rforest)


# #### c) Support Vector Machine
# SVM is sensitive to feature scaling since optimization is done be calcuating the distance between features
# 
# To perform the SVM  classifier we need to perform features scaling on the indepent varaibles.
# since the features are not normally Ditributed we will be Normalizing the data using the minmarx scale

# In[35]:


## Standard Scaler
from sklearn.preprocessing import MinMaxScaler
minmaxscaler = MinMaxScaler()
X_train = minmaxscaler.fit_transform(X_train)
X_test = minmaxscaler.transform(X_test)


# In[36]:


from sklearn.svm import LinearSVC
## Trainign the Model
supportvm = LinearSVC()
supportvm.fit(X_train,y_train)

## Testing the Model
y_pred_supportvm = supportvm.predict(X_test)


# In[37]:


## Evaluation
print(accuracy_score(y_test,y_pred_supportvm))
cf_supportvm = confusion_matrix(y_test,y_pred_supportvm)
sns.heatmap(cf_supportvm, annot=True, cmap='Blues', fmt='g')
plt.title("CONFUSION MATRIX",pad=20)
plt.xlabel("TRUE VALUE")
plt.ylabel("PREDICTED VALUE")
plt.savefig("svm",dpi=100)
plt.show()
print(classification_report(y_test,y_pred_supportvm))
cohen_kappa_score(y_test,y_pred_supportvm)


# ### d) K Nearest Neighbour

# In[38]:


from sklearn.neighbors import KNeighborsClassifier

accuracy={}

for i in range(3,20):
    
        knn= KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        accuracy[i]= accuracy_score(y_test,pred)
        #print("Accuracy,k(",i,") is ", accuracy_score(y_test,pred))


# In[39]:


from sklearn.neighbors import KNeighborsClassifier
## Training The Model

knn= KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)

## testing the Model
y_pred_knn = knn.predict(X_test)


# In[40]:


## Evaluation
print(accuracy_score(y_test,y_pred_knn))
cf_knn = confusion_matrix(y_test,y_pred_knn)
sns.heatmap(cf_knn, annot=True, cmap='Blues', fmt='g')
plt.title("CONFUSION MATRIX",pad=20)
plt.xlabel("TRUE VALUE")
plt.ylabel("PREDICTED VALUE")
plt.savefig("knn",dpi=100)
plt.show()
print(classification_report(y_test,y_pred_knn))
cohen_kappa_score(y_test,y_pred_knn)


# The random forest is the best model trained on the data, it has the highest accuracy and kohen cappa value
# It will be used in Prediction the Existing Employee with the probability of leaving
# 
# ### Predicting the Existing Employee with the Probability of leaving

# In[41]:


####   Data Prepocessing
exist = existing.drop(['Emp ID','Churn'], axis=1 )
exist.head(1)


# In[42]:


## Encoding The Categorical variable

from sklearn.preprocessing import LabelEncoder

# Label Encoding will be used for columns with 2 or less unique values
le_count = 0
for col in exist.columns:
    if exist[col].dtype == 'object':
        le.fit(exist[col])
        exist[col] = le.transform(exist[col])
        le_count += 1
print('{} columns were label encoded.'.format(le_count))


# In[43]:


### Predicting Using the Random Forest Model
churn_prediction = rforest.predict(exist)
churn_prediction= pd.Series(churn_prediction)
churn_prediction.value_counts()


# In[44]:


### Probability of falling into the class 
churn_probability = rforest.predict_proba(exist)
churn_probability = pd.DataFrame(churn_probability)
churn_probability


# 7 Existing Employees have the Probailty Of Leaving the Organization
# 
# ##### Extracing  the Employee with the Probability of Churn

# In[45]:


existing["Prediction"]= churn_prediction
existing["Probability"]= churn_probability[1]

existing.head(3)


# In[46]:


## Employees with Probability of Leaving
existing [existing["Prediction"]== 1]


# In[ ]:





# In[ ]:




