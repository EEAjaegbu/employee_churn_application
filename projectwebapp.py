#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Load libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
import xlrd


# In[11]:


np.random.seed(0)


# In[3]:


#load the datasets for Employee who has left.
#left = pd.read_excel("Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx",sheet_name=2)
#left.head(4)

left = pd.read_csv("employee_who_left.csv")
left.head(4)




# In[4]:


#load the datasets for Employee who is still existing.
#existing = pd.read_excel("Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx",sheet_name=1)

existing = pd.read_csv("existing_employee.csv")


# In[5]: In[5]:


## Add the atrribute Churn to Existing Employeee dataset
existing['Churn']= 'No'
existing.head(2)


# In[6]:


## Add the attribute churn to Employee who has left dataset
left['Churn']='Yes'
left.head(2)


# In[7]:


## Combining left and existing Dataframes together to create a single dataframes.
employee_attrition =  pd.concat([left, existing], ignore_index=True)
employee_attrition.head(10)


# In[9]:

#### Tile For the Web App
st.write(
         """
         # Employee Attrition Streamlit Web App 
         This Program is developed using the machine learning Algorithm,
         it predicts the Employee who will leave the organization
         with Approximately 99% Accuracy
         """)
if st.button("About Author"):
    st.write("""
        Name: Ajeegbu Ebuka Emmanuel
         """)
    st.write("""
         Email: ajaegbu35@gmail.com
         """)

# In[12]:

## Display
image = Image.open("employee_attrition.png")

st.image(image,use_column_width=True)

### Subtitile
st.subheader('Dataset:')

st.dataframe(employee_attrition.sample(frac=1).head(10))

##
st.header(
    """
    Exploratory Data Analysis of Employee Who Left
    The Descriptive Summary of the Employee who have left based on Departmemt
    """
)

### Number of Employee who have left in Each department
print(left['dept'].value_counts())

## Chart
fig1,ax= plt.subplots(figsize=(12,5))
ax.bar(left['dept'].value_counts().index,left['dept'].value_counts().values)
plt.ylabel("Number of Employee who have left")
plt.xlabel("Departments")
plt.title("What departments are they from?")
plt.grid()
st.pyplot(fig1)


# ## Data Preprocessing

# ### a) Checking For Missing Values

# Removing Redindant Variables
employee_attrition.drop('Emp ID', axis=1, inplace=True)
# Checking For Missing Values
employee_attrition.isnull().sum()


# ### b) Enconding All Categorical Variables

## Encoding The  Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le= LabelEncoder()


# In[18]:



# Label Encoding will be used for columns with 2 or less unique values
le_count = 0
for col in employee_attrition.columns:
    if employee_attrition[col].dtype == 'object':
        le.fit(employee_attrition[col])
        employee_attrition[col] = le.transform(employee_attrition[col])
        le_count += 1


print('{} columns were label encoded.'.format(le_count))

# In[20]:


## Selceting the Independent and Dependent Variables
X = employee_attrition.iloc[:,[0,1,2,3,4,5,6,7,8]].values ### Matrix of Independent faetures
y = employee_attrition.iloc[:,9].values    ### Vector of target varible 


# In[21]:


### Splitting the data into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30, random_state= 1000)


# ##   Training and Testing the Model

# In[22]:


from sklearn.metrics import accuracy_score,classification_report

# #### b) Random forest Classifier

# In[33]:

st.subheader(
    
"""
Evaluation Metrics
""")
from sklearn.ensemble import RandomForestClassifier

## Trainign the Model
rforest = RandomForestClassifier()
rforest.fit(X_train,y_train)

## Testing the Model
y_pred_rforest = rforest.predict(X_test)

## Classification Report

st.write("Classification Report:")
st.write(classification_report(y_test,y_pred_rforest))
 
# ### Predicting the Existing Employee with the Probability of leaving
st.sidebar.subheader(
"""
Predict Employee Who Will Leave
Enter Value For the Features Below
""")
def user_input():
    Satisfaction_level = st.sidebar.number_input("Satisfaction level",min_value=0.00, max_value= 0.99,value=0.5)
    Last_evaluation = st.sidebar.number_input("Last Evaluation",min_value=0.00, max_value= 0.99,value=0.5)
    number_project =st.sidebar.number_input('Number of project',min_value=0, max_value= 10,value=5)
    average_montly_hours = st.sidebar.number_input('The average montly hours',min_value=0.00, max_value= 1000.00,value=300.00)
    time_spend_company  = st.sidebar.number_input('Time spend in company',min_value=0, max_value= 20,value=5)
    Work_accident =st.sidebar.selectbox('Work accident',(0, 1))
    promotion_last_5years = st.sidebar.selectbox('Promotion last 5 years',(0, 1))
    dept = st.sidebar.selectbox('Department',("sales","technical","support","IT","hr","accounting","marketing","product_mng","randD","mangement"))
    Salary =  st.sidebar.selectbox('Salary Level ',("low","medium","high"))
    
    ### Dictionaries of Input
    input_user= {"Satisfaction_level":Satisfaction_level ,"Last_evaluation":Last_evaluation, "number_project":number_project,"average_montly_hours":average_montly_hours,"time_spend_company":time_spend_company,"Work_accident":Work_accident,"promotion_last_5years":promotion_last_5years, "dept":dept,"Salary":Salary}
               
    ### Cpnverting to a Dataframes
    input_user =pd.DataFrame(input_user,index=[0])
    return input_user

input_value = user_input()                               

print(input_value.info())
        
# Label Encoding will be used for columns with 2 or less unique values

## Encoding The  Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le1= LabelEncoder()

le1_count = 0
for col in input_value.columns:
    if input_value[col].dtypes == 'object':
        le1.fit(input_value[col])
        input_value[col] = le1.transform(input_value[col])
        le1_count += 1


print('{} columns were label encoded.'.format(le1_count))

if st.sidebar.button("Predict"):
    Prediction = rforest.predict(input_value)
    if Prediction == 0:
        result = pd.DataFrame({"Churn":Prediction,"Info":"The Employee will not Leave the Ogarnization"})
    else:
        result = pd.DataFrame({"Churn":Prediction,"Info":"The Employee wil Leave the Ogarnization"})                      
    
    st.write("""
    # The Result of the Classification:
    """)
    st.write("Attrition : ")  
    st.dataframe(result)
                                

   



