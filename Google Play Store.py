#!/usr/bin/env python
# coding: utf-8

# # Google Play Store

# * This data set is located in  https://github.com/gauthamp10/Google-Playstore-Dataset
# * We will assist Google in determining which applications work best on their platform in order to increase user engagement.
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import statistics as stats
import matplotlib.pyplot as plt 
import statistics as stats
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import scipy as sp 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc


# In[2]:


df=pd.read_csv('Master file final.csv')


# In[3]:


df.head()


# ## Descriptive Analysis

# ### Number of columns and rows

# In[4]:


#Number of columns and rows
print('Rows:',list(df.shape)[0])
print('Columns:',list(df.shape)[1])


# ### Percent of missing values by Columns

# In[5]:


# % of missing values by columbs
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
missing_value_df


# ### Mean, Standard Deviation, Min and Max using Descriptive Statistics

# In[6]:


#Summary statistics
df.describe()


# ### Median

# In[7]:


#Medians of each numerical variable
pd.DataFrame(df.median()).rename(columns={0:'Median_Value'})


# ### Mode

# In[8]:


#Modes of each variable

modes = []
for i in list(df.columns):
    modes.append([i,list(df[i].mode())[0]])

df_modes = pd.DataFrame(modes).rename(columns={0:'Variable',1:'Mode'}).set_index(keys='Variable')
df_modes


# ### Variance

# In[9]:


#Variance of each variable
variances = []
df_gpdescribe = pd.DataFrame(df.describe())
for i in df_gpdescribe.columns:
    variances.append([i,pow(list(df_gpdescribe[i])[2],2)])

df_variances = pd.DataFrame(variances).rename(columns={0:'Variable',1:'Variance'}).set_index(keys='Variable')
df_variances


# ### Correlation Matrix

# In[10]:


df.corr()


# ## Data cleaning and processing

# In[11]:


#Data visualization of missing values 
sns.heatmap(df.isnull(),cbar=False)


# In[12]:


# Creating a table of df type. null values and unique values for better visualization
def printinfo():
    temp = pd.DataFrame(index= df.columns)
    temp['data_type'] = df.dtypes
    temp['null_count'] =df.isnull().sum()
    temp['unique_count'] = df.nunique()
    return temp


# In[13]:


printinfo()


# #### From this point of view we consider that size category should be numeric

# ### Fixing Size

# ### We observe: 
# * Metabytes 
# * Kilobyte
# * Gigabyte
# * Varies with device
# ### We will transform all data to Metabytes

# In[14]:


for i in df['Size']:
    if type(i)==float:
        pass
    elif i[-1]=='k':
        pass
    elif i[-1]=='M':
        pass
    elif i=='Varies with device':
        pass
    else:
        print(i)


# In[15]:


x=[]
for i in df['Size']:
    if type(i)==float:
        i=i
    elif i=='Varies with device':
        i=np.nan
    elif i[-1]=='M':
        i=float(i[:-1].replace(',',''))
    elif i[-1]=='k':
        i=(float(i[:-1].replace(',','')))/1024
    elif i[-1]=='G':
        i=(float(i[:-1]))*1024
    x.append(i)


# In[16]:


df['Size']=np.array(x)
df['Size']


# In[17]:


printinfo()


# ### Fixing Install Columns

# In[18]:


df['Installs']


# #### We can observe it contains a + at the end. We proceed to erase it

# In[19]:


#Removing + at the end of installs
df['Installs']=df['Installs'].str.replace(r"[^a-zA-Z\d\_]+","")


# In[20]:


#Identifying NaN values in Installs in order to transform it to numeric variable
print(df[df['Installs'].isnull()])


# In[21]:


#Removing all non-numeric variables
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')


# In[22]:


#Removing all rows with Nan
df = df.dropna(subset=['Installs'])


# In[23]:


#Transforming Installs 
df['Installs']=df['Installs'].astype('int64')


# In[24]:


printinfo()


# ### Installs has been sucessfuly change into int

# ## Transforming Minimum Installs into Int

# In[25]:


df.head()


# In[26]:


#Transforming minimum Installs
df['Minimum Installs']=df['Minimum Installs'].astype('int64')


# In[27]:


df.head()


# ## Dropping Numerical Variables with Zero variance

# In[28]:


# Checks if there is any variables with zero variance
df.std()


# In[29]:


# Drops variables with 0 variance
df = df.drop(df.std()[df.std() == 0].index, axis = 1) 


# In[30]:


# Checks if there is any variables with zero variance
df.std() 


# #### We have not found numerical variables with zero variance

# ### Dropping Categorical Variables with Zero variance

# In[31]:


#Reviewing the colums
columns = df.columns
columns


# In[32]:


df.head()


# In[33]:


#Looking for Categorical columns missing values
   
categorical_var = list(set(df.dtypes[df.dtypes == object].index) - set(['Free'])) #appended categorical items that were showing as numerical for some reason
categorical_var


# In[34]:


zero_cardinality = []    

for i in categorical_var: # for each categorical variables
    if len(df[i].value_counts().index) == 1: # check how many levels it has and if it is one
        zero_cardinality.append(i) # the variable has zero variance as the cardinality is one 
        # append it to the list of categorical variables with zero variation
        
df = df.drop(zero_cardinality, axis = 1)


# In[35]:


zero_cardinality


# #### We have not found categorical variables with zero cardinality

# ### Dropping Categorical Variables with Many Levels

# In[36]:


#High_cardinality - Variables with more than 50000 values 
high_cardinality = [] 
for i in categorical_var: # for each categorical variables
    if len(df[i].value_counts().index) > 50000: # check how many levels it has and if it is more
        high_cardinality.append(i) # than 50000, variable has many levels
        # so append it to the list of categorical variables with high cardinality
        
print(high_cardinality)


# #### We consider that App Id and App Name are important and we have decide not to drop them

# In[37]:


# Drops variables with high cardinality
df = df.drop(columns=['Developer Id', 'Developer Website', 'Privacy Policy', 'Developer Email','Scraped Time', 'Unnamed: 0'])


# #### We have also decided to drop Scraped Time and Unamed 0 since it does not contribute to our analysis 

# ### Filling numerical variables with their mean

# In[38]:


df.head()


# In[39]:


df.columns


# In[40]:


printinfo()


# In[41]:


#Looking for True numerical columns missing values

numerical_var = set(df.columns) - set(df.dtypes[df.dtypes == object].index)
numerical_var = list(numerical_var - set(['In App Purchases','Editors Choice','Ad Supported','Free'])) #the attributes describe here are categorical and that is the reason I am adding them here
numerical_var


# In[42]:


#Filling Missing Values for numerical variables

df[numerical_var] = df[numerical_var].fillna(df[numerical_var].median(), inplace = False)


# In[43]:


printinfo()


# In[44]:


df.head()


# ### Filling categorical variables with their mode

# In[45]:


#Looking for Categorical columns missing values
   
categorical_var = list(set(df.dtypes[df.dtypes == object].index) - set(['Free'])) #appended categorical items that were showing as numerical for some reason
categorical_var


# In[46]:


# Fills in the missing values in categorical columns with mode
# and overwrites the result into the esxisting dataset
df[categorical_var] = df[categorical_var].fillna(df[categorical_var].mode(), inplace = False)


# In[47]:


printinfo()


# In[48]:


df.head()


# ### Dropping missing values

# In[49]:


df.dropna(inplace=True)   #dropping missing values for analysis purposes
df.info()


# In[50]:


# Checks the number of missing values by column
[sum(df[i].isnull()) for i in df.columns] 


# ### Remove duplicates Using App Id

# In[51]:


df.duplicated(subset=['App Id'])


# In[52]:


df.drop_duplicates(subset=['App Id'])


# In[53]:


#Number of columns and rows
print('Rows:',list(df.shape)[0])
print('Columns:',list(df.shape)[1])


# ### Data analysis

# ### Check for outliners

# In[54]:


#Transforming Released into float
df['Released']=df['Released'].astype('float')

#Transforming Released into integrer
df['Released']=df['Released'].astype('int64')


# In[55]:


#Identifying numeric vairables 

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

df_numeric = df.select_dtypes(include=numerics)
df_numeric


# In[56]:


#Indentify Interquartile Range
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

#Create DataFrame that identifies outliers variables based on Interquartile Range.
#Outlier values are indicated by 'True'
df_gpoutliers = (df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))
df_gpoutliers


# In[57]:


#Summarize the number of Outliers in each variable
outliers = []
for i in list(df_gpoutliers.columns):
    outliers.append([i, sum(list(df_gpoutliers[i]))])

df_outliercount = pd.DataFrame(outliers).rename(columns={0:'Variable',1:'Outlier_QTY'}).set_index(keys='Variable')

outlier_percentage = []
for i in list(df_gpoutliers.columns):
    outlier_percentage.append([sum(list(df_gpoutliers[i]))/len(list(df[i]))])
    
df_outliercount['Outlier_QTY_Percentage'] = outlier_percentage

df_outliercount


# In[58]:


#Visual representation of outliers in Installs
import seaborn as sns
sns.boxplot(x=df['Installs'])


# In[59]:


#Visual representation of outliers in Maximum Installs
import seaborn as sns
sns.boxplot(x=df['Maximum Installs'])


# In[60]:


#Visual representation of outliers in Rating Count
import seaborn as sns
sns.boxplot(x=df['Rating Count'])


# ### Data Visualization

# In[61]:


df.head()


# In[62]:


#Visualization of Maximum Installvs vs Free
sns.boxplot(x= 'Free',y='Maximum Installs',data = df[['Free','Maximum Installs']],linewidth=.5)


# In[63]:


#desrciptive analysis of application which supports Advertisements vs application which does not support advertisement 
df['Ad Supported'].value_counts().index 
df['Ad Supported'].value_counts().values

sns.set(style = 'darkgrid')
ax = sns.barplot(x=df['Ad Supported'].value_counts().index,
                 y=df['Ad Supported'].value_counts().values,
                 color='green')


# In[64]:


#data statistics
df.describe()


# In[65]:


df['Content Rating'].value_counts()


# In[66]:


# We noticed that the categories with the most apps are: Education, Music & Audio and Tools
# We noticed that the cateogires with the least apps are: Parenting and Comics
plt.figure(figsize=(12,10))
most_cat = df['Category'].value_counts()
sns.barplot(x=most_cat, y=most_cat.index, data=df)
plt.xticks(size=10)
plt.xlabel("Frequency",size=24,c="g")
plt.ylabel("Category",size=24,c="g")
plt.title("Total Apps of all Categories",size=28,c="r")
plt.show()


# In[67]:


# We noticed that the top three apps with the highest ratings are: Action, Strategy and Racing
print()
plt.figure(figsize=(14,12))
df.groupby('Category')['Rating Count'].mean().round(0).sort_values(ascending=True).plot(kind="barh")
plt.xticks(size=15)
plt.xlabel("Frequency",size=22,c="g")
plt.ylabel("Category",size=22,c="g")
plt.title("All Categories & Rating Count",size=26,c="r")
plt.show()


# In[68]:


# We noticed that the top three categories with the highest installs are:  Video Players and Editors, Action and Racing
print()
plt.figure(figsize=(14,12))
df.groupby('Category')['Installs'].mean().round(0).sort_values(ascending=True).plot(kind="barh")
plt.xticks(size=15)
plt.xlabel("Frequency",size=22,c="g")
plt.ylabel("Category",size=22,c="g")
plt.title("All Categories & Installs",size=26,c="r")
plt.show()


# In[69]:


# We can visualize most of the apps are free or low price
# We can visualize that the most expensive apps are related to Tools and Business.

plt.figure(figsize=(14,12)) 
sns.scatterplot(data=df,y="Category",x='Price',color="b")
plt.xticks(rotation='vertical',size=15)
plt.xlabel("Price",size=22,c="g")
plt.ylabel("Category",size=22,c="g")
plt.title("Visualization of Paid App",size=26,c="r")
plt.show()


# In[70]:


# We can visualize that highest Content Rating is Everyone followed by Teen and Mature 17+

plt.figure(figsize=(14,12))

sns.countplot(x="Content Rating",data=df)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlabel("Content Rating",size=22,c="g")
plt.ylabel("Frequency",size=22,c="g")
plt.title("Content Rating",size=26,c="r")
plt.show()


# In[71]:


#We can visualize on the heatmap below that installs and maximum installs has good relation

plt.figure(figsize=(10,7))
sns.heatmap(df[["App Name","Installs","Category","Free","Price","Maximum Installs"]].corr(), annot=True,linewidths=.4,fmt='.1f')
plt.title("Correlation Graph",c="r",size=25)
plt.show()


# In[72]:


# We can visualize the total % of free apps in store in pie chart
plt.figure(figsize=(8,8))
labels =df['Free'].value_counts(sort = True).index
sizes = df['Free'].value_counts(sort = True)
plt.pie(sizes, labels=labels,autopct='%1.1f%%', shadow=True, startangle=270,)
plt.title('Total % of Free App in store',size = 20)
plt.show()


# In[73]:


# We can visualize the install vs. rating boxplot
ax = plt.figure(figsize=(15,8))
sns.boxplot(x="Installs", y="Rating", data=df)
plt.title("Installs vs Rating",size=25,c="r")
plt.xticks(size=15,rotation=90)
plt.yticks(size=15)
plt.xlabel("Installs",size=20)
plt.ylabel("Rating",size=20)
plt.show()


# In[74]:


# We can visualize the distribution of Rating apps
plt.figure(figsize=(11,7))
plt.subplot(1,1,1)
sns.distplot(df['Rating'],color='r',kde_kws={'linewidth':3,'color':'b'});
plt.show()


# In[75]:


# We can visualize Rating, Size Price boxplot to find the outliers
col = ['Rating','Size','Price'] 
plt.figure(figsize=(18,12))
for i,v in enumerate(col):
    print(i,v)
    plt.subplot(3,2,i+1)
    sns.boxplot(x=v, data=df)
plt.show()


# In[76]:


# We can visualize that nearly all of the app size are under 200M
plt.hist(df['Size'],bins=10)
plt.xlabel("Size",size=11)
plt.ylabel("Frequency",size=11)
plt.show()


# In[77]:


# We can visualize the average price of all app categories by bar plots
plt.figure(figsize=(15,7))
df.groupby("Category")['Price'].mean().sort_values(ascending=False).plot(kind="bar")
plt.xlabel("Category",size=15,c="r")
plt.ylabel("Avg Cost",size=15,c="r")
plt.title("Avg Price of all Categories",size=28,c="k")


# ### Building a linear regression model 

# ####  Choosing a sample of 5000

# In[78]:


#Renaming Index
df.index.names = ['ID']

# Creating a random sample of 5000 records for the linear model  
df_sample = df.sample(5000, random_state=52)

df_sample


# In[79]:


# Review correlation 
df_sample.corr()


# ### Analysing Target Variable - 'Maximum Installs'

# #### We can observe that maximum installs has a high correlation with rating count, installs and minimum installs

# In[80]:


# Creating a new data frame for the regression
df_InstallsRegression=df_sample.copy()
df_InstallsRegression = df_InstallsRegression[['Installs','Maximum Installs','Minimum Installs','Rating','Rating Count',]]
df_InstallsRegression.head()


# ### Scale Independent Numerical Variables

# In[81]:


# Creating a random sample of 5000 records for the linear model  
df_sample = df.sample(5000, random_state=52)


# In[89]:


df_gpInstallsRegression=df_sample.copy()
df_gpInstallsRegression = df_gpInstallsRegression[['Maximum Installs','Minimum Installs','Rating','Rating Count']]
df_gpInstallsRegression.head()


# In[90]:


#Create a list of the numerical variables to scale (exclude dependents)
to_scale =  ['Minimum Installs','Rating Count','Rating']

#To use this library, we need to convert a pandas dataframe into a numpy array by doing the following:
InstallsRegressionarray = df_gpInstallsRegression[to_scale].values

#Create a min max scaler - this sets all values between 0 and 1 within the numpy array
data_scaler = StandardScaler().fit(InstallsRegressionarray) 

#Apply the scaler and overwrite the data into the existing dataframe
df_scaled = pd.DataFrame(data_scaler.fit_transform(InstallsRegressionarray), columns = to_scale)

df_gpInstallsRegression['Minimum Installs'] = df_scaled['Minimum Installs'].values
df_gpInstallsRegression['Rating Count'] = df_scaled['Rating Count'].values

df_gpInstallsRegression.head()


# In[91]:


#Analizing the target variable
print(df_gpInstallsRegression['Maximum Installs'].describe()) # describing output (target) variable
print('\n') # new line command
ax = sns.distplot(df_gpInstallsRegression['Maximum Installs']) #checking the distribution of the output variable


# ### Scatter Plots: Relationship with Numerical Variables

# In[92]:


#scatter plot to check how selected two variables are related to saleprice

fig, axs = plt.subplots(ncols=2, figsize= (8, 4)) # Divides the plotting area into sub-ares 
                                 
plt.subplot(1, 2, 1) # indicates there are one row two columns and this is the first plot
ax = sns.scatterplot(x='Minimum Installs', y='Maximum Installs', data=df_gpInstallsRegression) # scatter plot 

plt.subplot(1, 2, 2) # indicates there are one row two columns and this is the second plot
ax = sns.scatterplot(x='Rating', y='Maximum Installs', data=df_gpInstallsRegression) # scatter plot 


# ### Fiting the model

# In[93]:


#Create and defining the target variable
target_col= 'Maximum Installs'

#Determing the input variables
input_col= list(set(df_gpInstallsRegression.columns)-set(['Maximum Installs']))

# Creating a linear regression function
model = LinearRegression()

model.fit(df_gpInstallsRegression[input_col],df_gpInstallsRegression[target_col])

#Prints the model coefficients
print('Model coefficients:')
print(model.coef_)

#Prints the R2 value of the model
print('\n')
print('R2 value:'+ str(round(model.score(df_gpInstallsRegression[input_col],df_gpInstallsRegression[target_col]),2)))
print('\n')

#Calculating the residuals and then print the results
pred_vs_actual= pd.DataFrame()
pred_vs_actual['actual']=df_gpInstallsRegression[target_col]
pred_vs_actual['predicted']=np.round(model.predict(df_gpInstallsRegression[input_col]),6)
pred_vs_actual['error']=pred_vs_actual['actual']-pred_vs_actual['predicted']
print(pred_vs_actual.head())


# ### Assesing the model

# In[94]:


fig, ax = plt.subplots(figsize=(12,6)) # Determines the figure size
ax = sns.scatterplot(x='error', y='predicted', data=pred_vs_actual) 
#plots the error vs. predicted
limits = ax.set(xlim=(-50000, 50000), ylim=(0, 400000)) #sets the limit for x and y axis


# In[95]:


fig, ax = plt.subplots(figsize=(12,6)) # Determines the figure size


_, (__, ___, r) = sp.stats.probplot(pred_vs_actual['error'], plot=ax, fit=True) 
# generates the normality plot

limits = ax.set(ylim=(-100000, 200000)) # sets the limits for the graph


# ### Reviewing R and errors

# In[96]:


P=df_gpInstallsRegression.loc[:,df_gpInstallsRegression.columns!=target_col]
R=df_gpInstallsRegression.loc[:,target_col]


# In[97]:


P=sm.add_constant(P)
model=sm.OLS(R,P)


# In[98]:


results=model.fit()
results.params


# In[99]:


# Summary Table - Regression Model
print(results.summary())


# ### Multiple Regression Model - Rating Count

# In[100]:


df_RatingsRegression=df_sample.copy()
df_RatingsRegression = df_RatingsRegression[['Rating Count','Maximum Installs','Minimum Installs','Rating']]
df_RatingsRegression.head()


# ### Scale Independent Numerical Variables

# In[101]:


to_scale =  ['Maximum Installs','Minimum Installs','Rating']

#To use this library, we need to convert a pandas dataframe into a numpy array by doing the following:
RatingsRegressionarray = df_RatingsRegression[to_scale].values

data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

df_scaled = pd.DataFrame(data_scaler.fit_transform(RatingsRegressionarray), columns = to_scale)

df_RatingsRegression['Minimum Installs'] = df_scaled['Minimum Installs'].values
df_RatingsRegression['Maximum Installs'] = df_scaled['Maximum Installs'].values

df_RatingsRegression.head()


# ### Fiting the model

# In[102]:


#Determine the model

target = 'Rating Count'

#determines the input variables
input_col= list(set(df_RatingsRegression.columns)-set(['Rating Count']))

model= LinearRegression()

model.fit(df_RatingsRegression[input_col],df_RatingsRegression[target])

#prints the model coefficients
print('Model coefficients:')
print(model.coef_)

# prints the R2 value of the model
print('\n')
print('R2 value:'+ str(round(model.score(df_RatingsRegression[input_col],df_RatingsRegression[target]),2)))
print('\n')
# calculate the residuals and print the results
pred_vs_actual= pd.DataFrame()
pred_vs_actual['actual']=df_RatingsRegression[target]
pred_vs_actual['predicted']=np.round(model.predict(df_RatingsRegression[input_col]),6)
pred_vs_actual['error']=pred_vs_actual['actual']-pred_vs_actual['predicted']
print(pred_vs_actual.head())


# ### Assesing the model

# In[103]:


fig, ax = plt.subplots(figsize=(12,6)) # Determines the figure size
ax = sns.scatterplot(x='error', y='predicted', data=pred_vs_actual) 
#plots the error vs. predicted
limits = ax.set(xlim=(-50000, 50000), ylim=(0, 400000)) #sets the limit for x and y axis


# In[104]:


fig, ax = plt.subplots(figsize=(12,6)) # Determines the figure size


_, (__, ___, r) = sp.stats.probplot(pred_vs_actual['error'], plot=ax, fit=True) 
# generates the normality plot

limits = ax.set(ylim=(-100000, 200000)) # sets the limits for the graph


# ### Logistic Regression - Free or Paid

# In[105]:


#Create a list of variables that will categorized (Category & Content Rating)
categorical_dummies = list(set(df[['Category', 'Content Rating','Ad Supported','In App Purchases','Editors Choice']].columns))

#Create dummy variables using onehot encoding
dummy_cat_df = pd.get_dummies(df[categorical_dummies], drop_first=True) 

#Drops categorical variables from the df_googleplay
df = df.drop(categorical_dummies, axis = 1) 

#Adds the newly created dummy variables
df = pd.concat([df, dummy_cat_df], axis = 1)

df.head()


# In[106]:


# Create dummy variables using onehot encoding

dummy_free = pd.get_dummies(df, columns=['Free'], prefix=None, prefix_sep='.', drop_first=True)


# In[107]:


dummy_free.head()


# In[108]:


#Rename it 
df.rename(columns = {'Free.True':'Free'}, inplace = True)


# In[109]:


df.head()


# In[110]:


list(df.columns)


# In[111]:


df['Free']


# In[112]:


dummy_free.head()


# ### Scale Independent variables

# In[113]:


df_LogisticRegression=dummy_free.copy()
df_LogisticRegression = df_LogisticRegression[['Free.True', 'Minimum Installs','Maximum Installs','Rating Count','Installs','Price','Size']]
df_LogisticRegression.head()


# In[114]:


#Creating a list of the numerical variables
to_scale =  ['Rating Count','Minimum Installs','Maximum Installs', 'Price', 'Size']

LogisticRegressionarray = df_LogisticRegression[to_scale].values

#Min Max scaler
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

#Overwriting the data 
df_scaled = pd.DataFrame(data_scaler.fit_transform(LogisticRegressionarray), columns = to_scale)

df_LogisticRegression['Rating Count'] = df_scaled['Rating Count'].values
df_LogisticRegression['Minimum Installs'] = df_scaled['Minimum Installs'].values
df_LogisticRegression['Maximum Installs'] = df_scaled['Maximum Installs'].values
df_LogisticRegression['Price'] = df_scaled['Price'].values
df_LogisticRegression['Size'] = df_scaled['Size'].values


df_LogisticRegression.head()


# ### Fiting the model

# In[115]:


X=df_LogisticRegression.loc[:,['Size','Minimum Installs','Maximum Installs','Rating Count']]
Y=df_LogisticRegression.loc[:,'Free.True']


# In[116]:


# Test train split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size= 0.70, random_state=42)


# In[117]:


lr=LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
lr.fit(X_train,Y_train)


# In[118]:


def get_performance (actual_Y, pred_Y):
    cm=confusion_matrix(actual_Y, pred_Y)
    total=sum(sum(cm))
    accuracy= (cm[0,0]+cm[1,1])/total
    sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])
    specificity= cm[1,1]/(cm[1,0]+cm[1,1])
    return accuracy, sensitivity, specificity


# In[119]:


pred_Y_lr=lr.predict(X_test)
print(pred_Y_lr)


# In[120]:


accuracy_lr,sensitivity_lr,specificity_lr=get_performance(Y_test, pred_Y_lr)


# ### Assesing the model

# #### Accuracy table

# In[121]:


perf=pd.DataFrame ([accuracy_lr],columns=['accuracy'],index=['Logistic Regression'])
perf['sensitivity']= np.asarray([sensitivity_lr])
perf['specificity']=np.asarray([specificity_lr])
perfI 


# In[ ]:


Y_test.values


# In[ ]:


score_Y_dt=lr.predict_proba(X_test)
score_Y_dt


# ### ROC Curve

# In[ ]:


score_Y_dt=lr.predict_proba(X_test)
fpr,tpr,_=roc_curve(Y_test,score_Y_dt[:,1])
roc_auc= auc(fpr,tpr)

plt.figure()
lw=2
plt.plot(fpr,tpr, color='darkorange', lw=lw, label='ROC curve (area=%0.2f)'% roc_auc)
plt.plot([0,1],[0,1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# ### Confusion Matrix

# In[122]:


cmlogit = confusion_matrix(Y_test,pred_Y_lr)
plt.figure(figsize=(10,7))
sns.heatmap(cmlogit, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

