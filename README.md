<H3>ENTER YOUR NAME : RIYA P L</H3>
<H3>ENTER YOUR REGISTER NO : 212223240141</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 21.04.2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('Churn_Modelling.csv')
print(df)

print(df.isnull().sum())

df.fillna(df.select_dtypes(include='number').mean().round(1), inplace=True)
print(df.isnull().sum())

y = df.iloc[:, -1].values
print(y)

df.duplicated()

df['EstimatedSalary'].describe()
scaler = StandardScaler()
inc_cols = ['CreditScore', 'Tenure', 'Balance', 'EstimatedSalary']
scaled_values = scaler.fit_transform(df[inc_cols])
df[inc_cols] = pd.DataFrame(scaled_values, columns = inc_cols, index = df.index)
df

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("X Values")
x
print("Y Values")
y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print("X Training data")
x_train
print("X Testing data")
x_test

print(len(x_train))
print(len(x_test))
```


## OUTPUT:
<img width="826" height="505" alt="Screenshot 2026-04-21 134535" src="https://github.com/user-attachments/assets/c1429a40-8c86-49d9-95e5-ef6ebc3fff81" />

<img width="625" height="288" alt="Screenshot 2026-04-21 134555" src="https://github.com/user-attachments/assets/d81e9e26-2724-4911-9a14-2905b6cb5d4b" />

<img width="567" height="403" alt="Screenshot 2026-04-21 134602" src="https://github.com/user-attachments/assets/8f89596f-7a4e-45c3-bc95-53cc8498bba4" />

<img width="872" height="487" alt="Screenshot 2026-04-21 134607" src="https://github.com/user-attachments/assets/b7e18ad3-3680-4b9e-8b13-f4fac6d4c217" />

<img width="706" height="595" alt="Screenshot 2026-04-21 134614" src="https://github.com/user-attachments/assets/0745247e-e080-400e-8689-86dd616a2436" />

<img width="1563" height="486" alt="Screenshot 2026-04-21 134621" src="https://github.com/user-attachments/assets/6f08ec78-e400-48a3-87df-930b2596f5bf" />

<img width="592" height="573" alt="Screenshot 2026-04-21 134630" src="https://github.com/user-attachments/assets/e50e72ff-8a22-4f26-a6e0-7aa5d92d74e7" />

<img width="1535" height="527" alt="Screenshot 2026-04-21 134638" src="https://github.com/user-attachments/assets/d09c0646-38df-466f-9021-1ea3b75adf74" />

<img width="395" height="244" alt="Screenshot 2026-04-21 134643" src="https://github.com/user-attachments/assets/14e38657-bca5-433e-8cba-b835d977efbc" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


