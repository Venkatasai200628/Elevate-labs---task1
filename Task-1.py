#1.Import the dataset and explore basic info (nu ls, data types).
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


df = pd.read_csv("D:\my stuff\elevate labs\Titanic-Dataset.csv")
print("Dataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nFirst 5 Rows:")
print(df.head())

# 2. Handle Missing Values
# Age: Fill with median
df['Age'].fillna(df['Age'].median(), inplace=True)
# Embarked: Fill with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# Cabin: Drop due to excessive missing values
df.drop('Cabin', axis=1, inplace=True)
print("\nMissing Values After Handling:")
print(df.isnull().sum())

# 3. Convert Categorical Features to Numerical
# Label Encoding for 'Sex'
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # male: 1, female: 0
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)  # Avoid dummy variable trap
# Drop non-relevant columns for ML
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
print("\nDataset After Encoding:")
print(df.head())

# 4. Normalize/Standardize Numerical Features
# Define numerical columns
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
minmax_scaler = MinMaxScaler()
df[numerical_cols] = minmax_scaler.fit_transform(df[numerical_cols])
# Display dataset after scaling
print("\nDataset After Scaling:")
print(df.head())

# 5. Visualize and Remove Outliers
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare Before Outlier Removal')
plt.savefig('fare_boxplot_before.png')
plt.show()
# Remove outliers using IQR for 'Fare'
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]
# Boxplot for 'Fare' after outlier removal
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare After Outlier Removal')
plt.savefig('fare_boxplot_after.png')
plt.show()
df.to_csv('cleaned_titanic.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_titanic.csv'")