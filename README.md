# Elevate-labs---task1
Overview
This project focuses on cleaning and preprocessing the Titanic dataset to prepare it for machine learning tasks. It is part of an AI & ML internship task to demonstrate data wrangling, encoding, feature scaling, and outlier handling techniques.

Objectives
Understand the structure and content of the dataset
Handle missing values appropriately
Convert categorical variables into numerical format
Normalize/standardize numerical features
Visualize and remove outliers to ensure data quality

Dataset
The dataset used is the Titanic Dataset

Tools & Libraries Used
Python
Pandas
NumPy
Matplotlib & Seaborn
Scikit-learn

Steps Performed
1. Data Exploration
Loaded the dataset using Pandas.
Explored column data types and missing values.

2. Handling Missing Values
Replaced missing Age values with the median.
Replaced missing Embarked values with the mode.
Dropped the Cabin column due to excessive nulls.

3. Encoding Categorical Features
Converted Sex using Label Encoding (male = 1, female = 0).
Applied one-hot encoding to the Embarked column.
Dropped Name, Ticket, and PassengerId as they are not useful for ML.

4. Feature Scaling
Applied Min-Max scaling to numerical features: Age, Fare, SibSp, and Parch.

5. Outlier Detection & Removal
Visualized Fare outliers using a boxplot.Removed outliers using the Interquartile Range (IQR) method.

6. Export
The cleaned dataset was saved as cleaned_titanic.csv.

Visualizations
Boxplots for the Fare column were generated before and after outlier removal and saved as:
test-1figure1.png
test-1figure2.png

Conclusion
This task provided a hands-on experience in data preprocessing—an essential step in any ML workflow. The cleaned dataset is now ready for further analysis or model building
