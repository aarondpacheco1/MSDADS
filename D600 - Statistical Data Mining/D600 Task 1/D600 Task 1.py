import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#Import csv to Python
df=pd.read_csv(r'C:\Users\imret\Downloads\D600 Task 1, 2 and 3 Dataset Housing Information.csv')

#List all variables analyzed
description_vars = [
    'Price',
    'SquareFootage',
    'NumBathrooms',
    'NumBedrooms',
    'BackyardSpace',
    'CrimeRate',
    'SchoolRating',
    'AgeOfHome',
    'DistanceToCityCenter',
    'EmploymentRate',
    'RenovationQualityRate'
]

#Print descriptive statistics for all variables
print(df[description_vars].describe(include='all'))

#Plot histograms for each variable
for var in description_vars:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=var, kde=True, bins=30)
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

#List all independent nondiscrete variables
independent_nondiscrete_vars = [
    'SquareFootage',
    'BackyardSpace',
    'CrimeRate',
    'AgeOfHome',
    'DistanceToCityCenter',
    'EmploymentRate',
]

#Plot each independent non-discrete variable v. Price, the dependent variable in a scatter plot
for var in independent_nondiscrete_vars:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=var, y='Price')
    plt.title(f'{var} v. Price')
    plt.xlabel(var)
    plt.ylabel('Price')
    plt.tight_layout()
    plt.show()

#List all independent discrete variables
independent_discrete_vars = [
    'NumBathrooms',
    'NumBedrooms',
    'SchoolRating',
    'RenovationQualityRate'
]

#Plot each independent discrete variable v. Price, the dependent variable in a box & whisker plot
for var in independent_discrete_vars:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=var, y='Price')
    plt.title(f'{var} v. Price')
    plt.xlabel(var)
    plt.ylabel('Price')
    plt.tight_layout()
    plt.show()    

relevant_df = df[['Price',
    'SquareFootage',
    'NumBathrooms',
    'NumBedrooms',
    'BackyardSpace',
    'CrimeRate',
    'SchoolRating',
    'AgeOfHome',
    'DistanceToCityCenter',
    'EmploymentRate',
    'RenovationQualityRate']].copy()

#Count number of empty cells in each column of dataset
missing_values_per_column = relevant_df.isna().sum()
print(missing_values_per_column)

#Drop rows with missing values
relevant_df=relevant_df.dropna()

#Identify outliers
col_mean = relevant_df.mean()
col_std = relevant_df.std()

zscore = (relevant_df-col_mean)/col_std

#Create dataframe without outliers
clean_df = relevant_df[(zscore.abs()< 3).all(axis=1)]

#Split data into training set and test set
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(clean_df, test_size=0.3, random_state=83)

#Defining independent inputs and dependent output in training set
x_train = train_df.drop(columns=['Price'])
y_train = train_df['Price']

#Defining independent inputs and dependent output in training set 
x_test = test_df.drop(columns=['Price'])
y_test = test_df['Price']


#Perform Multiple Linear Regression and print summary of fit
x_train_sm = sm.add_constant(x_train)
model_sm = sm.OLS(y_train, x_train_sm).fit()
print(model_sm.summary())

#Build Backward Stepwise Function
def backward_stepwise(x, y):
    cols = list(x.columns)
    
    while True:
        #Fit the model
        x_sm = sm.add_constant(x[cols])
        model = sm.OLS(y, x_sm).fit()
        
        #Get p-values, excluding the constant
        pvalues = model.pvalues.drop('const')
        max_pvalue = pvalues.max()
        
        #If all p-values are below 0.05, stop
        if max_pvalue < 0.05:
            print("Final model reached.")
            break
        
        #Otherwise, remove the variable with the highest p-value
        removed = pvalues.idxmax()
        print(f"Removing: {removed} (p-value: {max_pvalue:.4f})")
        cols.remove(removed)
    
    print(model.summary())
    return model

#Perform backward stepwise elimination on training data
final_model = backward_stepwise(x_train, y_train)


#Filter test data to only the 6 significant variables
x_test_optimized = x_test[
    ['SquareFootage',
      'NumBathrooms',
      'NumBedrooms',
      'AgeOfHome',
      'DistanceToCityCenter',
      'RenovationQualityRate']]

#Add constant to match the trained model
x_test_sm = sm.add_constant(x_test_optimized)

#Generate predictions
y_pred = final_model.predict(x_test_sm)

#Calculate RMSE
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")

#Find mean to compare to RMSE
print(y_test.mean())
