import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np

#Import csv to Python
df=pd.read_csv(r'C:\Users\imret\Downloads\D600 Task 1, 2 and 3 Dataset Housing Information.csv')

#List all variables analyzed
description_vars = [
    'IsLuxury',
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

#List all independent variables
independent_vars = [
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

#Plot each independent variable v. IsLuxury, the dependent variable
for var in independent_vars:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='IsLuxury', y=var)
    plt.title(f'{var} v. IsLuxury')
    plt.xlabel('IsLuxury')
    plt.ylabel(var)
    plt.tight_layout()
    plt.show()

relevant_df = df[['IsLuxury',
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
train_df, test_df = train_test_split(clean_df, test_size=0.2, random_state=83, stratify=clean_df['IsLuxury'])

#Defining independent inputs and dependent output in training set
x_train = train_df.drop(columns=['IsLuxury'])
y_train = train_df['IsLuxury']

#Defining independent inputs and dependent output in training set 
x_test = test_df.drop(columns=['IsLuxury'])
y_test = test_df['IsLuxury']

#Perform Logistic Regression to Build Model
features = [
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
x_train_const = sm.add_constant(x_train[features])
logit_model = sm.Logit(y_train, x_train_const)
result = logit_model.fit(max_iter=1000)

print(result.summary())
print(f"AIC: {result.aic:.4f}")
print(f"BIC: {result.bic:.4f}")

#Optimize Model with Backward Elimination
def backward_stepwise(x, y):
    cols = list(x.columns)
    
    while True:
        #Fit the model
        x_sm = sm.add_constant(x[cols])
        model = sm.Logit(y, x_sm).fit(disp=0) 
        
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
    print(f"AIC: {model.aic:.4f}")
    print(f"BIC: {model.bic:.4f}")
    return model

#Perform backward stepwise elimination on training data
final_model = backward_stepwise(x_train, y_train)

#Add constant to test data and only include attributes deemed relevant following backward elimination
x_test_const = sm.add_constant(x_test[['SquareFootage', 'NumBathrooms', 'NumBedrooms', 'RenovationQualityRate']])

#Get probabilities
y_prob = final_model.predict(x_test_const)

#Convert probabilities to binary outputs
y_pred = (y_prob >= 0.5).astype(int)

#Create Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix = cm,
    display_labels = ['Not Luxury', 'Luxury']
)

disp.plot(cmap = 'Blues')
plt.title('Confusion Matrix - Optimized Logistic Regression')
plt.tight_layout()
plt.show()

#Ensure there is no multicollinearity in the data
from statsmodels.stats.outliers_influence import variance_inflation_factor

x_check = x_train[['SquareFootage', 'NumBathrooms', 'NumBedrooms', 'RenovationQualityRate']]
vif_df = pd.DataFrame({
    'Feature': x_check.columns,
    'VIF': [variance_inflation_factor(x_check.values, i)
            for i in range(x_check.shape[1])]
})
print(vif_df)

#Ensure linearity between the independent variables and the log-odds of the outcomes via Box-Tidwell test
x_check = x_train[['SquareFootage', 'NumBathrooms', 'NumBedrooms', 'RenovationQualityRate']].copy()

#Create interaction terms of each variable with its log
for col in x_check.columns:
    x_check[f'{col}_log'] = x_check[col] * np.log(x_check[col] + 1)

x_bt = sm.add_constant(x_check)
bt_model = sm.Logit(y_train, x_bt).fit(max_iter=1000)
print(bt_model.summary())

#Accuracy Results
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['Not Luxury', 'Luxury']))