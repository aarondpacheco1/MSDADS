import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler

#Import csv to Python
df=pd.read_csv(r'C:\Users\imret\Downloads\D600 Task 1, 2 and 3 Dataset Housing Information.csv')

#List all variables analyzed
description_vars = [
    'SquareFootage',
    'NumBathrooms',
    'BackyardSpace',
    'CrimeRate',
    'SchoolRating',
    'AgeOfHome',
    'DistanceToCityCenter',
    'EmploymentRate',
    'RenovationQualityRate',
    'LocalAmenities',
    'TransportAccess',
    'PreviousSalePrice'
]

relevant_df = df[['Price',
    'SquareFootage',
    'NumBathrooms',
    'BackyardSpace',
    'CrimeRate',
    'SchoolRating',
    'AgeOfHome',
    'DistanceToCityCenter',
    'EmploymentRate',
    'RenovationQualityRate',
    'LocalAmenities',
    'TransportAccess',
    'PreviousSalePrice']].copy()

#Drop rows with missing values
relevant_df=relevant_df.dropna()

#Identify outliers
col_mean = relevant_df.mean()
col_std = relevant_df.std()

zscore = (relevant_df-col_mean)/col_std

#Create dataframe without outliers
clean_df = relevant_df[(zscore.abs()< 3).all(axis=1)]

#Separate Target from Features
target = relevant_df['Price']
features = relevant_df.drop(columns=['Price'])

#Standardize Features
scaler = StandardScaler()
scaled_array = scaler.fit_transform(features)

scaled_df = pd.DataFrame(scaled_array, columns=features.columns, index=features.index)

#Begin PCA
from sklearn.decomposition import PCA

#Fit the PCA
pca = PCA()
pca.fit(scaled_df)

#Create Loading Matrix
loading_matrix = pd.DataFrame(
    pca.components_.T,
    index = features.columns,
    columns = [f'PC{i+1}'for i in range(len(features.columns))]
)

#Get eigenvalues and variance explained
eigenvalues = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
n_components = len(eigenvalues)
components = range(1, n_components + 1)

#Define Kaiser Rule
kaiser_components = np.sum(eigenvalues>1)
print(f"Kaiser Rule: Retain {kaiser_components} components (eigenvalue > 1)")

#Define Elbow Rule
second_derivative = np.diff(explained_variance_ratio, n=2)
elbow_point = np.argmin(second_derivative) + 2
print(f"Elbow Rule: Retain {elbow_point} components")

#Build Scree Plot
fig, ax1 = plt.subplots(figsize=(10,6))

#Bar Chart - Individual Variance
ax1.bar(components, explained_variance_ratio * 100,
        color='steelblue', alpha=0.7, label='Individual Variance %')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Variance Explained (%)', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_xticks(components)

#Line Plot - Cumulative Variance
ax2 = ax1.twinx()
ax2.plot(components, cumulative_variance * 100,
         color='darkorange', marker='o', linewidth=2, label='Cumulative Variance %')
ax2.set_ylabel('Cumulative Variance (%)', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')
ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')

#Kaiser Rule Line
ax1.axvline(x=kaiser_components + 0.5, color='red',
            linestyle='--', linewidth=1.5, label=f'Kaiser cutoff (PC{kaiser_components})')

#Elbow Rule Line
ax1.axvline(x=elbow_point + 0.5, color='green',
            linestyle='--', linewidth=1.5, label=f'Elbow cutoff (PC{elbow_point})')

#Combined Legend
lines1,labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 +labels2, loc='center right')

plt.title('Scree Plot with Kaiser and Elbow Rule Cutoffs')
plt.tight_layout()
plt.show()

#Proportion of variance for retained components
n_retained = 4

variance_table = pd.DataFrame({
    'Eigenvalue': eigenvalues[:n_retained].round(3),
    'Proportion of Variance (%)': (explained_variance_ratio[:n_retained] * 100).round(2),
    'Cumulative Variance (%)': (cumulative_variance[:n_retained] * 100).round(2)
}, index=[f'PC{i}' for i in range(1, n_retained + 1)])

print(variance_table)

#Project data onto retained components
pca_retained = PCA(n_components=n_retained)
x_pca = pca_retained.fit_transform(scaled_df)

x_pca_df = pd.DataFrame(x_pca,
                          columns=[f'PC{i+1}' for i in range(n_retained)],
                          index=scaled_df.index)

#Fit OLS model on full dataset
x_pca_const = sm.add_constant(x_pca_df)
ols_model = sm.OLS(target, x_pca_const).fit()

print(ols_model.summary())

#Find RMSE
residuals = ols_model.resid
rmse = np.sqrt(np.mean(residuals**2))
print(f"RMSE: {rmse:.2f}")
