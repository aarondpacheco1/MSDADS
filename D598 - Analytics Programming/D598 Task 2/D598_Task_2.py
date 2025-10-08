#Step 1: Import file into dataframe
import pandas as pd
file_path = r'C:\Users\imret\Downloads\D598 Data Set(1-150 V2).csv'
df = pd.read_csv(file_path)

#Step 2: Identify Duplicate Rows
df.drop_duplicates(inplace=True)

#Step 3: Group data by state and calculate descriptive statistics
cols_to_agg = [
    "Total Long-term Debt",
    "Total Equity",
    "Debt to Equity",
    "Total Liabilities",
    "Total Revenue",
    "Profit Margin"
]
state_stats = df.groupby("Business State")[cols_to_agg].agg(['mean','median','min','max'])
state_stats = state_stats.reset_index()
print(state_stats)

#Step 4: Filter the data frame for businesses with negative debt-to-equity ratios
negative_debt_equity = df[df["Debt to Equity"]<0]
print(negative_debt_equity)

# Step 5: Create new dataframe with Business ID and Debt to Income Ratio
dti_df = pd.DataFrame({
    "Business ID": df["Business ID"],
    "Debt to Income Ratio": (
        df["Total Long-term Debt"] / df["Total Revenue"]
    ).where(df["Total Revenue"] != 0, other=pd.NA)
})

#Step 6: Concatenate the Debt to Income dataframe and the orignal dataframe
df_final = df.merge(dti_df, on="Business ID", how="left")
print(df_final)
