import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("output", exist_ok=True)

dataset = pd.read_csv(
    "sales_data_sample.csv",
    encoding="latin1")


print("\n INITIAL DATASET ")
print(dataset.head())
print("\nInitial Dataset Shape:", dataset.shape)
print("\nColumn Names:", dataset.columns.tolist())

print("\n INFORMATION ")
dataset.info()

print("\nStatistical Summary:")
print(dataset.describe())

print("\nMissing Values Before Cleaning:")
print(dataset.isnull().sum())

duplicates = dataset.duplicated().sum()

dataset.drop_duplicates(inplace=True)

print("\nDuplicates Removed:", duplicates)

numeric_columns = [
    "SALES",
    "QUANTITYORDERED"]

for col in numeric_columns:
    dataset[col] = pd.to_numeric(
        dataset[col],
        errors="coerce")
    
dataset["ORDERDATE"] = pd.to_datetime(
    dataset["ORDERDATE"],
    errors="coerce")

dataset.loc[
    dataset["SALES"] < 0,
    "SALES"
] = np.nan

dataset.loc[
    dataset["QUANTITYORDERED"] < 0,
    "QUANTITYORDERED"
] = np.nan

for col in numeric_columns:
    if dataset[col].notna().any():
        dataset[col] = dataset[col].fillna(
            dataset[col].median())
        
if "PRODUCTLINE" in dataset.columns:
    mode_values = dataset["PRODUCTLINE"].mode()

    if not mode_values.empty:
        dataset["PRODUCTLINE"] = dataset[
            "PRODUCTLINE"
        ].fillna(mode_values[0])

dataset.drop(
    columns=["Unnamed: 0"],
    errors="ignore",
    inplace=True)

print("\n CLEANED DATASET ")
print(dataset.head())

print("\nFinal Dataset Shape:", dataset.shape)

print("\nRemaining Missing Values:")
print(dataset.isnull().sum())

print("\nRemaining Duplicates:")
print(dataset.duplicated().sum())

dataset.to_csv(
    "output/cleaned_sales_data.csv",
    index=False)

print("\nCleaned Dataset Exported Successfully!")

print("\n BUSINESS ANALYSIS ")

total_revenue = dataset["SALES"].sum()
average_sales = dataset["SALES"].mean()
highest_sales = dataset["SALES"].max()
lowest_sales = dataset["SALES"].min()

total_quantity = dataset["QUANTITYORDERED"].sum()

print("Total Revenue:", round(total_revenue, 2))
print("Average Sales:", round(average_sales, 2))
print("Highest Sales:", highest_sales)
print("Lowest Sales:", lowest_sales)
print("Total Quantity Ordered:", total_quantity)

product_sales = dataset.groupby(
    "PRODUCTLINE"
)["SALES"].sum()

best_product = product_sales.idxmax()

print("Best Performing Product:", best_product)
print("Best Product Revenue:", product_sales.max())

# ========== ONE BUSINESS VISUALIZATION ==========
product_sales = dataset.groupby(
    "PRODUCTLINE"
)["SALES"].sum().sort_values(ascending=False)

plt.figure(figsize=(11, 6))

product_sales.plot(
    kind="bar",
    color="#800020",
    edgecolor="black")

plt.title("Total Sales by Product Line", fontsize=16)
plt.xlabel("Product Line")
plt.ylabel("Total Revenue")

plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "output/sales_by_product_line.png",
    dpi=300,
    bbox_inches="tight")
plt.show()

print(" DATA CLEANING COMPLETED")
print(" BUSINESS ANALYSIS COMPLETED")
print(" VISUALIZATION COMPLETED")

print("\nAll output files saved in the output folder!")
