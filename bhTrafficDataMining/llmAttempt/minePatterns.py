import os
import polars as pl
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.preprocessing import KBinsDiscretizer

# Path to the output directory containing CSV files
output_directory = "bhTrafficDataMining/llmAttempt/dataProcessed"

# List to collect dataframes from CSV files
dataframes = []

# Iterate through each CSV file in the output directory
for csv_file in os.listdir(output_directory):
    if csv_file.endswith(".csv"):
        csv_path = os.path.join(output_directory, csv_file)
        # Read CSV file with polars
        df = pl.read_csv(csv_path)
        dataframes.append(df)

# Concatenate all dataframes into a single dataframe
full_df = pl.concat(dataframes, how="vertical").to_pandas()

# Drop datetime column and any unnamed index columns
full_df = full_df.loc[:, ~full_df.columns.str.contains("^Unnamed")]
full_df = full_df.drop(columns=["DATA HORA"], errors="ignore")

# List of numeric columns to discretize
numeric_columns = ["VELOCIDADE AFERIDA", "VELOCIDADE DA VIA", "TAMANHO", "FAIXA"]

# Check if numeric columns exist before discretizing
existing_numeric_columns = [col for col in numeric_columns if col in full_df.columns]

# Discretize numeric columns using KBinsDiscretizer if they exist
if existing_numeric_columns:
    est = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
    discretized_numeric = est.fit_transform(full_df[existing_numeric_columns])

    # Convert discretized numeric data to a dataframe and concatenate it with the original dataframe
    numeric_df = pd.DataFrame(
        discretized_numeric, columns=[f"{col}_bin" for col in existing_numeric_columns]
    )
    numeric_df.index = full_df.index  # Ensure the indices match for concatenation
    full_df = pd.concat([full_df, numeric_df], axis=1)

    # Drop original numeric columns
    full_df = full_df.drop(columns=existing_numeric_columns, errors="ignore")

# Convert discretized numeric columns to strings to prepare for FP-Growth
discretized_columns = [f"{col}_bin" for col in existing_numeric_columns]
for col in discretized_columns:
    if col in full_df.columns:
        full_df[col] = full_df[col].astype(str)

# Convert other categorical columns to strings
categorical_columns = [
    "CLASSIFICAÇÃO",
    "ENDEREÇO",
    "SENTIDO",
    "ACIMA VELOCIDADE PERMITIDA",
]
existing_categorical_columns = [
    col for col in categorical_columns if col in full_df.columns
]
for col in existing_categorical_columns:
    full_df[col] = full_df[col].astype(str)

# Combine all categorical columns for one-hot encoding
all_categorical_columns = discretized_columns + existing_categorical_columns

# Sample a fraction of the dataset to reduce memory usage for FP-Growth
full_df_sampled = full_df.sample(frac=0.1, random_state=42)
print(full_df_sampled)

# One-hot encode the sampled dataframe for FP-Growth, using only categorical columns
one_hot_df = pd.get_dummies(full_df_sampled, columns=all_categorical_columns)

# Ensure all values are 0 or 1 for FP-Growth
one_hot_df = one_hot_df.astype(bool)

# Apply FP-Growth to find frequent patterns
frequent_itemsets = fpgrowth(one_hot_df, min_support=0.1, use_colnames=True, verbose=1)

# Calculate association rules from frequent itemsets
num_transactions = one_hot_df.shape[0]
rules = association_rules(
    frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=num_transactions
)

# Display the frequent itemsets and association rules
print(frequent_itemsets)
frequent_itemsets.to_csv("bhTrafficDataMining/llmAttempt/results/frequent_itemsets.csv")
print(rules)
rules.to_csv("bhTrafficDataMining/llmAttempt/results/association_rules.csv")

# Metrics to evaluate the quality of mined patterns
if not frequent_itemsets.empty:
    average_support = frequent_itemsets["support"].mean()
    print(f"Average support of frequent itemsets: {average_support}")

if not rules.empty:
    average_confidence = rules["confidence"].mean()
    print(f"Average confidence of association rules: {average_confidence}")

    average_lift = rules["lift"].mean()
    print(f"Average lift of association rules: {average_lift}")
