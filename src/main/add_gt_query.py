import pandas as pd

# Load the two CSV files into DataFrames
df1 = pd.read_csv('/home/ntlpt19/Desktop/TF_release/TradeGPT/extraction_queries/ext_quries_sql_v2.csv')  # This contains 'updated_query' and 'GT'
df2 = pd.read_csv('/home/ntlpt19/Desktop/TF_release/TradeGPT/output_report.csv')  # This contains 'user_query'

# Merge df2 with df1 on the queries
df2_with_gt = pd.merge(df2, df1[['updated_query', 'ground_truth']], 
                       left_on='user_query', 
                       right_on='updated_query', 
                       how='left')

# Rename 'GT' column to 'ground_truth'
df2_with_gt.rename(columns={'ground_truth': 'ground_truth'}, inplace=True)

# Drop the 'updated_query' column from the merged DataFrame
df2_with_gt.drop(columns=['updated_query'], inplace=True)

# Save the result back to a new CSV file
df2_with_gt.to_csv('csv_file2_with_ground_truth.csv', index=False)

print("New CSV file with ground truth column saved as 'csv_file2_with_ground_truth.csv'")
