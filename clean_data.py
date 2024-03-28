import pandas as pd

raw_data_df = pd.read_csv('src_data/census.csv', skipinitialspace=True)

# for col in list(raw_data_df.columns.values):
# print( raw_data[col].describe() )
# clean_data_df [col] = raw_data_df

clean_data_df = raw_data_df.dropna()

clean_data_df.to_csv('./src_data/clean_census.csv', index=False)
