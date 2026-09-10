import pandas as pd
df = pd.read_csv('/home/mathew/SRP/microlens/processed/item_df.csv')
print(df['has_image_embedding'].value_counts())