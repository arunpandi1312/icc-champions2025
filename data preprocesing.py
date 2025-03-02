import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


file_path = "/content/Train Data_ICC_Intra College Datathon 2.0.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")


if 'Outcome' in df.columns:
    df.drop(columns=['Outcome'], inplace=True)


df.dropna(subset=['Outcome.1'], inplace=True)


if 'NRR' in df.columns:
    df['NRR'] = pd.to_numeric(df['NRR'], errors='coerce')


label_encoders = {}
categorical_cols = ['Team', "Group", "Team’s Top Scorer", "Top Wicket-Taker"]
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le


numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])


processed_file_path = "/content/Processed_Data_ICC.csv"
df.to_csv(processed_file_path, index=False)

print(f"Preprocessed data saved to {processed_file_path}")
