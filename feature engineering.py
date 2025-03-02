import pandas as pd


data_path = "/content/Processed_Data_ICC.csv"
df = pd.read_csv(data_path)



if all(col in df.columns for col in ['Matches Played', 'Matches Won']):
    df['Win Ratio'] = df['Matches Won'] / df['Matches Played']
    df['Loss Ratio'] = 1 - df['Win Ratio']


if all(col in df.columns for col in ['Runs Scored', 'Matches Played']):
    df['Avg Runs per Match'] = df['Runs Scored'] / df['Matches Played']


if all(col in df.columns for col in ['Wickets Taken', 'Matches Played']):
    df['Wickets per Match'] = df['Wickets Taken'] / df['Matches Played']


if 'Last 5 Matches Runs' in df.columns:
    df['Avg Last 5 Matches Runs'] = df['Last 5 Matches Runs'] / 5


if all(col in df.columns for col in ['Strike Rate', 'Batting Average']):
    df['Strike Rate × Average'] = df['Strike Rate'] * df['Batting Average']

if all(col in df.columns for col in ['NRR', 'Matches Won']):
    df['NRR × Wins'] = df['NRR'] * df['Matches Won']


feature_engineered_path = "/content/Feature_Engineered_Data_ICC.csv"
df.to_csv(feature_engineered_path, index=False)

print(f"Feature-engineered data saved to {feature_engineered_path}")
