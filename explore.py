import pandas as pd

df = pd.read_csv("E0.csv")



df['imp_home'] = 1 / df['PSCH']
df['imp_draw'] = 1 / df['PSCD']
df['imp_away'] = 1 / df['PSCA']
df['overround'] = df['imp_home'] + df['imp_draw'] + df['imp_away']
df['fair_home'] = df['imp_home'] / df['overround']
df['fair_draw'] = df['imp_draw'] / df['overround']
df['fair_away'] = df['imp_away'] / df['overround']

print(df[['HomeTeam', 'AwayTeam', 'FTR', 'fair_home', 'fair_draw', 'fair_away', 'overround']].head())

print(df['overround'].mean())
print(df['overround'].min())
print(df['overround'].max())
print(df['overround'].describe())

print(df[['PSCH', 'PSCD', 'PSCA']].isnull().sum())