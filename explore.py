import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("E23-24.csv")



df['imp_home'] = 1 / df['PSCH']
df['imp_draw'] = 1 / df['PSCD']
df['imp_away'] = 1 / df['PSCA']
df['overround'] = df['imp_home'] + df['imp_draw'] + df['imp_away']
df['fair_home'] = df['imp_home'] / df['overround']
df['fair_draw'] = df['imp_draw'] / df['overround']
df['fair_away'] = df['imp_away'] / df['overround']

df['home_won'] = (df['FTR'] == 'H').astype(int)
df['prob_bucket'] = pd.cut(df['fair_home'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])


calib = df.groupby('prob_bucket', observed=True)['home_won'].agg(['mean', 'count']).reset_index()
midpoints = [0.1, 0.3, 0.5, 0.7, 0.9]

plt.figure(figsize=(7, 7))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.scatter(midpoints, calib['mean'], s=calib['count']*3, alpha=0.7, label='Pinnacle')
plt.xlabel('Predicted probability of home win')
plt.ylabel('Actual home win rate')
plt.title('Calibration: Pinnacle closing odds, Premier League 2023/24')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('calibration.png', dpi=150, bbox_inches='tight')
plt.show()
