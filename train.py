import pandas as pd

# Load each season
df_21 = pd.read_csv("E21-22.csv")
df_22 = pd.read_csv("E22-23.csv")
df_23 = pd.read_csv("E23-24.csv")

# Add a season label
df_21['Season'] = '2021-22'
df_22['Season'] = '2022-23'
df_23['Season'] = '2023-24'

# Combine 21/22 and 22/23 for TRAINING
train_df = pd.concat([df_21, df_22], ignore_index=True)
test_df = df_23.copy()

# Identify teams missing from training
train_teams = set(train_df['HomeTeam']) | set(train_df['AwayTeam'])
test_teams = set(test_df['HomeTeam']) | set(test_df['AwayTeam'])
missing_teams = test_teams - train_teams

# Filter test set to predictable matches only
mask_predictable = ~(test_df['HomeTeam'].isin(missing_teams) | test_df['AwayTeam'].isin(missing_teams))
test_df = test_df[mask_predictable].reset_index(drop=True)

print(f"Training matches: {len(train_df)}")
print(f"Test matches (predictable): {len(test_df)}")


import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

# Set up teams index from TRAINING data
teams = sorted(set(train_df['HomeTeam']) | set(train_df['AwayTeam']))
n_teams = len(teams)
team_idx = {team: i for i, team in enumerate(teams)}

def neg_log_likelihood(params, df, team_idx):
    n = len(team_idx)
    attack = params[0:n]
    defense = params[n:2*n]
    gamma = params[2*n]
    
    log_lik = 0.0
    for _, row in df.iterrows():
        i = team_idx[row['HomeTeam']]
        j = team_idx[row['AwayTeam']]
        lambda_home = attack[i] * defense[j] * gamma
        lambda_away = attack[j] * defense[i]
        log_lik += poisson.logpmf(row['FTHG'], lambda_home)
        log_lik += poisson.logpmf(row['FTAG'], lambda_away)
    return -log_lik

initial_params = np.ones(2 * n_teams + 1)
bounds = [(0.01, None)] * (2 * n_teams + 1)

print("Fitting model on training data (this may take ~30s)...")
result = minimize(
    neg_log_likelihood, initial_params,
    args=(train_df, team_idx),
    method='L-BFGS-B', bounds=bounds,
)

attack_fit = result.x[0:n_teams]
defense_fit = result.x[n_teams:2*n_teams]
gamma_fit = result.x[2*n_teams]

print(f"Converged: {result.success}")
print(f"Home advantage (gamma): {gamma_fit:.3f}")

results_df = pd.DataFrame({
    'team': teams,
    'attack': attack_fit,
    'defense': defense_fit,
}).sort_values('attack', ascending=False)

print(results_df.to_string(index=False))