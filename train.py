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

results_df = pd.DataFrame({
    'team': teams,
    'attack': attack_fit,
    'defense': defense_fit,
}).sort_values('attack', ascending=False)




# Pinnacle fair probabilities on test set 
test_df['imp_home'] = 1 / test_df['PSCH']
test_df['imp_draw'] = 1 / test_df['PSCD']
test_df['imp_away'] = 1 / test_df['PSCA']
test_df['overround'] = test_df['imp_home'] + test_df['imp_draw'] + test_df['imp_away']
test_df['market_home'] = test_df['imp_home'] / test_df['overround']
test_df['market_draw'] = test_df['imp_draw'] / test_df['overround']
test_df['market_away'] = test_df['imp_away'] / test_df['overround']


# Model predictions for every test match
def predict_match(home_team, away_team, attack, defense, gamma, team_idx, max_goals=10):
    i = team_idx[home_team]
    j = team_idx[away_team]
    lambda_home = attack[i] * defense[j] * gamma
    lambda_away = attack[j] * defense[i]
    home_probs = poisson.pmf(np.arange(max_goals), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals), lambda_away)
    score_matrix = np.outer(home_probs, away_probs)
    p_home = np.tril(score_matrix, -1).sum()
    p_draw = np.diag(score_matrix).sum()
    p_away = np.triu(score_matrix, 1).sum()
    # Normalize to handle the 10x10 truncation
    total = p_home + p_draw + p_away
    return p_home/total, p_draw/total, p_away/total

# Apply to every row in test_df
model_home, model_draw, model_away = [], [], []
for _, row in test_df.iterrows():
    p_h, p_d, p_a = predict_match(
        row['HomeTeam'], row['AwayTeam'],
        attack_fit, defense_fit, gamma_fit, team_idx
    )
    model_home.append(p_h)
    model_draw.append(p_d)
    model_away.append(p_a)

test_df['model_home'] = model_home
test_df['model_draw'] = model_draw
test_df['model_away'] = model_away

# Have a look at the comparison
cols = ['HomeTeam', 'AwayTeam', 'FTR',
        'model_home', 'market_home',
        'model_draw', 'market_draw',
        'model_away', 'market_away']

# Score model vs market with proper scoring rules

# Build the "actual outcome" column as 0/1 indicators
test_df['actual_home'] = (test_df['FTR'] == 'H').astype(int)
test_df['actual_draw'] = (test_df['FTR'] == 'D').astype(int)
test_df['actual_away'] = (test_df['FTR'] == 'A').astype(int)

# Log loss: -mean(log(probability assigned to the actual outcome))
def log_loss(probs_home, probs_draw, probs_away, actual_home, actual_draw, actual_away):
    # For each match, pick the probability the model assigned to whatever actually happened
    chosen = probs_home * actual_home + probs_draw * actual_draw + probs_away * actual_away
    # Clip to avoid log(0) blow-ups
    chosen = np.clip(chosen, 1e-15, 1)
    return -np.mean(np.log(chosen))

# Brier score: mean squared error across all three outcomes
def brier_score(probs_home, probs_draw, probs_away, actual_home, actual_draw, actual_away):
    return np.mean(
        (probs_home - actual_home)**2 +
        (probs_draw - actual_draw)**2 +
        (probs_away - actual_away)**2
    )

model_ll = log_loss(test_df['model_home'], test_df['model_draw'], test_df['model_away'],
                    test_df['actual_home'], test_df['actual_draw'], test_df['actual_away'])
market_ll = log_loss(test_df['market_home'], test_df['market_draw'], test_df['market_away'],
                     test_df['actual_home'], test_df['actual_draw'], test_df['actual_away'])

model_bs = brier_score(test_df['model_home'], test_df['model_draw'], test_df['model_away'],
                       test_df['actual_home'], test_df['actual_draw'], test_df['actual_away'])
market_bs = brier_score(test_df['market_home'], test_df['market_draw'], test_df['market_away'],
                        test_df['actual_home'], test_df['actual_draw'], test_df['actual_away'])

print(f"\n--- Results on {len(test_df)} test matches ---")
print(f"Log loss   |  Model: {model_ll:.4f}   Market: {market_ll:.4f}   Diff: {model_ll - market_ll:+.4f}")
print(f"Brier      |  Model: {model_bs:.4f}   Market: {market_bs:.4f}   Diff: {model_bs - market_bs:+.4f}")
print(f"(Lower = better. Positive Diff means market is sharper.)")


# Dixon-Coles

def tau(i, j, lh, la, rho):
    """Low-score correction factor."""
    if i == 0 and j == 0:
        return 1 - lh * la * rho
    elif i == 1 and j == 0:
        return 1 + la * rho
    elif i == 0 and j == 1:
        return 1 + lh * rho
    elif i == 1 and j == 1:
        return 1 - rho
    else:
        return 1.0

def neg_log_likelihood_dc(params, df, team_idx):
    n = len(team_idx)
    attack = params[0:n]
    defense = params[n:2*n]
    gamma = params[2*n]
    rho = params[2*n + 1]
    
    log_lik = 0.0
    for _, row in df.iterrows():
        i = team_idx[row['HomeTeam']]
        j = team_idx[row['AwayTeam']]
        lh = attack[i] * defense[j] * gamma
        la = attack[j] * defense[i]
        
        hg = int(row['FTHG'])
        ag = int(row['FTAG'])
        
        # Base Poisson log-probs
        ll = poisson.logpmf(hg, lh) + poisson.logpmf(ag, la)
        
        # Add Dixon-Coles correction in log-space
        t = tau(hg, ag, lh, la, rho)
        if t <= 0:
            # Optimizer wandered into bad territory; return huge penalty
            return 1e10
        ll += np.log(t)
        
        log_lik += ll
    return -log_lik

# Initial guess: same as before, plus rho = 0 (no DC correction yet)
initial_params_dc = np.concatenate([np.ones(2 * n_teams + 1), [0.0]])
bounds_dc = [(0.01, None)] * (2 * n_teams + 1) + [(-0.3, 0.3)]  # rho is bounded

print("\nFitting Dixon-Coles model (this will take a bit longer)...")
result_dc = minimize(
    neg_log_likelihood_dc, initial_params_dc,
    args=(train_df, team_idx),
    method='L-BFGS-B', bounds=bounds_dc,
)

attack_dc = result_dc.x[0:n_teams]
defense_dc = result_dc.x[n_teams:2*n_teams]
gamma_dc = result_dc.x[2*n_teams]
rho_dc = result_dc.x[2*n_teams + 1]

print(f"Converged: {result_dc.success}")
print(f"Negative log-likelihood: {result_dc.fun:.2f}")
print(f"Home advantage (gamma): {gamma_dc:.3f}")
print(f"DC parameter (rho):     {rho_dc:.4f}")


def predict_match_dc(home_team, away_team, attack, defense, gamma, rho, team_idx, max_goals=10):
    i = team_idx[home_team]
    j = team_idx[away_team]
    lh = attack[i] * defense[j] * gamma
    la = attack[j] * defense[i]
    
    home_probs = poisson.pmf(np.arange(max_goals), lh)
    away_probs = poisson.pmf(np.arange(max_goals), la)
    score_matrix = np.outer(home_probs, away_probs)
    
    # Apply DC correction to the four low-score cells
    score_matrix[0, 0] *= (1 - lh * la * rho)
    score_matrix[1, 0] *= (1 + la * rho)
    score_matrix[0, 1] *= (1 + lh * rho)
    score_matrix[1, 1] *= (1 - rho)
    
    p_home = np.tril(score_matrix, -1).sum()
    p_draw = np.diag(score_matrix).sum()
    p_away = np.triu(score_matrix, 1).sum()
    total = p_home + p_draw + p_away
    return p_home/total, p_draw/total, p_away/total

# Apply DC predictions to test set
dc_home, dc_draw, dc_away = [], [], []
for _, row in test_df.iterrows():
    p_h, p_d, p_a = predict_match_dc(
        row['HomeTeam'], row['AwayTeam'],
        attack_dc, defense_dc, gamma_dc, rho_dc, team_idx
    )
    dc_home.append(p_h)
    dc_draw.append(p_d)
    dc_away.append(p_a)

test_df['dc_home'] = dc_home
test_df['dc_draw'] = dc_draw
test_df['dc_away'] = dc_away

dc_ll = log_loss(test_df['dc_home'], test_df['dc_draw'], test_df['dc_away'],
                 test_df['actual_home'], test_df['actual_draw'], test_df['actual_away'])
dc_bs = brier_score(test_df['dc_home'], test_df['dc_draw'], test_df['dc_away'],
                    test_df['actual_home'], test_df['actual_draw'], test_df['actual_away'])

print(f"\n--- Dixon-Coles vs Poisson vs Market ---")
print(f"Log loss   |  Poisson: {model_ll:.4f}   DC: {dc_ll:.4f}   Market: {market_ll:.4f}")
print(f"Brier      |  Poisson: {model_bs:.4f}   DC: {dc_bs:.4f}   Market: {market_bs:.4f}")


# Time-weighted Dixon-Coles

# Convert Date column to datetime
train_df['Date_parsed'] = pd.to_datetime(train_df['Date'], dayfirst=True)
test_df['Date_parsed'] = pd.to_datetime(test_df['Date'], dayfirst=True)

# Reference date: first day of the test season (so all training matches are "in the past")
ref_date = test_df['Date_parsed'].min()
train_df['days_ago'] = (ref_date - train_df['Date_parsed']).dt.days

print(f"\nReference date (start of test season): {ref_date.date()}")
print(f"Oldest training match: {train_df['days_ago'].max()} days ago")
print(f"Newest training match: {train_df['days_ago'].min()} days ago")

# Time-weighted likelihood
XI = 0.0015  # decay rate -- start here

def neg_log_likelihood_dc_weighted(params, df, team_idx, xi):
    n = len(team_idx)
    attack = params[0:n]
    defense = params[n:2*n]
    gamma = params[2*n]
    rho = params[2*n + 1]
    
    log_lik = 0.0
    for _, row in df.iterrows():
        i = team_idx[row['HomeTeam']]
        j = team_idx[row['AwayTeam']]
        lh = attack[i] * defense[j] * gamma
        la = attack[j] * defense[i]
        
        hg = int(row['FTHG'])
        ag = int(row['FTAG'])
        
        ll = poisson.logpmf(hg, lh) + poisson.logpmf(ag, la)
        t = tau(hg, ag, lh, la, rho)
        if t <= 0:
            return 1e10
        ll += np.log(t)
        
        # Apply time weight
        weight = np.exp(-xi * row['days_ago'])
        log_lik += weight * ll
    return -log_lik

print(f"\nFitting time-weighted Dixon-Coles (xi={XI})...")
result_tw = minimize(
    neg_log_likelihood_dc_weighted, initial_params_dc,
    args=(train_df, team_idx, XI),
    method='L-BFGS-B', bounds=bounds_dc,
)

attack_tw = result_tw.x[0:n_teams]
defense_tw = result_tw.x[n_teams:2*n_teams]
gamma_tw = result_tw.x[2*n_teams]
rho_tw = result_tw.x[2*n_teams + 1]

print(f"Converged: {result_tw.success}")
print(f"Home advantage (gamma): {gamma_tw:.3f}")
print(f"DC parameter (rho): {rho_tw:.4f}")

# Predict test matches
tw_home, tw_draw, tw_away = [], [], []
for _, row in test_df.iterrows():
    p_h, p_d, p_a = predict_match_dc(
        row['HomeTeam'], row['AwayTeam'],
        attack_tw, defense_tw, gamma_tw, rho_tw, team_idx
    )
    tw_home.append(p_h)
    tw_draw.append(p_d)
    tw_away.append(p_a)

test_df['tw_home'] = tw_home
test_df['tw_draw'] = tw_draw
test_df['tw_away'] = tw_away

tw_ll = log_loss(test_df['tw_home'], test_df['tw_draw'], test_df['tw_away'],
                 test_df['actual_home'], test_df['actual_draw'], test_df['actual_away'])
tw_bs = brier_score(test_df['tw_home'], test_df['tw_draw'], test_df['tw_away'],
                    test_df['actual_home'], test_df['actual_draw'], test_df['actual_away'])

print(f"\n--- Time-Weighted DC vs DC vs Poisson vs Market ---")
print(f"Log loss   |  Poisson: {model_ll:.4f}   DC: {dc_ll:.4f}   TW-DC: {tw_ll:.4f}   Market: {market_ll:.4f}")
print(f"Brier      |  Poisson: {model_bs:.4f}   DC: {dc_bs:.4f}   TW-DC: {tw_bs:.4f}   Market: {market_bs:.4f}")