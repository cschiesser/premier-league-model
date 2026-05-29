import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


df = pd.read_csv("E23-24.csv")

# Get all unique teams
teams = sorted(set(df['HomeTeam']) | set(df['AwayTeam']))
n_teams = len(teams)
team_idx = {team: i for i, team in enumerate(teams)}


def neg_log_likelihood(params, df, teams, team_idx):
    n = len(teams)
    
    # Unpack the flat parameter array into named pieces
    attack = params[0:n]              # 20 numbers
    defense = params[n:2*n]           # 20 numbers
    gamma = params[2*n]               # 1 number (home advantage)
    
    log_lik = 0.0
    
    # Loop through every match
    for _, row in df.iterrows():
        i = team_idx[row['HomeTeam']]
        j = team_idx[row['AwayTeam']]
        
        # Expected goals for this matchup
        lambda_home = attack[i] * defense[j] * gamma
        lambda_away = attack[j] * defense[i]
        
        # Probability of the actual score under Poisson, log-space
        log_lik += poisson.logpmf(row['FTHG'], lambda_home)
        log_lik += poisson.logpmf(row['FTAG'], lambda_away)
    
    return -log_lik   # negate, because we'll minimize

# Initial guess: every team identical, no home advantage
initial_params = np.ones(2 * n_teams + 1)  # 41 ones

# Bounds: every parameter must be positive (λ > 0 required for Poisson)
bounds = [(0.01, None)] * (2 * n_teams + 1)

# Run the optimizer
result = minimize(
    neg_log_likelihood,
    initial_params,
    args=(df, teams, team_idx),
    method='L-BFGS-B',
    bounds=bounds,
)

# print("Converged:", result.success)
# print("Negative log-likelihood:", result.fun)

# Unpack the fitted parameters
attack_fit = result.x[0:n_teams]
defense_fit = result.x[n_teams:2*n_teams]
gamma_fit = result.x[2*n_teams]

# print(f"\nHome advantage (gamma): {gamma_fit:.3f}\n")

# Make a tidy table
results_df = pd.DataFrame({
    'team': teams,
    'attack': attack_fit,
    'defense': defense_fit,
}).sort_values('attack', ascending=False)

# print(results_df.to_string(index=False))

def predict_match(home_team, away_team, attack, defense, gamma, max_goals=10):
    i = team_idx[home_team]
    j = team_idx[away_team]
    
    lambda_home = attack[i] * defense[j] * gamma
    lambda_away = attack[j] * defense[i]
    
    # Probability of each possible goal count for each side
    home_probs = poisson.pmf(np.arange(max_goals), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals), lambda_away)
    
    # Outer product → 10x10 matrix of P(home=k, away=l)
    score_matrix = np.outer(home_probs, away_probs)
    
    # Aggregate into match outcomes
    p_home_win = np.tril(score_matrix, -1).sum()  # below diagonal
    p_draw     = np.diag(score_matrix).sum()       # diagonal
    p_away_win = np.triu(score_matrix, 1).sum()   # above diagonal
    
    return {
        'lambda_home': lambda_home,
        'lambda_away': lambda_away,
        'p_home_win': p_home_win,
        'p_draw': p_draw,
        'p_away_win': p_away_win,
    }
