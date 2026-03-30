import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd

################################
## INITIAL MODEL PARAMETERS

S0 = 1.0             # Initial asset price
y0 = 0.0             # Initial value of the y_t process
N = 252              # number of steps
gamma = 3.0          # mean reversion rate
eta = 0.6            # volatility of y_t
rho = -0.6           # correlation b/t two brownian motions
m = 0.1              # costant volatility parameter
M = 20000             # number of Monte Carlo trajectories
K = 1.163            # strike

################################
## TRAJECTORIES

def simulate_y_euler(y0, gamma, eta, N, dt):
    y = np.zeros(N+1)
    y[0] = y0
    for n in range(N):
        xi = np.random.normal()
        y[n+1] = y[n] - gamma * y[n] * dt + eta * np.sqrt(dt) * xi
    return y

def simulate_y_exact(y0, gamma, eta, N, dt):
    y = np.zeros(N+1)
    y[0] = y0
    for n in range(N):
        xi = np.random.normal()
        var = (1 - np.exp(-2 * gamma * dt)) / (2 * gamma)
        y[n+1] = y[n] * np.exp(-gamma * dt) + eta * np.sqrt(var) * xi
    return y

# Simulate a trajectory
def simulate_S(S0, y, m, rho, dt, return_increments=False):
    S = np.zeros(len(y))
    S[0] = S0
    dW_list = []
    dZ_list = []
    
    for n in range(len(y)-1):
        xi1 = np.random.normal()
        xi2 = np.random.normal()
        dZ = np.sqrt(dt) * xi1
        dW = np.sqrt(dt) * (rho * xi1 + np.sqrt(1 - rho**2) * xi2)
        sigma = np.sqrt(m) * np.exp(y[n] / 2)
        S[n+1] = S[n] * np.exp(-0.5 * sigma**2 * dt + sigma * dW)
        
        if return_increments:
            dW_list.append(dW)
            dZ_list.append(dZ)  

    if return_increments:
        return S, np.array(dW_list), np.array(dZ_list)
    else:
        return S

# Simulate M-trajectories
def simulate_paths(S0, y0, m, gamma, eta, rho, T, N, M, method='exact'):
    dt = T / N
    S_mat = np.zeros((M, N+1))
    y_mat = np.zeros((M, N+1))
    dW_all = np.zeros((M, N))
    dZ_all = np.zeros((M, N))
    
    for i in range(M):
        
        if method == 'euler':
            y = simulate_y_euler(y0, gamma, eta, N, dt)
            y_mat[i, :] = y
            
        else:
            y = simulate_y_exact(y0, gamma, eta, N, dt)
            y_mat[i, :] = y
        
        S, dW, dZ = simulate_S(S0, y, m, rho, dt, return_increments=True)
        S_mat[i, :] = S
        dW_all[i, :] = dW
        dZ_all[i, :] = dZ
    
    return S_mat, y_mat, dW_all, dZ_all

def monte_carlo_option_price(S_paths, K, kind='put'):
    S_T = S_paths[:, -1]
    
    if kind == 'call':
        payoffs = np.maximum(S_T - K, 0)
        
    else:
        payoffs = np.maximum(K - S_T, 0)
    
    price = np.mean(payoffs)
    std_dev = np.std(payoffs) #evaluate the population standard deviation 
    std_error = std_dev / np.sqrt(len(S_T))
        
    return price, std_error


################################
## CHECKS AND VALIDATION

#  MARTINGALITY CHECK 

def martingality_check(S_paths, S0, T, N, M):
    
    mean_S = S_paths.mean(axis=0)
    mc_err = 2*(np.std(S_paths, axis=0) / np.sqrt(M))              
    
    deviation = mean_S - S0
    time_grid = np.linspace(0, T, N+1)
    
    print(f"Max deviation from S0 over time: {deviation.max():.4e}, at t={(time_grid[deviation.argmax()]):.3f}")
    print(f"Std error at final time: {mc_err[-1]:.4e}") 
    
    mask_fail = np.abs(mean_S - S0) > mc_err
    
    plt.figure(figsize=(10, 5))
    plt.plot(time_grid, mean_S, label='E[S_t]', color='blue')
    plt.fill_between(time_grid, S0 - mc_err, S0 + mc_err, color='gray', alpha=0.3, label='± 2*stderr band')
    plt.axhline(S0, color='black', linestyle='--', label='S0')
    
    # failing test points
    plt.scatter(time_grid[mask_fail], mean_S[mask_fail], color='red', label='Fail', zorder=5)
    
    plt.title("Martingality Test on $S_t$")
    plt.xlabel("Time")
    plt.ylabel("$\mathbb{E}[S_t]$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#  CORRELATION CHECK

def correlation_check(dW_all, dZ_all, rho, dt, N, M):
    
    dW_flat = dW_all.flatten() # from 2 dimensional array to 1 dimensional array
    dZ_flat = dZ_all.flatten()
    products = dW_flat * dZ_flat
    emp_cov = np.mean(products)
    emp_corr = emp_cov/(np.std(dW_flat, ddof=1)*np.std(dZ_flat, ddof=1))
    theoretical_cov = rho * dt
    
    print("\n=== CORRELATION CHECK ===")
    print(f"Empirical Cov(dW, dZ):   {emp_cov:.6e}")
    print(f"Theoretical Cov:         {theoretical_cov:.6e}")
    print(f"Difference:              {abs(emp_cov - theoretical_cov):.2e}")
    print()
    print(f"Empirical Corr(dW, dZ):  {emp_corr:.4f}")
    print(f"Target Corr:             {rho}")
    
    # Confidance Interval for Covariance
    sample_var = np.var(products, ddof=1)
    sample_std = np.sqrt(sample_var)
    N_total = len(products)
    std_error_cov = sample_std / np.sqrt(N_total)
    
    CI_low = emp_cov - 1.96 * std_error_cov
    CI_high = emp_cov + 1.96 * std_error_cov
    
    print("\n--- Confidence Interval for Cov(dW, dZ) ---")
    print(f"95% CI: [{CI_low:.6e}, {CI_high:.6e}]")
    print(f"Contains theoretical cov ({theoretical_cov:.6e})? {'YES' if CI_low <= theoretical_cov <= CI_high else 'NO'}")
    
    # Expected order of the error
    N_total = M * N
    expected_order = 1 / np.sqrt(N_total)
    abs_error = abs(emp_cov - theoretical_cov)
    ratio = abs_error / expected_order
    
    print("\n--- Order-of-Magnitude Check ---")
    print(f"Absolute error:             {abs_error:.2e}")
    print(f"Expected O(1/sqrt(MN)):     {expected_order:.2e}")
    print(f"Error / expected:           {ratio:.2f}")
    print(f"Matches O(1/√MN)?           {'YES' if ratio < 5 else 'NO (too large)'}")

# CONVERGENCE ERROR

def montecarlo_error_convergence(S0, y0, K, T, N, M_values, m, gamma, eta, rho, kind='put', method='exact'):
    mc_prices = []
    std_errors = []

    for M in M_values:
        S_paths, _, _, _ = simulate_paths(S0, y0, m, gamma, eta, rho, T, N, M, method)
        price, std_error = monte_carlo_option_price(S_paths, K, kind)
        
        mc_prices.append(price)
        std_errors.append(std_error)

        print(f"M={M}, Price={price:.5f}, StdErr={std_error:.5f}")

    return mc_prices, std_errors


# Plot error vs M

def plot_convergence(prices, errors, M_values):
    
    plt.figure(figsize=(8, 5))
    plt.plot(M_values, errors, 'o-', label='Empirical Std Error')
    plt.plot(M_values, [errors[0] * np.sqrt(M_values[0] / M) for M in M_values],
             'k--', label=r'$\mathcal{O}(1/\sqrt{M})$ reference')
    
    plt.xlabel('Number of Monte Carlo Paths (M)')
    plt.ylabel('Standard Error')
    plt.title('Monte Carlo Error vs Sample Size')
    plt.xscale('log') # Set logarithmic scale for M (horizontal axis)
    plt.yscale('log') # Set logarithmic scale for error (vertical axis)
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()


# COMPARISON WITH BENCHMARK

def mcvsbenchmark(benchmark_df, kind = 'Put'):
    benchmark = benchmark_df[kind]
    mc = benchmark_df[f'Scott{kind}']
    err = benchmark_df[f'ErrMC_{kind}']
    
    plt.figure(figsize=(6, 6))
    plt.scatter(benchmark, mc, alpha=0.7, edgecolors='k', label=f'{kind} Prices' )
    
    max_val = max(benchmark.max(), mc.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='1:1 Line')
    
    plt.plot(benchmark, benchmark + err, 'g--', linewidth=0.5, label='Upper Bound (MC error)')
    plt.plot(benchmark, benchmark - err, 'g--', linewidth=0.5, label='Lower Bound (MC error)')
    
    plt.xlabel(f'Benchmark {kind} Price')
    plt.ylabel(f'Monte Carlo {kind} Price')
    plt.title(f'1:1 Price Comparison — {kind} with MC Error Band')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
################################
## IMPLIED VOLATILITIES

def bs_call_price(S0, K, T, sigma):
    
    if sigma <= 0:
        return 0.0
    
    d1 = (np.log(S0 / K) + ( 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * norm.cdf(d2)

def bs_put_price(S0, K, T, sigma):
    
    if sigma <= 0:
        return 0.0
    
    d1 = (np.log(S0 / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return  K * norm.cdf(-d2) - S0 * norm.cdf(-d1) 

def implied_volatility(price_MC, S0, K, T, kind='call'):
    
    def objective(sigma):
        
        if kind == 'call':
            bs_price = bs_call_price(S0, K, T, sigma)
        
        else:
            bs_price = bs_put_price(S0, K, T, sigma)
            
        return bs_price - price_MC
    
    try:
        return brentq(objective, 1e-6, 3.0) #brentq minimize the objective function
    
    except ValueError:
        return np.nan  # doesn't converge

def bisection_scheme(price_MC, S0, K, T, kind='call', tol = 1e-12, max_iter = 100):
    low = 0
    high = 3
    
    for i in range(max_iter):
        
        mid = (high + low)/2
        
        if kind == 'call':
            price = bs_call_price(S0, K, T, mid)
        
        else:
            price = bs_put_price(S0, K, T, mid)
            
        diff = price - price_MC
        
        if abs(diff) < tol:
            
            return mid
        
        elif diff > 0:
            high = mid
        
        else:
            low = mid
    
    return mid


def plot_iv_comparison(df, o = 'K', method = 'ScottIV_brent'):

    x = benchmark_df['impVol']
    y = benchmark_df[method]
    c = benchmark_df[o]
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Add a 5% margin
    x_margin = 0.05 * (x_max - x_min)
    y_margin = 0.05 * (y_max - y_min)
    
    eps = 0.01

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=c, cmap='viridis', edgecolors='k', alpha=0.8)
    
    plt.plot([x_min, x_max], [x_min, x_max], 'r--', label='1:1 Line')
    
    plt.plot([x_min, x_max], [x_min + eps, x_max + eps], 'g--', linewidth=1, label='+0.01 Band')
    plt.plot([x_min, x_max], [x_min - eps, x_max - eps], 'g--', linewidth=1, label='-0.01 Band')
    
    plt.xlabel("Benchmark Implied Volatility")
    plt.ylabel("Scott Model Implied Volatility")
    plt.title("Implied Volatility: Scott vs Benchmark (±0.01 Band)")
    if o == 'K':
        plt.colorbar(sc, label='Strike $K$')
    else:
        plt.colorbar(sc, label='Maturity $T$')
    plt.grid(True)
    plt.legend()
    
    # Automatic zoom
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout()
    plt.show()

def smile_section(T, df):
    
    df_section = df[df['T'] == T]
    df_section = df_section.sort_values(by='K')
    
    plt.figure(figsize=(8, 5))
    plt.plot(df_section['K'], df_section['ScottIV_bis'], marker='o', linestyle='-')
    plt.xlabel("Strike $K$")
    plt.ylabel("Implied Volatility")
    plt.title(f"Volatility Smile — Maturity $T={T}$")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
###############################
## MONTE CARLO VALIDATION WITH BENCHMARK 

benchmark_df = pd.read_csv('scott_benchmark.csv')
benchmark_df.columns = benchmark_df.columns.str.strip()

benchmark_df['ScottCall'] = np.nan
benchmark_df['ScottPut'] = np.nan
benchmark_df['ErrMC_Call'] = np.nan
benchmark_df['ErrMC_Put'] = np.nan
benchmark_df['ScottIV_brent'] = np.nan
benchmark_df['ScottIV_bis'] = np.nan

# One simulation for each maturity

# for T, group in grouped:
#     # Simula solo UNA volta per questo T
#     N_sim = int(N * T / 1.0)
#     S_paths, y_paths, dW_all, dZ_all = simulate_paths(S0, y0, m, gamma, eta, rho, T, N_sim, M, method='exact')
#     martingality_check(S_paths, S0, T, N_sim, M)
#     dt = T/N_sim
#     correlation_check(dW_all, dZ_all, rho, dt, N_sim, M)
    
#     # Itera sugli strike associati a questa T
#     for idx, row in group.iterrows():
        
#         K = row['K']
#         price_mc, err_mc = monte_carlo_option_price(S_paths, K, kind='call')
#         price_mc_put, err_mc_put = monte_carlo_option_price(S_paths, K, kind='put')
#         sigma_iv_brent = implied_volatility(price_mc_put, S0, K, T, kind='put')
#         sigma_iv_bis = bisection_scheme(price_mc_put, S0, K, T, kind='put')

#         # Inserisci nei campi giusti del DataFrame
#         benchmark_df.at[idx, 'ScottCall'] = price_mc
#         benchmark_df.at[idx, 'ErrMC_Call'] = err_mc
#         benchmark_df.at[idx, 'ScottPut'] = price_mc_put
#         benchmark_df.at[idx, 'ErrMC_Put'] = err_mc_put
#         benchmark_df.at[idx, 'ScottIV_brent'] = sigma_iv_brent
#         benchmark_df.at[idx, 'ScottIV_bis'] = sigma_iv_bis


T = benchmark_df["T"].max()
N_sim = int(N * T / 1.0)

# M_values = [500, 1000, 2000, 4000, 8000, 10000, 20000]
# prices, errors = montecarlo_error_convergence(S0, y0, K, T, N_sim, M_values, m, gamma, eta, rho, kind='put', method='exact')
# plot_convergence(prices, errors, M_values)

# One simulation for all maturity
S_paths, y_paths, dW_all, dZ_all = simulate_paths(S0, y0, m, gamma, eta, rho, T, N_sim, M, method='exact')
dt = T / N_sim
martingality_check(S_paths, S0, T, N_sim, M)
correlation_check(dW_all, dZ_all, rho, dt, N_sim, M)

# Group by maturity
grouped = benchmark_df.groupby('T')

for T, group in grouped:
    
    t_idx = int(T/dt)
    S_paths_T = S_paths[:,:t_idx + 1]
    
    # Iterate on associated strike for each T
    for idx, row in group.iterrows():
        
        K = row['K']
        price_mc, err_mc = monte_carlo_option_price(S_paths_T, K, kind='call')
        price_mc_put, err_mc_put = monte_carlo_option_price(S_paths_T, K, kind='put')
        sigma_iv = implied_volatility(price_mc_put, S0, K, T, kind='put')
        sigma_iv_bis = bisection_scheme(price_mc_put, S0, K, T, kind='put')

        # Put in exact index of the data frame
        benchmark_df.at[idx, 'ScottCall'] = price_mc
        benchmark_df.at[idx, 'ErrMC_Call'] = err_mc
        benchmark_df.at[idx, 'ScottPut'] = price_mc_put
        benchmark_df.at[idx, 'ErrMC_Put'] = err_mc_put
        benchmark_df.at[idx, 'ScottIV_brent'] = sigma_iv
        benchmark_df.at[idx, 'ScottIV_bis'] = sigma_iv_bis

# PLOT COMPARISON

mcvsbenchmark(benchmark_df)
mcvsbenchmark(benchmark_df, 'Call')

benchmark_df['Put_OK'] = (
    (benchmark_df['Put'] >= benchmark_df['ScottPut'] - 2*  benchmark_df['ErrMC_Put']) &
    (benchmark_df['Put'] <= benchmark_df['ScottPut'] + 2* benchmark_df['ErrMC_Put'])
)

benchmark_df['Call_OK'] = (
    (benchmark_df['Call'] >= benchmark_df['ScottCall'] - 2* benchmark_df['ErrMC_Call']) &
    (benchmark_df['Call'] <= benchmark_df['ScottCall'] + 2* benchmark_df['ErrMC_Call'])
)

n = len(benchmark_df)
n_call_ok = benchmark_df['Call_OK'].sum()
n_put_ok = benchmark_df['Put_OK'].sum()

print(f"Call within MC error: {n_call_ok}/{n} ({100 * n_call_ok / n:.1f}%)")
print(f"Put within MC error: {n_put_ok}/{n} ({100 * n_put_ok / n:.1f}%)")


# PUT-CALL PARITY TEST

benchmark_df['Call_from_Put'] = benchmark_df['ScottPut'] + (S0 - benchmark_df['K'])
benchmark_df['Put_from_Call'] = benchmark_df['ScottCall'] - (S0 - benchmark_df['K'])


benchmark_df['Tol_Call'] = 3 * benchmark_df['ErrMC_Put']
benchmark_df['Tol_Put']  = 3 * benchmark_df['ErrMC_Call']


benchmark_df['Call_OK_PCP'] = np.abs(benchmark_df['Call_from_Put'] - benchmark_df['ScottCall']) <= benchmark_df['Tol_Call']
benchmark_df['Put_OK_PCP']  = np.abs(benchmark_df['Put_from_Call'] - benchmark_df['ScottPut']) <= benchmark_df['Tol_Put']

violations_call = (~benchmark_df['Call_OK_PCP']).sum()
violations_put  = (~benchmark_df['Put_OK_PCP']).sum()
total = len(benchmark_df)

print(f"Call PCP violated in {violations_call}/{total} cases ({100 * violations_call / total:.1f}%)")
print(f"Put PCP violated in  {violations_put}/{total} cases ({100 * violations_put  / total:.1f}%)")


# IMPLIED VOLATILITY COMPARISON

plot_iv_comparison(benchmark_df, 'K')
plot_iv_comparison(benchmark_df, 'K', 'ScottIV_bis')
plot_iv_comparison(benchmark_df, 'T')
plot_iv_comparison(benchmark_df, 'T', 'ScottIV_bis')

# Comparison brent vs bisection
benchmark_df['IV_OK'] = ((benchmark_df['ScottIV_brent'] - benchmark_df['ScottIV_bis'] < -0.0001) | (benchmark_df['ScottIV_brent'] - benchmark_df['ScottIV_bis'] > 0.00001))


n = len(benchmark_df)
n_iv_ok = benchmark_df['IV_OK'].sum()
print(f"IV out of range: {n_iv_ok}/{n} ({100 * n_iv_ok / n:.1f}%)")

smile_section(T, benchmark_df)

##############################################
## PRICING WITH NEW PARAMETERS

S0 = 1.0             # Initial asset price
y0 = 0.0             # Initial value of the y_t process
N = 252              # number of steps
gamma = 2.0          # mean reversion rate
eta = 1.8            # volatility of y_t
rho = -0.4          # correlation b/t two brownian motions
m = 0.19              # costant volatility parameter
M = 20000             # number of Monte Carlo trajectories
K = 1.163            # strike

df_new = benchmark_df[['T', 'K']].copy()

df_new['Put'] = np.nan
df_new['Call'] = np.nan
df_new['Err'] = np.nan
df_new['ImpVol'] = np.nan

T = benchmark_df["T"].max()
N_sim = int(N * T / 1.0)
S_paths, y_paths, dW_all, dZ_all = simulate_paths(S0, y0, m, gamma, eta, rho, T, N_sim, M, method='exact')
dt = T / N_sim
martingality_check(S_paths, S0, T, N_sim, M)
correlation_check(dW_all, dZ_all, rho, dt, N_sim, M)

# Group by maturity
grouped = benchmark_df.groupby('T')

for T, group in grouped:
    
    t_idx = int(T/dt)
    S_paths_T = S_paths[:,:t_idx + 1]
    
    # Iterate on associated strike for each T
    for idx, row in group.iterrows():
        
        K = row['K']
        price_mc_put, err_mc_put = monte_carlo_option_price(S_paths_T, K, kind='put')
        price_mc_call = price_mc_put + (S0-K)
        sigma_iv = bisection_scheme(price_mc_put, S0, K, T, kind='put')

        # Put in exact index of the data frame
        df_new.at[idx, 'Put'] = price_mc_put
        df_new.at[idx, 'Call'] = price_mc_call
        df_new.at[idx, 'Err'] = err_mc_put
        df_new.at[idx, 'ImpVol'] = sigma_iv

df_new[['T', 'K', 'Put', 'Call', 'Err', 'ImpVol']].to_csv("results.csv", index=False)








