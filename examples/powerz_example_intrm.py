
"""
"""
from gbayesdesign import power, power_1d


def main():
    delta = 0.3
    sigma = 1.0
    n_range = list(range(10, 101, 10))
    for n in n_range:
        p = power(n, delta, sigma=sigma, prior_var=1.0, nsim=500, seed=42)
        print(f"n={n:3d} bayes_power={p:.3f}")

if __name__ == '__main__':
    main()
