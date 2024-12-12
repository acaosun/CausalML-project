import numpy as np
import matplotlib.pyplot as plt

from inference_methods import *
from data_generation import *


def iid_exp():
    """
    6.1: i.i.d Experiments
    """
    Ns = [500, 1500]
    etas = [0.0, 0.3, 0.5]
    xis = [0.2, 0.4, 0.6]

    # xis = [0.6, 0.6]
    # etas = [0, 0]
    # Ns = [1500]

    num_sims = 1000

    # Represents the dimension of the parameter space for GMM.
    k = 7

    fig, axs = plt.subplots(len(xis), len(etas))

    # I don't think OLS and IPW are working as intended. What are the conditioning variables? Why does IPW always predict such a high ATE?
    for xi_i in range(len(xis)):
        xi = xis[xi_i]

        for eta_i in range(len(etas)):
            eta = etas[eta_i]

            subplot = axs[xi_i, eta_i]
            if eta_i == 0:
                subplot.set(ylabel = f"xi = {xi}")

            if xi_i == len(xis) - 1:
                subplot.set(xlabel = f"eta = {eta}")

            # Aggregate results over N.
            gmm_results_ = []
            ipw_results_ = []
            ols_results_ = []
    
            
            for N_i in range(len(Ns)):
                N = Ns[N_i]
                print(f"eta: {eta}, xi: {xi}, N: {N}")
                gmm_results = []
                ipw_results = []
                ols_results = []
                for i in range(num_sims):
                    if i % 100 == 0:
                        print(f"Iteration {i}")
                    U, V, W, X, Y, Z = data_generation.get_iid_data(eta, xi, N)
                    # print(data)
                    b_matrix = np.c_[np.ones(N), V, W, X, X*V, X*W]
                    b1_matrix = np.c_[np.ones(N), V, W, np.ones(N), V, W]
                    b0_matrix = np.c_[np.ones(N), V, W, np.zeros(N), np.zeros(N), np.zeros(N)]
                    def b(V, W, X, gamma):
                        if np.all(X == np.ones(N)):
                            return b1_matrix @ gamma
                        elif np.all(X == np.zeros(N)): 
                            return b0_matrix @ gamma
                        else:
                            return b_matrix @ gamma
                    
                    q_matrix = np.c_[np.ones(N), V, X, Z, X*V, X*Z]
                    def q(V, X, Z):
                        return q_matrix
                    

                    result = binary_proximal_GMM(V, W, X, Y, Z, b, q, k)
                    gmm_results.append(result)

                    result = ipw(V, W, X, Y, Z)
                    ipw_results.append(result)

                    result = ols(V, W, X, Y, Z)
                    ols_results.append(result)
            
                gmm_results_.append(gmm_results)
                ipw_results_.append(ipw_results)
                ols_results_.append(ols_results)


            subplot.boxplot(gmm_results_, showfliers=False, positions = range(1, len(Ns)+1), widths = 0.5)
            subplot.boxplot(ipw_results_, showfliers=False, positions = range(len(Ns)+2, 2*len(Ns)+2), widths = 0.5)
            subplot.boxplot(ols_results_, showfliers=False, positions = range(2*len(Ns)+3, 3*len(Ns)+3), widths = 0.5)
            subplot.axhline(y=0.5, linestyle='--')

            subplot.set_xticks([(1 + len(Ns)) / 2, (len(Ns)+2 + 2*len(Ns)+1) / 2, (2*len(Ns)+3 + 3*len(Ns)+2) / 2], labels = ["NC", "IPW", "OLS"])
    plt.show()


def time_series_exp():
    """
    6.2: Time Series Experiments:
    """
    Ns = [500, 1500]
    etas = [0.0, 0.3, 0.5]
    xis = [0.7, 0.8, 0.9]

    # xis = [0.6, 0.9]
    # etas = [0, 0.5]
    # Ns = [1500]

    num_sims = 1000

    # Represents the dimension of the parameter space for GMM.
    k = 7

    fig, axs = plt.subplots(len(xis), len(etas))

    # I don't think OLS and IPW are working as intended. What are the conditioning variables? Why does IPW always predict such a high ATE?
    for xi_i in range(len(xis)):
        xi = xis[xi_i]

        for eta_i in range(len(etas)):
            eta = etas[eta_i]

            subplot = axs[xi_i, eta_i]
            if eta_i == 0:
                subplot.set(ylabel = f"xi = {xi}")

            if xi_i == len(xis) - 1:
                subplot.set(xlabel = f"eta = {eta}")

            # Aggregate results over N.
            gmm_results_ = []
            lag_results_ = []
            ols_results_ = []
    
            
            for N_i in range(len(Ns)):
                N = Ns[N_i]
                print(f"xi: {xi}, eta: {eta}, N: {N}")
                gmm_results = []
                lag_results = []
                ols_results = []
                for i in range(num_sims):
                    if i % 100 == 0:
                        print(f"Iteration {i}")

                    U, V, V_prev, W, X, X_prev, Y, Z = data_generation.get_time_series_data(eta, xi, N)
                    b_matrix = np.c_[np.ones(N), V, V_prev, W, X, X_prev]
                    b1_matrix = np.c_[np.ones(N), V, V_prev, W, np.ones(N), X_prev]
                    b0_matrix = np.c_[np.ones(N), V, V_prev, W, np.zeros(N), X_prev]

                    def b(V, V_prev, W, X, X_prev, gamma):
                        if np.all(X == np.ones(N)):
                            return b1_matrix @ gamma
                        elif np.all(X == np.zeros(N)): 
                            return b0_matrix @ gamma
                        else:
                            return b_matrix @ gamma
                    
                    q_matrix = np.c_[np.ones(N), V, V_prev, X, X_prev, Z]
                    def q(V, V_prev, X, X_prev, Z):
                        return q_matrix
                    

                    result = time_series_binary_proximal_GMM(V, V_prev, W, X, X_prev, Y, Z, b, q, k)
                    gmm_results.append(result)

                    # Lagged_ols needs access to the previous X as well.
                    result = lagged_ols(V, V_prev, W, X, X_prev, Y, Z)
                    lag_results.append(result)

                    result = ols(V, W, X, Y, Z)
                    ols_results.append(result)
            
                gmm_results_.append(gmm_results)
                lag_results_.append(lag_results)
                ols_results_.append(ols_results)


            subplot.boxplot(gmm_results_, showfliers=False, positions = range(1, len(Ns)+1), widths = 0.5)
            subplot.boxplot(lag_results_, showfliers=False, positions = range(len(Ns)+2, 2*len(Ns)+2), widths = 0.5)
            subplot.boxplot(ols_results_, showfliers=False, positions = range(2*len(Ns)+3, 3*len(Ns)+3), widths = 0.5)
            subplot.axhline(y=0.7, linestyle='--')

            subplot.set_xticks([(1 + len(Ns)) / 2, (len(Ns)+2 + 2*len(Ns)+1) / 2, (2*len(Ns)+3 + 3*len(Ns)+2) / 2], labels = ["NC", "LAG_OLS", "OLS"])
            subplot.set_ylim(0.3, 1.1)
    plt.show()



if __name__ == "__main__":
    iid_exp()


    # time_series_exp()

