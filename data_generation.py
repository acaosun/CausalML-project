import numpy as np


def get_iid_data(eta : float, xi: float, N: int):
    """
    eta: The level of confounding
    xi: Level of association between counfounding variables and negative treatment outcome
    N: The amount of data to be generated
    """
    UV = np.random.multivariate_normal([0,0], [[1, 0.5], [0.5, 1]], size = N)
    U = UV[:,0]
    V = UV[:,1]

    epsilon_Z = np.random.standard_normal(size = N)
    epsilon_W = np.random.standard_normal(size = N)
    Z = 0.5 + 0.5*V + U + epsilon_Z
    W = 1 - V + xi*U + epsilon_W

    probs = -0.5 + Z + 0.5*V + eta*U
    probs = 1 / (1 + np.exp(-1 * probs))

    X = np.random.binomial(n = 1, p = probs)

    Y = 1 + 0.5*X + 2*V + U + 1.5*X*U + 2*epsilon_W

    return U, V, W, X, Y, Z


def get_time_series_data(eta : float, xi: float, N: int):
    """
    eta: The level of confounding
    xi: Autocorrelation coefficient between time-adjacent confounding states
    N: The amount of data to be generated
    Output: U, V, X, Y of lengths N+2, W,Z of length N.
    """
    # We will need to generate N + 2 time steps. Using indices 1->1000 (inclusive), we need access to Y_0 and X_1001
    epsilon_U = np.random.standard_normal(N+2)
    
    U = np.zeros(N+2)
    for i in range(N+2):
        if i == 0:
            U[i] = np.random.standard_normal()
        else:
            U[i] = xi*U[i-1] + epsilon_U[i]

    epsilon_V = np.random.standard_normal(N+2)
    V = 0.6*U + epsilon_V

    epsilon_X = np.random.standard_normal(N+2)
    X = 0.4 + 1.5*V + eta*U+ epsilon_X

    epsilon_Y = np.random.standard_normal(N+2)
    Y = 0.5 + 0.7*X + 1.5*V + 0.9*U + epsilon_Y

    # Determine W, Z using above variables
    W = Y[0:N]
    Z = X[2:N+2]



    return U[1:N+1], V[1:N+1], V[0:N], W, X[1:N+1], X[0:N], Y[1:N+1], Z

if __name__ == "__main__":
    print(get_iid_data(0.5, 0.5, 10))
    print(get_time_series_data(0.5, 0.5, 10))