from typing import Callable

import numpy as np
import scipy
import pandas as pd
import sklearn

import data_generation



from causallib.datasets import load_nhefs
from causallib.estimation import IPW
from causallib.evaluation import evaluate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def gmm(g : Callable, Omega, k: int):
    """
    g: Moment restriction function which maps an input set of parameters to a k-vector
    Omega: User-defined PSD matrix needed to define minimization objective g^T Omega g
    n: Number of dimensions of parameter space. Needed to figure out input to minimize() call.
    """
    def objective(theta):
        v = g(theta)
        return v @ Omega @ v
    result = scipy.optimize.minimize(objective, np.zeros(k), method = "L-BFGS-B")
    # if result.success:
    #     print("GMM optimization successful")
    # else:
    #     print("GMM optimization failed.")
    #     print(result.message)
    return result.x


def binary_proximal_GMM(V: np.ndarray, W: np.ndarray, X, Y, Z, b: Callable, q: Callable, k: int):
    """
    V: The conditioning variables. Shape (N, n_V)
    W: The negative control outcomes. Shape (N, n_U)
    X: The binary primary treatment/exposure. Shape (N, n_X)
    Y: The primary outcome. Shape (N,)
    Z: The negative control exposures. Shape (N, N_Z)
    b: Bridge function which maps V, W, X, gamma to a vector (trying to predict Y)
    q: User specified function that helps define the moment restrictions. A function of V, X, Z mapping to a vector.
       Generally, the dimension of the output of q has to be as large as that of b.
    k: Dimension space of parameter to optimize.
    """
    # Reshape given inputs to be 2-dimensional.
    N = Y.shape[0]


    def g(theta):
        gamma = theta[:k-1]
        delta = theta[k-1]

        # (n,1) * (n, n_q) -> (n, n_q)
        moment1 = (Y - b(V, W, X, gamma)).reshape((N,1)) * q(V, X, Z)
        # (n_q)
        moment1 = np.mean(moment1, axis=0)

        # (1,) - (n, 1) -> (n, 1)
        # (1,)
        moment2 = delta - (b(V, W, np.ones(N), gamma) - b(V, W, np.zeros(N), gamma))
        moment2 = np.mean(moment2, axis=0)

        moment = np.r_[moment1, moment2]

        return moment
        
    Omega = np.identity(k)

    return gmm(g, Omega, k)[k-1] #Last entry should be Delta, the ATE.


def ipw(V, W, X, Y, Z):
    """
    Assume that V, W, X, Y, Z are all of shape (N,)
    """
    N = Y.shape[0]

    data = pd.DataFrame({
        'V': V,
        'Z': Z,
        'W': W,
    })

    le = LabelEncoder()
    treatment = pd.Series(le.fit_transform(X))
    outcome = pd.Series(Y)


    learner = LogisticRegression(solver="liblinear")
    ipw = IPW(learner)
    ipw.fit(data, treatment)
    outcomes = ipw.estimate_population_outcome(data, treatment, outcome)
    effect = ipw.estimate_effect(outcomes[1], outcomes[0], effect_types=["diff"])

    eval_results = evaluate(ipw, data, treatment, outcome)
    # eval_results.plot_all()

    # # Use V as features to fit a logistic model for X
    # p_model = sklearn.linear_model.LogisticRegression()
    # features = np.c_[V]
    # train_size = features.shape[0] // 2
    # test_size = features.shape[0] - train_size

    # train_features = features[:train_size]
    # test_features = features[train_size:]

    # p_model.fit(train_features, X[:train_size])


    # probs = p_model.predict_proba(test_features)

    # effect1 = 1/N * np.sum(Y[train_size:] * X[train_size:]/probs[:, 1])
    # effect0 = 1/N * np.sum(Y[train_size:] * (1 - X[train_size:])/probs[:,0])

    # print(effect1)
    # print(effect0)
    return effect['diff']





def ols(V, W, X, Y, Z):
    model = sklearn.linear_model.LinearRegression()
    features = np.c_[V, W, X, Z]
    X_index = 2

    model.fit(features, Y)

    return model.coef_[X_index]


def time_series_binary_proximal_GMM(V: np.ndarray, V_prev, W: np.ndarray, X, X_prev, Y, Z, b: Callable, q: Callable, k: int):
    """
    binary_proximal GMM modified to work for time_series data. Incredibly scuffed.
    """
    # Reshape given inputs to be 2-dimensional.
    N = Y.shape[0]


    def g(theta):
        gamma = theta[:k-1]
        delta = theta[k-1]

        # (n,1) * (n, n_q) -> (n, n_q)
        moment1 = (Y - b(V, V_prev, W, X, X_prev, gamma)).reshape((N,1)) * q(V, V_prev, X, X_prev, Z)
        # (n_q)
        moment1 = np.mean(moment1, axis=0)

        # (1,) - (n, 1) -> (n, 1)
        # (1,)
        moment2 = delta - (b(V, V_prev, W, np.ones(N), X_prev, gamma) - b(V, V_prev, W, np.zeros(N), X_prev, gamma))
        moment2 = np.mean(moment2, axis=0)

        moment = np.r_[moment1, moment2]

        return moment
        
    Omega = np.identity(k)

    return gmm(g, Omega, k)[k-1] #Last entry should be Delta, the ATE.


def lagged_ols(V, V_prev, W, X, X_prev, Y, Z):
    model = sklearn.linear_model.LinearRegression()
    N = Y.shape[0]
    features = np.c_[V, V_prev, W, X, X_prev, Z]
    X_index = 3

    model.fit(features, Y)

    return model.coef_[X_index]


if __name__ == "__main__":
    N = 1500

    # 1, V, W, X = gamma has size 4, delta has size 1, 4 + 1 = 5
    # I don't think OLS and IPW are working as intended. What are the conditioning variables? Why does IPW always predict such a high ATE?
    k = 7
    eta = 0
    xi = 0.6
    U, V, W, X, Y, Z = data_generation.get_iid_data(eta, xi, N)
    # print(data)
    def b(V, W, X, gamma):
        return np.c_[np.ones(N), V, W, X, X*V, X*W] @ gamma
    
    def q(V, X, Z):
        return np.c_[np.ones(N), V, X, Z, X*V, X*Z]
    

    result = binary_proximal_GMM(V, W, X, Y, Z, b, q, k)
    print(f"GMM: {result}")


    result = ipw(V, W, X, Y, Z)
    print(f"IPW: {result}")


    result = ols(V, W, X, Y, Z)
    print(f"OLS: {result}")