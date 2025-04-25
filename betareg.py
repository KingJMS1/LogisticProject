import numpy as np
from scipy.special import gamma, digamma, polygamma

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_mu(X, beta):
    return np.clip(X @ beta, 1e-12, 1 - 1e-12)

def log_likelihood(params, X, y):
    beta = params[:-1]  # regression coefficients
    phi = params[-1]    # precision parameter
    mu = get_mu(X, beta)
    
    ll = (len(y) * np.log(gamma(phi)) - 
          np.sum(np.log(gamma(mu * phi))) - 
          np.sum(np.log(gamma((1 - mu) * phi))) + 
          np.sum((mu * phi - 1) * np.log(y)) + 
          np.sum(((1 - mu) * phi - 1) * np.log(1 - y)))
    return ll

def score(params, X, y):
    beta = params[:-1]
    phi = params[-1]
    mu = get_mu(X, beta)
    
    score_phi = (len(y) * digamma(phi) - 
                 np.sum(mu * digamma(mu * phi)) - 
                 np.sum((1 - mu) * digamma((1 - mu) * phi)) + 
                 np.sum(mu * np.log(y)) + 
                 np.sum((1 - mu) * np.log(1 - y)))
    
    term = np.log(y / (1 - y)) + digamma((1 - mu) * phi) - digamma(mu * phi)
    score_beta = phi * (X.T @ (term * mu * (1 - mu)))
    
    return -np.concatenate([score_beta, [score_phi]])

def hessian(params, X, y):
    beta = params[:-1]
    phi = params[-1]
    mu = get_mu(X, beta)
    n, p = X.shape
    
    # 2nd derivative w.r.t. phi
    d2l_dphi2 = (n * polygamma(1, phi) - 
                 np.sum(mu**2 * polygamma(1, mu * phi)) - 
                 np.sum((1 - mu)**2 * polygamma(1, (1 - mu) * phi)))
    
    # 2nd derivative w.r.t. beta (matrix)
    d1 = phi * mu * (1 - mu) * (polygamma(1, (1 - mu) * phi) + polygamma(1, mu * phi))
    term = np.log(y / (1 - y)) + digamma((1 - mu) * phi) - digamma(mu * phi)
    d2 = (2 * mu - 1) * term
    d2l_dbeta_dbeta = -phi * (X.T @ ((mu * (1 - mu) * (d1 + d2))[:, None] * X))
    
    # Mixed derivative w.r.t. beta and phi
    score_beta = phi * (X.T @ (term * mu * (1 - mu)))
    d2l_dbeta_dphi = (score_beta / phi + 
                      phi * (X.T @ (mu * (1 - mu) * (polygamma(1, (1 - mu) * phi) * (1 - mu) - 
                                              polygamma(1, mu * phi) * mu))))
    
    J = np.zeros((p + 1, p + 1))
    J[:p, :p] = -d2l_dbeta_dbeta
    J[:p, p] = -d2l_dbeta_dphi
    J[p, :p] = -d2l_dbeta_dphi 
    J[p, p] = -d2l_dphi2          
    return J