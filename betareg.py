import numpy as np
from scipy.special import gamma, digamma, polygamma, loggamma
import scipy.stats as stats

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_mu(X, beta):
    return np.clip(sigmoid(X @ beta), 1e-12, 1 - 1e-12)

def log_likelihood(params, X, y, lambda_ridge=0.0):
    beta = params[:-1]  # regression coefficients
    phi = params[-1]    # precision parameter
    mu = get_mu(X, beta)
    
    ll = (len(y) * (loggamma(phi)) - 
          np.sum(loggamma(mu * phi)) - 
          np.sum(loggamma((1 - mu) * phi)) + 
          np.sum((mu * phi - 1) * np.log(y)) + 
          np.sum(((1 - mu) * phi - 1) * np.log(1 - y)))
    
    penalty = lambda_ridge * np.sum(beta**2)
    return ll - penalty

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
    
    return np.concatenate([score_beta, [score_phi]])

def hessian(params, X, y):
    beta = params[:-1]
    phi = params[-1]
    mu = get_mu(X, beta)
    n, p = X.shape
    
    # 2nd derivative w.r.t. phi
    d2l_dphi2 = (n * polygamma(1, phi) - 
                 np.sum((mu**2) * polygamma(1, mu * phi)) - 
                 np.sum(((1 - mu)**2) * polygamma(1, (1 - mu) * phi)))
    
    # 2nd derivative w.r.t. beta (matrix)
    d1 = phi * mu * (1 - mu) * (polygamma(1, (1 - mu) * phi) + polygamma(1, mu * phi))
    term = np.log(y / (1 - y)) + digamma((1 - mu) * phi) - digamma(mu * phi)
    d2 = (2 * mu - 1) * term
    d2l_dbeta_dbeta = -phi * (X.T @ ((mu * (1 - mu) * (d1 + d2))[:, None] * X))
    
    # Mixed derivative w.r.t. beta and phi
    score_beta = phi * (X.T @ (term * mu * (1 - mu)))
    d2l_dbeta_dphi = ((score_beta / phi) + 
                      phi * (X.T @ (mu * (1 - mu) * (polygamma(1, (1 - mu) * phi) * (1 - mu) - 
                                              polygamma(1, mu * phi) * mu))))
    
    nJ = np.zeros((p + 1, p + 1))
    nJ[:p, :p] = d2l_dbeta_dbeta
    nJ[:p, p] = d2l_dbeta_dphi
    nJ[p, :p] = d2l_dbeta_dphi
    nJ[p, p] = d2l_dphi2
    return nJ

def fisher_info(params, X, y):
    beta = params[:-1]
    phi = params[-1]
    mu = get_mu(X, beta)
    n, p = X.shape
    
    # 2nd derivative w.r.t. phi
    d2l_dphi2 = (n * polygamma(1, phi) - 
                 np.sum((mu**2) * polygamma(1, mu * phi)) - 
                 np.sum(((1 - mu)**2) * polygamma(1, (1 - mu) * phi)))
    
    # 2nd derivative w.r.t. beta (matrix)
    d1 = phi * mu * (1 - mu) * (polygamma(1, (1 - mu) * phi) + polygamma(1, mu * phi))
    d2l_dbeta_dbeta = -phi * (X.T @ ((mu * (1 - mu) * d1)[:, None] * X))
    
    # Mixed derivative w.r.t. beta and phi
    d2l_dbeta_dphi = phi * (X.T @ (mu * (1 - mu) * (polygamma(1, (1 - mu) * phi) * (1 - mu) - polygamma(1, mu * phi) * mu)))
    
    nI = np.zeros((p + 1, p + 1))
    nI[:p, :p] = d2l_dbeta_dbeta
    nI[:p, p] = d2l_dbeta_dphi
    nI[p, :p] = d2l_dbeta_dphi
    nI[p, p] = d2l_dphi2
    return -1 * nI

def fit_regression(X, y, lambda_ridge=0.0):
    y = np.clip(y, 1e-12, 1 - 1e-12)

    # Support null model
    if X is not None:
        n, p = X.shape

        # add intercept to X
        X = np.hstack([np.ones((n, 1)), X]) 
    else:
        n = len(y)
        X = np.ones((n, 1))

    n, p = X.shape
    
    # Initial parameter guess
    beta = np.zeros(p)
    phi = 1
    params = np.concatenate([beta, [phi]])
    j = 0
    stopBeta = False

    old_lik = -1e99
    curr_lik = log_likelihood(params, X, y)
    print(curr_lik)
    while True:
        j += 1
        # Check for convergence
        if curr_lik < old_lik + 1e-4:
            break
        old_lik = curr_lik

        if not stopBeta:
            # Calculate gradient for beta
            grad = score(params, X, y)
            grad[-1] = 0
            lr = 1e-2

            propParams = params + lr * grad
            curr_lik = log_likelihood(propParams, X, y)

            i = 0
            while curr_lik < old_lik + 1e-6:
                lr *= 0.8
                propParams = params + lr * grad
                curr_lik = log_likelihood(propParams, X, y)
                i += 1
                if i > 120:
                    stopBeta = True
                    break
                
            params = propParams

        if j % 30 == 0:
            stopBeta = False

        # Calculate gradient for phi
        grad = score(params, X, y)
        grad[:p] = 0
        lr = 1

        propParams = params + lr * grad
        curr_lik = log_likelihood(propParams, X, y, lambda_ridge)

        i = 0
        while (propParams[-1] < 1e-8) or (curr_lik < old_lik + 1e-6):
            lr *= 0.6
            propParams = params + lr * grad
            curr_lik = log_likelihood(propParams, X, y, lambda_ridge)
            i += 1
            if i > 500:
                break

        params = propParams

        if j % 100 == 0:
            print(params, curr_lik)

    return params, X, y, fisher_info(params, X, y)

def std_resids(params, X, y):
    n, p = X.shape
    mu = get_mu(X, params[:p])
    return (y - mu) / (np.sqrt((mu) *  (1 - mu) / (1 + params[p])))

def summary(params, names, X, y, I):
    n, p = X.shape
    print("Beta Regression Summary")
    try:
        wald = params[:p] / np.sqrt(np.diag(np.linalg.inv(I[:p,:p])))
        pval = 1 - stats.chi2(1).cdf(wald ** 2)

        names = ["Intercept"] + names

        w1 = max([len(x) + 3 for x in names] + [len("Coefficient") + 3])
        print(f"{'Coefficient':<{w1}}{'Value':<8}{'Wald':<8}{'P-value':<8}")
        for i, name in enumerate(names):
            print(f"{name:<{w1}}{params[i]:<8.2f}{wald[i]:<8.2f}{pval[i]:<8.2f}")
        print()
        print(f"Log-Likelihood:   {log_likelihood(params, X, y):.2f}")

    except np.linalg.LinAlgError:
        names = ["Intercept"] + names
        print("Information Matrix is Singular. No test results will be printed")
        w1 = max([len(x) + 3 for x in names] + [len("Coefficient") + 3])
        print(f"{'Coefficient':<{w1}}{'Value':<8}")
        for i, name in enumerate(names):
            print(f"{name:<{w1}}{params[i]:<8.2f}")
        print()
        print(f"Log-Likelihood:   {log_likelihood(params, X, y):.2f}")
    
def lr_test(params1, params2, X1, X2, y):
    k = np.abs(len(params2) - len(params1))
    stat = 2 * np.abs(log_likelihood(params2, X2, y) - log_likelihood(params1, X1, y))
    print(f"LR Test: {k} parameter difference:")
    print(f"X2 Stat: {stat}")
    print(f"P-Value: {1 - stats.chi2(k).cdf(stat)}")