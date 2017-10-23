import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def bestfit(X, k):
    m, n = X.shape
    U, S, V = la.svd(X)
    B = U[:, k:]
    dist = la.norm(X.T.dot(B))
    return B, dist

def vanleastsqr(X, k):
    m, n = X.shape
    Y = X[:k+1, :]
    Z = X[k+1:, :]
    A = (Y.dot(Y.T))

def test():

    X1 = np.array([
        np.linspace(-1, 1, 21),
        np.random.uniform(-1, 1, 21)
    ])
    k = 1
    B, dist =  bestfit(X1, k)
    print(dist)
    a = float(B[1, :] / B[0, :])
    plt.plot([-1, 1], [-a, a])
    plt.scatter(X1[0, :], X1[1, :]) 
    plt.show()
if __name__ == "__main__":
    test()
     
