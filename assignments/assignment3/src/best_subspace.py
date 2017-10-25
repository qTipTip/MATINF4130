import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def vanleastsqr(X, k):
    Y = X[0:k, 0:]
    Z = X[k:, 0:]
    A = la.inv(Y.dot(Y.T)).dot(Y).dot(Z.T).T
    dist = la.norm(A.dot(Y) - Z)
    return A, dist

def bestfit(X, k):
    U = la.svd(X)[0]
    B = U[:, :k]
    W = U[:, k:]
    dist = la.norm(X.T.dot(W))
    return B, dist

def test():
    X1 = np.array([
        np.linspace(-1, 1, 21),
        np.random.uniform(-1, 1, 21)
    ])
    X2 = np.array([
        np.random.uniform(-1, 1, 21),
        np.random.uniform(-1, 1, 21)
    ])
    k = 1

    for i, X in enumerate([X1, X2]):
        B, distB = bestfit(X, k)
        A, distA = vanleastsqr(X, k)

        aB = B[1] / B[0]
        aA = A[0]

        start = np.min(X[0,:])
        stop  = np.max(X[0,:])
        plt.subplot(2, 1, i+1)
        plt.plot([start, stop], [aB*start, aB*stop], label='bestfit distance = %.3f' % distB)
        plt.plot([start, stop], [aA*start, aA*stop], label='vanilla distance = %.3f' % distA)
        plt.title('$\\mathbf{X}_%d$' % i)
        plt.legend()
        plt.scatter(X[0,:], X[1,:], alpha=0.8)

    plt.tight_layout()
    plt.savefig('best_subspace.pdf')

if __name__ == "__main__":
    test()
    
