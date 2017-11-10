import numpy as np
import matplotlib.pyplot as plt

def read_network(filename):
    """
    Reads a file containing a set of links of the form

    origin1,target1
    origin2,target2

    and returns corresponding dictionary.
    """
    links = {}
    with open(filename, 'r') as network:
        for link in network:
            origin, target = map(int, link.split(','))
            if origin - 1 in links.keys():
                links[origin-1].append(target - 1)
            else:
                links[origin-1] = [target - 1]
    return links

def assemble_B_matrix(links, N):
    """
    Assembles the matrix B used in the PageRank algorithm.
    """
    B = np.zeros((N, N))
    for origin in links:
        for target in links[origin]:
            B[origin, target] = 1 / len(links[origin])
    return B.T

def page_rank_iterate(B, alpha=0.85, maxiter=100, residual=True):
    """
    Given a matrix B and a dampening factor alpha, iterates :maxiter: number of
    times, and returns the resulting popularity vector p.  If :residual: =
    True, also returns the length of the residual vector for each iteration.
    """
    M, N = B.shape 
    p = np.random.random(size=(N, 1)) # random initial vector
    p = p / p.sum(axis=0)             # normalize p 

    r = np.zeros(maxiter)             # array for residuals
    
    zero_col_index = np.where(~B.any(axis=0))[0] # indices corresponding to zero-columns

    for iteration in range(maxiter):
        Cpk = alpha / N * np.sum(p[zero_col_index])    # correction terms
        Opk = (1 - alpha)/ N * sum(p) # correction terms
        p_new = alpha * B.dot(p) + Cpk + Opk
        if residual:
            r[iteration] = np.linalg.norm(p_new - p)
        p = p_new 

    if not residual:
        return p.flatten()
    else:
        return p.flatten(), r

def test_assemble_matrix():
    filename = 'test_network.dat'
    L = read_network(filename)
    computed_B = assemble_B_matrix(L, N=5)
    expected_B = (1/2) * np.array([
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 1, 1, 1, 0]
    ])
    
    np.testing.assert_almost_equal(computed_B, expected_B)

def test_pagerank():
    B = (1/2) * np.array([
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 1, 1, 1, 0]
    ])
    
    computed_p = page_rank_iterate(B, maxiter=100)
    expected_p = np.array([
        [ 0.12425168],
        [ 0.20275927],
        [ 0.16897966],
        [ 0.22176866],
        [ 0.28224073]
    ])
    
    np.testing.assert_almost_equal(computed_p, expected_p)
    np.testing.assert_almost_equal(sum(computed_p), 1)
    
if __name__ == "__main__":
    L = read_network('network.dat')
    B = assemble_B_matrix(L, 10000)
    p, r = page_rank_iterate(B, maxiter=100, residual=True)
    
    print('Total: ', sum(p)) 
    plt.plot(r)
    plt.xlabel('Iteration number')
    plt.ylabel('Residual error')
    plt.title('PageRank convergence')
    plt.savefig('convergence.pdf')
    
    for i in np.argsort(-p)[:10]:
        print("Page #{} \t Popularity {:.3g}%".format(i, 100*p[i]))
