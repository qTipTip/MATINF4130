import matplotlib.pyplot as plt
import numpy as np
import pytest
from pandas import DataFrame, Series
from time import time

from NULL import gaussian_elimination, gaussian_elimination_pivots, housetriang_solve

@pytest.mark.test_system
def test_system():
    A = np.array([
        [1, 1, 2],
        [2, 2, 1],
        [1, 2, 3]
    ], dtype=np.float64)

    b = np.array([ 9, 9, 14 ], dtype=np.float64)

    x = np.array([ 1, 2, 3 ], dtype=np.float64)

    for solver in [gaussian_elimination, gaussian_elimination_pivots, housetriang_solve]:
        computed_x = solver(A, b)
        np.testing.assert_almost_equal(computed_x, x, err_msg="%s:  %s != %s" % (solver.__name__, computed_x, x))

def stability():
    b = np.array([3, 4], dtype=np.float64)
    A = np.array([
        [0, 2],
        [1, 1]
    ], dtype=np.float64)
    for solver in [gaussian_elimination, gaussian_elimination_pivots, housetriang_solve]:
        print(solver.__name__ + ":")
        for eps in [1.0e-12, 1.0e-14, 1.0e-16]:
            A[0, 0] = eps
            x = solver(A, b)
            print("relative error (%g): %g " % (eps, relative_error(A, x, b)))

def timer(solver, n_values, N = 5):
    print("Computing time scaling: %s" % solver.__name__) 
    times = {n : 0 for n in n_values}
    for n in n_values:
        print("\t n = %g" % n)
        A = np.random.randint(-1000, 1000, size=(n, n))
        x = np.random.randint(-1000, 1000, size=(n))
        b = np.dot(A, x)

        for i in range(N):
            start = time()
            solver(A, b)    
            elapsed_time = time() - start
            times[n] += elapsed_time
        
        times[n] /= N
    
    return times

def computational_cost():
    n_values = [50 * 2 ** m for m in range(6)]
    solvers = [s.__name__ for s in [gaussian_elimination, gaussian_elimination_pivots, housetriang_solve]]
    table = DataFrame(columns=solvers, index=n_values)
    for solver in [gaussian_elimination, gaussian_elimination_pivots, housetriang_solve]:
        times = timer(solver, n_values)
        table[solver.__name__] = Series(times)
    with open("table_file.tex", 'w') as out:
        out.write(table.to_latex())     
    return n_values, table

def plot_runtime(n_values, table):
    
    for col in table:
        plt.plot(n_values, table[col], label=col)
    plt.xlabel('$n$')
    plt.ylabel('$\\bar{t}[s]$')
    plt.legend()
    plt.savefig('runtime.pdf')

def relative_error(A, x, b):
    return np.linalg.norm(A.dot(x) - b) / np.linalg.norm(b)
     

if __name__ == "__main__":
    stability() 
    #n_values, table = computational_cost()
    #plot_runtime(n_values, table)
