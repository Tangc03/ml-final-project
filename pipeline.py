import numpy as np
from utils.LLL import LLL_reduction, gramschmidt
from utils.new_search_fixed import search_nearest_point

# Placeholder functions
def URAN(n):
    # Uniformly randomize: return a vector of n points from [0,1)
    return np.random.rand(n)

def GRAN(n, m):
    # Gaussian randomize: return a (n x m) matrix from standard normal distribution
    return np.random.randn(n, m)

def CLP(B, x):
    # Closest lattice point function: return the closest lattice point to x
    n = B.shape[0]
    G = np.linalg.cholesky(B @ B.T)  # Ensure G is lower triangular
    nearest_point = search_nearest_point(n, G, x)
    return nearest_point[1:]  # Return only the first n elements (ignore the extra element)

def RED(B):
    # Reduction: return a reduced generator matrix using LLL algorithm
    return LLL_reduction(B)

def ORTH(B):
    # Orthogonal transformation: return an orthogonalized matrix
    # Use Gram-Schmidt to orthogonalize the basis
    return np.array(gramschmidt(B))

def iterative_lattice_construction(n, T, Tr):
    # Step 1: Initialize B with orthogonal reduction
    B = ORTH(GRAN(n, n))
    
    for t in range(T):
        # Step 5: Update u
        u = t / (T - 1)
        
        # Step 6: Generate random vector z
        z = URAN(n)
        
        # Step 7: Find closest lattice point y
        y = z - CLP(B, z @ B)
        
        # Step 8: Compute e
        e = y @ B
        
        # Steps 9-14: Update B
        for i in range(n):
            for j in range(i):
                B[i, j] -= u * y[i] * e[j]
            B[i, i] -= u * (y[i] * e[i] - (i + 1) * np.linalg.norm(e)**2)
        
        # Steps 15-19: Periodically reduce and orthogonalize B
        if t % Tr == Tr - 1:
            B = RED(B)
            B = ORTH(B)
    
    return B

# Example usage
n = 5  # Dimension
T = 100  # Number of iterations
Tr = 10  # Reduction interval
B = iterative_lattice_construction(n, T, Tr)
print("Generator matrix B:\n", B)