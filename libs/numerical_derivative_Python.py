
"""
Compute numerical derivatives on a non-uniform (but strictly increasing) grid,
using quadratic Lagrangian interpolation to generate the difference matrix.
"""
import numpy as np
from scipy.sparse import csr_matrix

def differenceMatrix(x):
    """Generates the difference matrix for a non-uniform (but strictly 
    increasing) abscissa.  We use a two-sided finite difference for the 
    interior points, with one-sided differences for the boundary.  Interior 
    point coefficients are calculated using the derivatives of the 2nd-order 
    Lagrange polynomials for the approximation.
    ARGS:
        x: array.
            strictly increasing 1D grid.  Must contain at least 2 elements.
    """
    n = len(x)
    h = x[1:]-x[:n-1]   # length n-1 array of grid spacings

    # use one-sided differences at point 0 and n; coeffs are derivative of Lagrange
    # polynomials at interior points

    # a-coefficients below diagonal
    a0 = -(2*h[0]+h[1])/(h[0]*(h[0]+h[1]))
    ak = -h[1:]/(h[:n-2]*(h[:n-2]+h[1:]))
    an = h[-1]/(h[-2]*(h[-1]+h[-2]))

    # b-coefficients on diagonal
    b0 = (h[0]+h[1])/(h[0]*h[1]) 
    bk = (h[1:] - h[:n-2])/(h[:n-2]*h[1:])
    bn = -(h[-1]+h[-2])/(h[-1]*h[-2])

    # c-coefficients above diagonal
    c0 = -h[0]/(h[1]*(h[0]+h[1]))
    ck = h[:n-2]/(h[1:]*(h[:n-2]+h[1:]))
    cn = (2*h[-1]+h[-2])/(h[-1]*(h[-2]+h[-1]))

    # construct sparse difference matrix
    val  = np.hstack((a0,ak,an,b0,bk,bn,c0,ck,cn))
    row = np.tile(np.arange(n),3)
    dex = np.hstack((0,np.arange(n-2),n-3))
    col = np.hstack((dex,dex+1,dex+2))
    D = csr_matrix((val,(row,col)),shape=(n,n))
    return D

def strictlyIncreasing(x):
    """Checks that an input array is strictly increasing.
    ARGS:
        x: array-like.
            Numerical array.
    """
    isIncreasing = True
    for x1,x2 in zip(x,x[1:]):
        isIncreasing = isIncreasing and x1<x2
    return isIncreasing

def deriv(*args):
    """Calculates numerical derivative for strictly increasing, non-uniform grid,
    using quadrating Lagrangian interpolation to generate the difference matrix.
    ARGS:
        called with one or two 1-D arrays; 2 arrays must be of equal length.
        When called with one array, deriv(y), computes the 1st derivative of 
        that array using a uniform, integer-spaced grid, similar to numpy.gradient.
        When called with two arrays, assumes the form deriv(x,y), calculates the 
        derivative of y on the non-uniform grid x.
    """
    if len(args) == 1:
        y = args[0]
        x = np.arange(0,len(y),dtype=np.float64)
    else:
        x = args[0]
        y = args[1]
        if len(x) != len(y):
            raise ValueError("Input arrays must be of equal size.")
        if len(x) < 2:
            raise ValueError("Input array(s) must contain at least 2 elements")
        #if not strictlyIncreasing(x):
        #    raise ValueError("Input grid must be strictly increasing")

    # condition inputs
    try:
        x = np.array(x,dtype=np.float64)
        y = np.array(y,dtype=np.float64)
    except:
        raise ValueError("Inputs could not be conditioned to float arrays.")
    if len(x.shape) > 1 or len(y.shape) > 1:
        raise ValueError("Inputs must be 1-D arrays")

    D = differenceMatrix(x)
    return D*y
