##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    # compute eps
    eps = sqrt(6) / sqrt(m + n)
    # get uni[-eps, eps]
    A0 = random.rand(m, n) * 2 * eps - eps
    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0