import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    #Your code here
    A = np.random.rand(n, 1)
    return A

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    #Your code here
    A = np.random.rand(h, w)
    B = np.random.rand(h, w)
    s = np.sum(A) + np.sum(B)
    return A, B, s


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    #Your code here
    C = A + B
    s = np.sqrt(np.sum(np.power(C,2)))
    return s


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    #Your code here
    transposeWeights = np.transpose(weights)
    multResult = np.matmul(transposeWeights, inputs)
    res = np.tanh(multResult)
    return res

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    #Your code here
    if y >= x:
        return x * y
    else:
        return x / y

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    #Your code here
    vFunc = np.vectorize(scalar_function)
    return vFunc(x, y)

def main():
    A = np.array([[3], [5], [9]])
    B = np.array([[2], [1], [4]])

    print('randomization:')
    print(randomization(6))
    print('\noperations:')
    print(operations(2,1))
    print('\nnorm:')
    print(norm(A, B))
    print('\nneural_network:')
    print(neural_network(A[1:], B[1:]))
    print('\nscalar_function:')
    print(scalar_function(5,2))
    print('\nvector_function:')
    print(vector_function([4, 3, 1, 2], 2))


if __name__ == "__main__":
    main()