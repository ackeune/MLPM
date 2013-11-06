import numpy

# X is a M x T data matrix (M is the dimensionality and T is the numer of examples)
def whiten(X):
    (M,T) = X.shape
    Covariance = numpy.cov(X)
    eigenvalues,eigenvectors = numpy.linalg.eig(Covariance) #this only works for square matrices
    Lambda = eigenvalues*numpy.identity(M) #puts the eigenvalues in a diagonal matrix
    #print Lambda
    print eigenvectors
    y = eigenvectors.T*x
    
#X = numpy.matrix('1 1 1; 2 23 45;5 5 12')
X = numpy.matrix('1; 2; 5')
#X = numpy.random.random((2,2))
X = numpy.matrix(X)
whiten(X)

