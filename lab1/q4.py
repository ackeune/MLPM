%pylab inline

def whiten(X):
    X = numpy.matrix(X)

    #making the data have a zero mean
    X = X - numpy.mean(X)
    #calculating the covariance
    A = numpy.dot(X.T,X)
    #calculating the eigenvalues and eigenvectors
    d,V = numpy.linalg.eigh(A)

    epsilon = 0.1    
    D = numpy.diag(1./numpy.sqrt(d+epsilon))
        
    #whitening matrix
    W = numpy.dot(numpy.dot(V,D),V.T)
    
    #whitening the data
    X = numpy.dot(X,W)
    return X
    
    
Xwhitened = whiten(X)

Xwhitened = np.squeeze(np.asarray(Xwhitened))

C = numpy.cov(Xwhitened)
ax = imshow(C, cmap='gray', interpolation='nearest')
