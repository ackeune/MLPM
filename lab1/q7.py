import math
import numpy


# Signal generators
def sawtooth(x, period=0.2, amp=1.0, phase=0.):
    return (((x / period - phase - 0.5) % 1) - 0.5) * 2 * amp

def sine_wave(x, period=0.2, amp=1.0, phase=0.):
    return numpy.sin((x / period - phase) * 2 * numpy.pi) * amp

def square_wave(x, period=0.2, amp=1.0, phase=0.):
    return ((numpy.floor(2 * x / period - 2 * phase - 1) % 2 == 0).astype(float) - 0.5) * 2 * amp

def triangle_wave(x, period=0.2, amp=1.0, phase=0.):
    return (sawtooth(x, period, 1., phase) * square_wave(x, period, 1., phase) + 0.5) * 2 * amp

def random_nonsingular_matrix(d=2):
    """
    Generates a random nonsingular (invertible) matrix if shape d*d
    """
    epsilon = 0.1
    A = numpy.random.rand(d, d)
    while abs(numpy.linalg.det(A)) < epsilon:
        A = numpy.random.rand(d, d)
    return A
    
def make_mixtures(S, A):
    return numpy.matrix(A) * numpy.matrix(S)

num_sources = 5
signal_length = 500
t = numpy.linspace(0, 1, signal_length)
S = numpy.c_[sawtooth(t), sine_wave(t, 0.3), square_wave(t, 0.4), triangle_wave(t, 0.25), numpy.random.randn(t.size)].T

X = make_mixtures(S, random_nonsingular_matrix(5))






def ICA(X, activation_function, learning_rate):
    A = random_nonsingular_matrix(d=1) #mixing matrix
    W = numpy.linalg.inv(A) #demixing matrix
    
    treshold = 0.01
    max_iterations = 10000
    iterations = 0
    difference = 9000
    while(difference > treshold) and (iterations < max_iterations):
        #put X through a linear mapping
        A = numpy.dot(W,X)

        #put a through a nonlinear map
        Z = map(activation_function,A) #todo repmat A/loop or sth
        
        #put a back through W
        X_prime = numpy.dot(W.T, A)
    
        #adjust the weights
        W_delta = learning_rate*(W + numpy.dot(Z, X_prime.T))
        
        difference = max(abs(abs(numpy.diag(numpy.dot(W_delta, W.T))) - 1)) #???

        W = W_delta
        iterations += 1
    return W

Xtest = X[0,:]s
W = ICA(Xtest, lambda a: -math.tanh(a), 0.1)
