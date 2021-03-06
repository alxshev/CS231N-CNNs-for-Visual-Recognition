from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
	"""
	Softmax loss function, naive implementation (with loops)

	Inputs have dimension D, there are C classes, and we operate on minibatches
	of N examples.

	Inputs:
	- W: A numpy array of shape (D, C) containing weights.
	- X: A numpy array of shape (N, D) containing a minibatch of data.
	- y: A numpy array of shape (N,) containing training labels; y[i] = c means
	  that X[i] has label c, where 0 <= c < C.
	- reg: (float) regularization strength

	Returns a tuple of:
	- loss as single float
	- gradient with respect to weights W; an array of same shape as W
	"""
	# Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using explicit loops.     #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	N, D = X.shape
	_, C = W.shape
	# First, we compute the function's output:
	# Computes a matrix of shape (N, C), where each row corresponds
	# to an item, with each column a class
	scores = X @ W 
	# We interpret this as the unnormalized log probabilities. 
	# First, let's shrink values to avoid overflow. 
	# We do this by subtracting the maximum of each item from every element
	F = scores - np.max(scores, axis=1, keepdims=True)

	# Now take the exponent, and normalize
	F = np.exp(F)
	F /= np.sum(F, axis=1, keepdims=True)

	N = y.shape[0]
	correctScores = F[range(N), y]
	loss = -np.log(correctScores).sum() / N + reg * (W * W).sum()

	# Finding the gradient
	# grad 
	predictions = X @ W
	
	for i in range(N):
		dScores = np.zeros(C)
		S = np.exp(predictions[i, :]).sum()
		for cl in range(C):
			dScores[cl] += np.exp(predictions[i, cl]) / S
			if cl == y[i]:
				dScores[cl] -= 1
		dW += X[i, :, None] @ dScores[None, :]
	dW = dW / N + 2 * reg * W
	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

	return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
	"""
	Softmax loss function, vectorized version.

	Inputs and outputs are the same as softmax_loss_naive.
	"""
	# Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using no explicit loops.  #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	N, D = X.shape
	# Loss
	scores = X @ W # (N x C)
	scores -= np.max(scores, axis=1, keepdims=True)
	exps = np.exp(scores)
	softmaxScores = exps / exps.sum(axis=1, keepdims=True)

	losses = -np.log(softmaxScores[range(N), y])
	loss = losses.sum() / N + reg * (W ** 2).sum()

	# Gradient
	dScores = softmaxScores
	dScores[range(N), y] -= 1
	dW = X.T @ dScores / N
	dW += 2 * reg * W

	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

	return loss, dW
