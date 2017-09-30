import numpy as np
from random import shuffle
from past.builtins import xrange

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

  num_train = X.shape[0]
  num_class = W.shape[1]
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  S = X.dot(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    s_i = S[i, :]
    sum_j = np.sum(np.exp(s_i))
    loss += - np.log(np.exp(s_i[y[i]]) / sum_j)
    
    for k in range(num_class):
        p_k = np.exp(s_i[k]) / sum_j
        dW[:, k] += (p_k - (k == y[i])) * X[i]
    
  
  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  num_train = X.shape[0]
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  S = X.dot(W)
  S_sum = np.sum(np.exp(S), axis = 1).reshape(-1, 1)
  p = np.exp(S) / S_sum
  loss = np.sum(-1 * np.log(p[np.arange(num_train), y]))
  
  m = np.zeros_like(p)
  m[np.arange(num_train), y] = 1 
  dW = X.T.dot(p - m)
   
  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += reg * W
 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

