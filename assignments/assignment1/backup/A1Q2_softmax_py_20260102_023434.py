from builtins import range
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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]  # i.e. C from "- W: A numpy array of shape (D, C) containing weights."
    num_train = X.shape[0] # N
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize

        # # start to implement gradient.
        # _, pred_label = max(logp)
        # if y[i] == pred_label:
        #   one_matrix_entry = 1
        # else:
        #   one_matrix_entry = 0

        # # p is the p before it gets to the log.

        # dW = dW + (-1)/p*(p - one_matrix_entry)
        # # end of implement gradient.


        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
        '''
        explain why y[i] can be the index to logp:

        As defined in the docstring, y is an array of training labels where y[i] = c means the $i$-
        th example belongs to class $c$.The values in y are integers ranging from $0$ to $C
        -1$.Because y[i] is an integer that represents a class category, it functions perfectly as
        an index to look up values in any vector of size $C$.


        Imagine you have 3 classes: [0: cat, 1: dog, 2: bird].
        Labels: Suppose for image $i$, the label is a dog, so y[i] = 1.
        Probabilities: Your model outputs p = [0.1, 0.7, 0.2].
        Logs: logp becomes [log(0.1), log(0.7), log(0.2)].
        Selection: logp[y[i]] is logp[1], which is log(0.7).
        Loss: Your code then does loss -= log(0.7), which correctly calculates the negative log-likelihood of the correct class.

        '''
        # 4. Compute Gradient (Backprop)
        # We need to update dW for every class j
        for j in range(num_classes):
            # Calculate dL/dz for this specific class j
            # Dimension: Scalar
            if j == y[i]:
                dscore = p[j] - 1
            else:
                dscore = p[j]
            
            # Update the gradient for the j-th column of W
            # X[i] shape: [D,]
            # dW[:, j] shape: [D,]
            dW[:, j] += X[i] * dscore        




    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    # W * W Element-wise Operation, where every entry is $w_{i,j}^2$.
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    


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
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################

    # compute the loss and the gradient
    num_classes = W.shape[1]  # i.e. C from "- W: A numpy array of shape (D, C) containing weights."
    num_train = X.shape[0] # N
    # for i in range(num_train):
    scores = X @ W

    # compute the probabilities in numerically stable way
    # scores -= np.max(scores)
    scores -= np.max(scores, axis=1, keepdims=True)  # (N, C)
    # p = np.exp(scores)
    # p /= p.sum()  # normalize
    # 3. Compute softmax probabilities
    exp_scores = np.exp(scores)  # (N, C)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (N, C)
    # # start to implement gradient.
    # _, pred_label = max(logp)
    # if y[i] == pred_label:
    #   one_matrix_entry = 1
    # else:
    #   one_matrix_entry = 0

    # # p is the p before it gets to the log.

    # dW = dW + (-1)/p*(p - one_matrix_entry)
    # # end of implement gradient.


    # logp = np.log(p)
    log_probs = np.log(probs)  # (N, C)

    # loss -= sum(logp[y])  # negative log probability is the loss
    # Create row indices
    # row_indices = np.arange(logp.shape[0])  # [0, 1, 2, ..., N-1]
    # 5. Select correct class log probabilities and compute loss
    correct_log_probs = log_probs[np.arange(num_train), y]  # (N,)

    # Select the correct column for each row
    # loss -= np.sum(logp[row_indices, y])
    data_loss = -np.sum(correct_log_probs) / num_train
    # 4. Compute Gradient (Backprop)
    # We need to update dW for every class j
    # for j in range(num_classes):
    #     # Calculate dL/dz for this specific class j
    #     # Dimension: Scalar
    #     if j == y[i]:
    #         dscore = p[j] - 1
    #     else:
    #         dscore = p[j]
        
    #     # Update the gradient for the j-th column of W
    #     # X[i] shape: [D,]
    #     # dW[:, j] shape: [D,]
    #     dW[:, j] += X[i] * dscore        




    # normalized hinge loss plus regularization
    # loss = loss / num_train + reg * np.sum(W * W)
    loss = data_loss + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                     
    # dscore = np.zeros(num_train, num_classes)   
    # dscore = np.zeros_like(log_probs)                                             #
    # for i in range(num_train): # range(0,N):
    #   for j in range(num_classes):
    #     if j == y[i]:
    #         dscore[i][j] = probs[i][j] - 1  # use probs, not log_probs.
    #     else:
    #         dscore[i][j] = probs[i][j]

    # Start with probs (the gradient for incorrect classes)
    dscore = probs.copy()  # (N, C)

    # Subtract 1 from the correct class positions
    dscore[np.arange(num_train), y] -= 1  # This handles the (p_j - 1) case

    dW = dW + X.T @ dscore

    dW /= num_train
    dW += 2 * reg * W    
    #############################################################################


    return loss, dW
