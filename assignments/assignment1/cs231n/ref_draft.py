from builtins import range
import numpy as np
# import math


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # 1. Get the batch size N from the input shape
    N = x.shape[0]    
    
    # x = np.reshape(x, [N, D])
    # using -1 lets NumPy automatically calculate dimension D (d_1 * ... * d_k)
    x_reshaped = x.reshape(N, -1)
    
    # out = x @ w
    out = x_reshaped @ w + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # Important: Cache the ORIGINAL x, not the reshaped one.
    # The backward pass needs the original shape to reshape dx correctly.
    
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # dx = dout @ w.T
    # dw = x.T @ dout
    # db = np.ones(1,N) @ dout
    
# 1. Get the batch size N
    N = x.shape[0]
    # 2. Re-create the flattened input (N, D) just like in forward pass
    x_reshaped = x.reshape(N, -1)
    
    # 3. Calculate dx (Gradient of inputs)
    # dout: (N, M), w.T: (M, D) -> Result: (N, D)
    dx_flat = dout @ w.T
    # RESHAPE back to original (N, d_1, ..., d_k)
    dx = dx_flat.reshape(x.shape)
    
    # 4. Calculate dw (Gradient of weights)
    # x_reshaped.T: (D, N), dout: (N, M) -> Result: (D, M)
    dw = x_reshaped.T @ dout
    
    
    # Fix db: Use sum, not ones The Problem: np.ones(1,N) is a syntax error (it needs an extra pair of parenthesis ((1,N))). More importantly, using matrix multiplication to sum columns is valid but rarely done in Python.The Fix: It is standard practice to use np.sum(dout, axis=0). This collapses the batch dimension $N$, leaving you with a gradient vector of size $M$.
    # meaning: db = np.ones(1,N) @ dout is correct, but rare in practice. the same with "db = np.sum(dout, axis=0)"
    
    # 5. Calculate db (Gradient of biases)
    # Sum gradients across the batch dimension (axis 0)
    # dout: (N, M) -> Result: (M,)
    db = np.sum(dout, axis=0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # if x >= 0:
    #     output = x
    # else:
    #     output = 0    

    # This applies: f(x) = max(0, x) element-wise to the entire array
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
# if one element of the x <= 0:
#     dx = 0
#     else:
#         dx = dout # i.e.  dout * 1

    # 1. Start with the upstream gradient (dout)
    dx = dout.copy()
    
    # 2. Apply the "Gate" logic: 
    # If x[i] <= 0, the local gradient is 0, so 0 * dout = 0.
    # If x[i] > 0, the local gradient is 1, so 1 * dout = dout.
    dx[x <= 0] = 0
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # compute the loss and the gradient
    num_classes = x.shape[1]  # i.e. C from "- W: A numpy array of shape (D, C) containing weights."
    num_train = x.shape[0] # N

    # compute the probabilities in numerically stable way
    # scores -= np.max(scores)
    scores = x # scores = X @ W
    
    scores = scores - np.max(scores, axis=1, keepdims=True) # missed this line, lead to results dx error 0.3333
    
    scores -= np.max(scores, axis=1, keepdims=True)  # (N, C)  # one error here, did two times. but results does not chnage much.) it's that I did two times of the minus max. first time to all become 0.1 0.05 etc, then again minues the 0.1, be close to 0.
    
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

    loss = data_loss


    # Start with probs (the gradient for incorrect classes)
    dscore = probs.copy()  # (N, C)

    # Subtract 1 from the correct class positions
    dscore[np.arange(num_train), y] -= 1  # This handles the (p_j - 1) case
    
    dx = dscore
    
    dx /= num_train

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # sample_mean = mean(x)
        sample_mean = np.mean(x, axis=0)

        # sample_var = (std(x))**2
        sample_var = np.var(x, axis=0)

        # out = (x - sample_mean)/sqrt(sample_var)
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)

        out = gamma * x_hat + beta

        # missed this.
        # beta and gamma is to be learned.

        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var        
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        pass
        cache = (x, x_hat, gamma, sample_mean, sample_var, eps)
    
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # out = (x - running_mean)/sqrt(running_var)
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # dx = 1 / np.sqrt(cache[4] + cache[5]) * cache[2] * dout
    # Unpack cache
    x, x_hat, gamma, sample_mean, sample_var, eps = cache
    N = x.shape[0]

    sigma_eps = (sample_var + eps) ** (-0.5)
    dx = (1.0 / N) * gamma * sigma_eps * (
        N * dout
        - np.sum(dout, axis=0)
        - x_hat * np.sum(dout * x_hat, axis=0)
    )

    # cache[2] is gamma
    # dgamma = cache[1] * dout
    dgamma = np.sum(cache[1] * dout, axis=0)

    # cache[1] is x_hat
    # dbeta = 1*dout
    dbeta = np.sum(1*dout, axis=0)   

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    dgamma = np.sum(cache[1] * dout, axis=0)
    # cache[1] is x_hat
    # dbeta = 1*dout
    dbeta = np.sum(1*dout, axis=0)          

    # my attempt start       
    x, x_hat, gamma, sample_mean, sample_var, eps = cache
    N = x.shape[0]

      # # sigma_eps = (sample_var + eps) ** (-0.5)
      # dx = (1/N*(x-sample_mean)*(sample_var+eps)**(-0.5) * (x - sample_mean) -1/N  )*dout
    # my attempt end

    # gemini corrections:

    # Conceptual example of Question 1's intended solution:
    
    dxhat = dout * gamma
    dvar = np.sum(dxhat * (x - sample_mean) * -0.5 * (sample_var + eps)**(-1.5), axis=0)
    dmu = np.sum(dxhat * -1 / np.sqrt(sample_var + eps), axis=0) + dvar * np.mean(-2 * (x - sample_mean), axis=0)

    dx1 = dxhat * 1 / np.sqrt(sample_var + eps) # gradient from x_hat directly
    dx2 = dvar * 2 * (x - sample_mean) / N      # gradient from variance
    dx3 = dmu * 1 / N                           # gradient from mean

    dx = dx1 + dx2 + dx3

    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # 
    # sample_mean_feature = np.mean(x, axis=1)
    # sample_var_feature = np.var(x, axis=1)
    # x_hat_feature = (x - sample_mean_feature) / np.sqrt(sample_var_feature + eps)

    # x = x_hat_feature
    # sample_mean = np.mean(x, axis=0)

    # # sample_var = (std(x))**2
    # sample_var = np.var(x, axis=0)

    # # out = (x - sample_mean)/sqrt(sample_var)
    # x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)

    # out = gamma * x_hat + beta    

    # cache = (x, x_hat, gamma, sample_mean, sample_var, eps)

    # second attempt:

    # x = x.T
    # sample_mean = np.mean(x, axis=0)

    # # sample_var = (std(x))**2
    # sample_var = np.var(x, axis=0)

    # # out = (x - sample_mean)/sqrt(sample_var)
    # x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)

    # 
    # x_hat = x_hat.T

    # # You need to apply gamma and beta after transposing back, or handle the shapes carefully.

    # out = gamma * x_hat + beta    

    # cache = (x, x_hat, gamma.T, sample_mean.T, sample_var.T, eps)

    # third attempt (correct):
    # Transpose trick: x (N,D) -> x_t (D,N), so batchnorm over axis=0 = layernorm
    x_t = x.T                                              # (D, N)
    sample_mean = np.mean(x_t, axis=0)                      # (N,)
    sample_var = np.var(x_t, axis=0)                         # (N,)
    x_hat_t = (x_t - sample_mean) / np.sqrt(sample_var + eps)  # (D, N)

    # here, we make the gamma_batchnorm inside the layernorm's batchnorm's code to be 1, beta_batchnorm to be 0.

    # Transpose back to (N, D), then apply gamma and beta
    x_hat = x_hat_t.T                                       # (N, D)
    out = gamma * x_hat + beta                               # (N, D)
    # this gamma and beta belongs to the layernorm, not the batchnorm.

    # Cache transposed x and x_hat for backward, plus gamma
    cache = (x_t, x_hat_t, gamma, sample_mean, sample_var, eps)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################

    # dout = dout.T
    # x, x_hat, gamma, sample_mean, sample_var, eps = cache

    # N = x.shape[0]

    # sigma_eps = (sample_var + eps) ** (-0.5)
    # dx = (1.0 / N) * gamma * sigma_eps * (
    #     N * dout
    #     - np.sum(dout, axis=0)
    #     - x_hat * np.sum(dout * x_hat, axis=0)
    # )

    # # cache[2] is gamma
    # # dgamma = cache[1] * dout
    # dgamma = np.sum(cache[1] * dout, axis=0)

    # # cache[1] is x_hat
    # # dbeta = 1*dout
    # dbeta = np.sum(1*dout, axis=0)   

    # dx = dx.T

    # correct version:
    x_t, x_hat_t, gamma, sample_mean, sample_var, eps = cache
    # x_t: (D, N), x_hat_t: (D, N), gamma: (D,)

    # Step 1: backprop through out = gamma * x_hat + beta  (in N,D space)
    x_hat = x_hat_t.T                                      # (N, D)
    dgamma = np.sum(dout * x_hat, axis=0)                   # (D,)
    dbeta = np.sum(dout, axis=0)                             # (D,)

    # dx_hat = dout * gamma, then transpose to (D, N) for normalization backward

    dx_hat = dout * gamma
    dx_hat_t = dx_hat.T                             # (D, N)

    # Step 2: backprop through normalization (same as batchnorm backward, but gamma=1)
    N_feat = x_t.shape[0]   # D - the "batch size" in transposed space
    sigma_eps = (sample_var + eps) ** (-0.5)                 # (N,)
    dx_t = (1.0 / N_feat) * sigma_eps * ( # gamma_batchnorm is removed, because we set it to 1.0 in the forward pass.
        N_feat * dx_hat_t
        - np.sum(dx_hat_t, axis=0)
        - x_hat_t * np.sum(dx_hat_t * x_hat_t, axis=0)
    )

    # Step 3: transpose back
    dx = dx_t.T                                             # (N, D)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # for i in random(0,1), if i>p, x[i_index] = 0 .  eg, p=0.25, we have 25% chance to keep a value, 75% to drop it. so the drop rate is 75%.
        # as said there in the results: Fraction of train-time output set to zero:  0.749784
        # out = x  
        mask = (np.random.rand(*x.shape) < p) / p  # the " / p", intuitively, should be at the end of line 677, but put it there to make the backward easier and more concise.
        out = x * mask        
         
         # the "/p" is due to these logic below:
        # 	Vanilla Dropout	                 | Inverted Dropout
        # Train time	out = x * mask          |	out = x * mask / p ← scale here
        # Test time	out = x * p ← scale here	|out = x (nothing!)        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = mask * dout
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    # F = np.shape(w, 0)
    F = w.shape[0]
  
    K = F
    P = conv_param['pad'] # conv_param[1]
    S = conv_param['stride'] # conv_param[0]
    # x = np.pad(x, P)
    x_padded = np.pad(x, ((0,0), (0,0), (P,P), (P,P)), mode='constant')

# Bug 4: N, C, H, W, HH, WW are never defined
# Add 
    N, C, H, W = x.shape # and 
    F, _, HH, WW = w.shape # before using them.

    H_prime = 1 + (H + 2 * P - HH) // S
    W_prime = 1 + (W + 2 * P - WW) // S

    # missed this
    # 5. Initialize the output array
    out = np.zeros((N, F, H_prime, W_prime))

    #  The correct loops are n → f → i → j (image, filter, output row, output col). 
    # At each position, extract a patch of shape (C, HH, WW) and do np.sum(patch * w[f]) + b[f].

    # for f in range(0,F):
    #   for i in range(0,N):
    #       for c in range(0,C):
    #           x_prime[0,0] = x[0:HH,0:WW] * w
    #           # then WW = WW+1, x_prime[0,1] = x[0:HH,1:WW+1] * w , ...
    


    # 6. Perform the convolution — NO loop over channels!
    #    One filter spans ALL C channels simultaneously.
    for n in range(N):                    # each image in the batch
        for f in range(F):                # each filter
            for i in range(H_prime):      # each output row
                for j in range(W_prime):  # each output column
                    # Spatial window start positions
                    h_start = i * S
                    w_start = j * S
                    # Extract the receptive field: all C channels, spatial window (HH x WW)
                    patch = x_padded[n, :, h_start:h_start+HH, w_start:w_start+WW]
                    # Element-wise multiply across (C, HH, WW), sum everything → one scalar
                    out[n, f, i, j] = np.sum(patch * w[f]) + b[f]

    # out = x_prime + b
    # H_prime = 1 + (H + 2 * P - HH) / stride
    # W_prime = 1 + (W + 2 * P - WW) / stride    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    (x, w, b, conv_param) = cache
    # F = np.shape(w, 0)
    F = w.shape[0]
  
    K = F
    P = conv_param['pad'] # conv_param[1]
    S = conv_param['stride'] # conv_param[0]
    # x = np.pad(x, P)
    x_padded = np.pad(x, ((0,0), (0,0), (P,P), (P,P)), mode='constant')

# Bug 4: N, C, H, W, HH, WW are never defined
# Add 
    N, C, H, W = x.shape # and 
    F, _, HH, WW = w.shape # before using them.

    H_prime = 1 + (H + 2 * P - HH) // S
    W_prime = 1 + (W + 2 * P - WW) // S

    #  The correct loops are n → f → i → j (image, filter, output row, output col). 
    # At each position, extract a patch of shape (C, HH, WW) and do np.sum(patch * w[f]) + b[f].

    # for f in range(0,F):
    #   for i in range(0,N):
    #       for c in range(0,C):
    #           x_prime[0,0] = x[0:HH,0:WW] * w
    #           # then WW = WW+1, x_prime[0,1] = x[0:HH,1:WW+1] * w , ...
    

    # initialize
    # dx = np.size_like(x)

    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # 6. Perform the convolution — NO loop over channels!
    #    One filter spans ALL C channels simultaneously.
    for n in range(N):                    # each image in the batch
        for f in range(F):                # each filter
            for i in range(H_prime):      # each output row
                for j in range(W_prime):  # each output column
                    # Spatial window start positions
                    h_start = i * S
                    w_start = j * S
                    # Extract the receptive field: all C channels, spatial window (HH x WW)
                    patch = x_padded[n, :, h_start:h_start+HH, w_start:w_start+WW]
                    # Element-wise multiply across (C, HH, WW), sum everything → one scalar
                    # dx_padded[n, f, i, j] = np.sum(w[f])*dout[n, f, i, j]
                    dpatch = w[f] * dout[n, f, i, j]  
                    # this is from z = patch element_wise_* w[f]
                    # dx_padded[n, :, h_start:h_start+HH, w_start:w_start+WW] += np.sum(w[f])*dout[n, f, i, j]
                    dx_padded[n, :, h_start:h_start+HH, w_start:w_start+WW] += dpatch
                    # dw[n, f, i, j] = np.sum(patch)*dout[n, f, i, j]
                    # dw[f] has shape (C, HH, WW) — same as the filter. You accumulate (+=) across all n, i, j.
                    dw[f] += patch*dout[n, f, i, j]
                    '''
                    dw[f, c, hh, ww] += patch[c, hh, ww] × dout[n, f, i, j]
                    Why you can write it without indices
                    Since this same formula applies to every [c, hh, ww] independently, NumPy lets you write all 27 (or however many) at once:

                    python
                    dw[f] += patch * dout[n, f, i, j]
                    This is just shorthand for all 27 lines at once. dout[n,f,i,j] is a scalar that gets broadcast to every element.
                    '''
                    # dpatch[c, hh, ww]  = w[f, c, hh, ww]  × dout[n,f,i,j]
                    
                    # correction: 
                    # 1. need "+=", it's matrix accumulation.
                    # 2. should use "dx_padded"
                    #                     
                    # db[n, f, i, j] = 1*dout[n, f, i, j]
                    # this is equivalent to line 885.
                    # The forward pass was: out[n, f, i, j] = np.sum(patch * w[f]) + b[f]
                    # b[f] is added to every output position (n, i, j) for filter f      
                    #               
    # same level with the first "for" loop beginner.
    db = np.sum(dout, axis=(0, 2, 3))   # sum over batch, height, width → shape (F,)


                    
    if P > 0:
        dx = dx_padded[:, :, P:-P, P:-P]
    else:
        dx = dx_padded

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # 
    N, C, H, W = x.shape
    # H_prime = 1 + math.floor((H - pool_param["pool_height"]) / pool_param["stride"])
    # W_prime = 1 + math.floor((W - pool_param["pool_width"]) / pool_param["stride"])   
    
    H_prime = 1 + (H - pool_param["pool_height"]) // pool_param["stride"]
    W_prime = 1 + (W - pool_param["pool_width"]) // pool_param["stride"]     


    # F = np.shape(w, 0)

    
    S = pool_param["stride"]
    HH = pool_param["pool_height"]
    WW = pool_param["pool_width"]


    # 5. Initialize the output array
    out = np.zeros((N, C, H_prime, W_prime))

# 

    #  The correct loops are n → f → i → j (image, filter, output row, output col). 
    # At each position, extract a patch of shape (C, HH, WW) and do np.sum(patch * w[f]) + b[f].

    # for f in range(0,F):
    #   for i in range(0,N):
    #       for c in range(0,C):
    #           x_prime[0,0] = x[0:HH,0:WW] * w
    #           # then WW = WW+1, x_prime[0,1] = x[0:HH,1:WW+1] * w , ...
    


    # 6. Perform the convolution — NO loop over channels!
    #    One filter spans ALL C channels simultaneously.
    for n in range(N):                    # each image in the batch
        for c in range(C):                # each channel
            for i in range(H_prime):      # each output row
                for j in range(W_prime):  # each output column
                    # Spatial window start positions
                    h_start = i * S
                    w_start = j * S
                    # Extract the receptive field: all C channels, spatial window (HH x WW)
                    patch = x[n, c, h_start:h_start+HH, w_start:w_start+WW]
                    # Element-wise multiply across (C, HH, WW), sum everything → one scalar
                    out[n,c, i, j] = np.max(patch)

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # 

    (x, pool_param) = cache
    N, C, H, W = x.shape
    # H_prime = 1 + math.floor((H - pool_param["pool_height"]) / pool_param["stride"])
    # W_prime = 1 + math.floor((W - pool_param["pool_width"]) / pool_param["stride"])   
    
    H_prime = 1 + (H - pool_param["pool_height"]) // pool_param["stride"]
    W_prime = 1 + (W - pool_param["pool_width"]) // pool_param["stride"]         

    # 5. Initialize the output array
    dx = np.zeros_like(x)

    
    S = pool_param["stride"]
    HH = pool_param["pool_height"]
    WW = pool_param["pool_width"]

    for n in range(N):                    # each image in the batch
        for c in range(C):                # each channel
            for i in range(H_prime):      # each output row
                for j in range(W_prime):  # each output column
                    # Spatial window start positions
                    h_start = i * S
                    w_start = j * S
                    # Extract the receptive field: all C channels, spatial window (HH x WW)
                    patch = x[n, c, h_start:h_start+HH, w_start:w_start+WW]
                    # Element-wise multiply across (C, HH, WW), sum everything → one scalar
                    # dx_padded[n, f, i, j] = np.sum(w[f])*dout[n, f, i, j]
                    # dpatch = w[f] * dout[n, f, i, j]  
                    # # this is from z = patch element_wise_* w[f]
                    # # dx_padded[n, :, h_start:h_start+HH, w_start:w_start+WW] += np.sum(w[f])*dout[n, f, i, j]
                    # dx_padded[n, :, h_start:h_start+HH, w_start:w_start+WW] += dpatch
                    # # dw[n, f, i, j] = np.sum(patch)*dout[n, f, i, j]
                    # # dw[f] has shape (C, HH, WW) — same as the filter. You accumulate (+=) across all n, i, j.
                    # dw[f] += patch*dout[n, f, i, j]
                    
                    # dx[n,c,i,j] = 1*dout[n,c,i,j]
                    # dx[n,c,h_start,w_start] = 1*dout[n,c,i,j]
                    # dx[n, c, h_start:h_start+HH, w_start:w_start+WW] += 0
                    # dx[n, c,(patch == np.max(patch))[2,3]] += 1*dout[n,c,i,j]

                    mask = (patch == np.max(patch))
                    gradient_to_add = mask * dout[n, c, i, j]
                    dx[n, c, h_start:h_start+HH, w_start:w_start+WW] += gradient_to_add

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
