from builtins import range
import numpy as np
from copy import deepcopy

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = x.shape[0]
    z = x.reshape(N, -1) @ w + b
    out = z
    # out = np.maximum(0, z)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    # Forward propogate
    z = x_flat @ w + b

    # Backward propogate
    # dout[z < 0] = 0 # ReLU kills these gradients
    dw = x_flat.T @ dout # Matmul backprop
    dx = (dout @ w.T).reshape(x.shape) # Matmul backprop
    db = dout.sum(axis=0)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dx = dout
    dx[x < 0] = 0
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

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
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        mu = x.mean(axis=0)
        var = ((x - mu) ** 2).mean(axis=0) # [D]
        std = np.sqrt(var + eps)
        z = (x - mu) / std # [N x D]
        out = gamma * z + beta # [N x D]

        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var

        cache = (x, std, z, gamma)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * out + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache

def divide_forward(a, b, eps=1e-5):
    cache = (a, b)
    out = a / b
    return (out, cache)

def divide_backward(dout, cache):
    a, b = cache
    da = dout / b
    db = (- a * dout / b ** 2).sum(axis=0)
    return (da, db)

def std_forward(delta):
    N = delta.shape[0]

    deltaSq = delta ** 2 # [N x D]
    var = deltaSq.sum(axis=0) / N # [D]
    std = np.sqrt(var) # [D]

    out = std
    cache = (delta, std)
    return out, cache

def std_backward(dout, cache):
    delta, std = cache
    N, D = delta.shape

    dstd = dout # [D]
    dVar = dstd / (2 * std) # [D]
    dDeltaSq = np.ones((N, D)) * dVar / N # [N x D]
    dDelta = 2 * delta * dDeltaSq # [N x D]

    return dDelta

def substractMean_forward(X):
    out = X - X.mean(axis=0)
    cache = X
    return out, cache

def substractMean_backward(dout, cache):
    X = cache
    
    N, D = X.shape
    # dout has shape [N x D]

    dX1 = dout
    
    dmu = - dout.sum(axis=0) # [D]
    dX2 = np.ones((N, D)) * dmu / N # [N x D]

    dX = dX1 + dX2
    return dX



def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X, std, z, gamma = cache
    N, D = X.shape

    dbeta = dout.sum(axis=0) # [D]
    dgamma = (z * dout).sum(axis=0) # [D]
    dz = gamma * dout # [N x D]

    # =============== Modular Solution =====================
    delta = X - X.mean(axis=0)
    divide_cache = (delta, std)
    dDelta1, dSigma = divide_backward(dz, divide_cache)

    std_cache = (delta, std)
    dDelta2 = std_backward(dSigma, std_cache)

    dDelta = dDelta1 + dDelta2

    subtractMean_cache = X
    dx = substractMean_backward(dDelta, subtractMean_cache)


    #=============== Non-modular Solution ==================
    # Find delta1
    # dDelta1 = dz / (std) # [N x D]
    # # Find delta2
    # dstd = (-(X - mu) / ((std_cache) ** 2) * dz).sum(axis=0) # [D]
    # dvar = dstd / (2 * std) # [D]
    # dDeltaSq = np.ones((N, D)) * dvar / N # [N x D]
    # dDelta2 = 2 * (X - mu) * dDeltaSq # [N x D]
    # # Combine
    # dout_ = dDelta1 + dDelta2 # [N x D]

    # # Find dX1
    # dX1 = dout_ # [N x D]
    # # Find dX2
    # dmu = -dout_.sum(axis=0) # [D]
    # dX2 = np.ones((N, D)) * dmu / N # [N x D]
    # # Combine
    # dx = dX1 + dX2

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

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
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, std, z, gamma = cache

    dbeta = dout.sum(axis=0) # [D]
    dgamma = (z * dout).sum(axis=0) # [D]

    dz = gamma * dout
    dx = dz - dz.mean(axis=0) - z * (gamma * dout * z).mean(axis=0)
    dx /= std
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

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
    # the batch norm code and leave it almost unchanged? Yes, take the transpose                     #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Transform params
    N, D = x.shape

    # Apply batchnorm methods
    mu = x.mean(axis=1).reshape((N, 1)) # [N x 1]
    var = ((x - mu) ** 2).mean(axis=1).reshape((N, 1)) # [N x 1]
    std = np.sqrt(var + eps) # [N x 1]
    z = (x - mu) / std # [N x D]
    out = gamma.reshape((1, D)) * z + beta.reshape((1, D)) # [N x D]

    cache = (x, std, z, gamma)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, std, z, gamma = cache
    N, D = x.shape

    dbeta = dout.sum(axis=0)
    dgamma = np.sum(dout * z, axis=0)

    batchnorm_cache = (x.T, std.reshape((1, N)), z.T, gamma.reshape((D, 1)))
    dx = batchnorm_backward_alt(dout.T, batchnorm_cache)[0].T
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

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

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # dist = np.random.rand(x.shape)
        mask = np.random.rand(*x.shape) < p
        out = x * mask / p
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        mask = np.ones(x.shape)
        out = x
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dx = dout * mask / dropout_param["p"]
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    stride, pad = conv_param["stride"], conv_param["pad"]
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    # print('w shape', w.shape)
    Hprime = 1 + (H + 2 * pad - HH) // stride
    Wprime = 1 + (W + 2 * pad - WW) // stride

    pad_width = [(0, 0), (0, 0), (pad, pad), (pad, pad)]
    padded = np.pad(x, pad_width)
    out = np.zeros((N, F, Hprime, Wprime))

    # --------------- Method using matrix multiplication ------------------------   
    w_row = w.reshape((F, -1))
    for i in range(N):
        x_i = padded[i, :, :]
        x_col = np.zeros((C * HH * WW, Hprime * Wprime))
        for hprime in range(Hprime):
            for wprime in range(Wprime):
                ay = hprime * stride
                by = ay + HH
                ax = wprime * stride
                bx = ax + WW

                index = hprime * Hprime + wprime
                x_col[:, index] = x_i[:, ay:by, ax:bx].reshape(-1) # collapse matrix

        score_i = (w_row @ x_col).reshape((F, Hprime, Wprime))
        out[i, :, :, :] = score_i + b[:, None, None]

    # -------------- Method using array traversal ---------------------
    # for i in range(N):
    #     x_i = padded[i, :, :]
    #     score_i = np.zeros((F, Hprime, Wprime))
    #     for filter_num in range(F):
    #         for hprime in range(Hprime):
    #             for wprime in range(Wprime):
    #                 row = hprime * stride
    #                 col = wprime * stride

    #                 v = x_i[:, row : row + HH, col : col + WW]
    #                 f = w[filter_num, :, :]
    #                 convolved = (v * f).sum()
    #                 score_i[filter_num, hprime, wprime] = convolved + b[filter_num]
    #     out[i] = score_i


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, w, b, conv_param = cache
    stride, pad = conv_param["stride"], conv_param["pad"]
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    Hprime = 1 + (H + 2 * pad - HH) // stride
    Wprime = 1 + (W + 2 * pad - WW) // stride

    db = dout.sum(axis=(0, 2, 3)) # sum across examples, width and height (everything but filters)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)

    # # Idea: repeat convolution procedure, constructing dx as you go

    # Semi-Vectorized
    for hprime in range(Hprime):
        for wprime in range(Wprime):
            ay = hprime * stride
            ax = wprime * stride
            grad_cell = dout[:, :, hprime, wprime, None]
            for convrow in range(HH):
                for convcol in range(WW):
                    iX = ay + convrow - pad # unpad indices
                    jX = ax + convcol - pad # unpad indices
                    if not (0 <= iX < H and 0 <= jX < W):
                        continue
                    kernel_weight = w[None, :, :, convrow, convcol]
                    x_weight = x[:, None, :, iX, jX]
                    dx[:, :, iX, jX] += (kernel_weight * grad_cell).sum(axis=1)
                    dw[:, :, convrow, convcol] += (x_weight * grad_cell).sum(axis=0)

    # Unvectorized
    # for i in range(N):
    #     for filter_num in range(F):
    #         for hprime in range(Hprime):
    #             for wprime in range(Wprime):
    #                 ay = hprime * stride
    #                 ax = wprime * stride
    #                 f = w[filter_num]
    #                 grad_cell = dout[i, filter_num, hprime, wprime]
    #                 for channel in range(C):
    #                     for convrow in range(HH):
    #                         for convcol in range(WW):
    #                             iX = ay + convrow - pad # unpad indices
    #                             jX = ax + convcol - pad # unpad indices
                        
    #                             if not (0 <= iX < H and 0 <= jX < W):
    #                                 continue
    #                             kernel_weight = f[channel, convrow, convcol]
    #                             # print(f.shape)
    #                             x_weight = x[i, channel, iX, jX]
    #                             # print(kernel_weight.shape, grad_cell.shape)
    #                             dx[i, channel, iX, jX] += kernel_weight * grad_cell
    #                             dw[filter_num, channel, convrow, convcol] += x_weight * grad_cell
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    stride, pool_height, pool_width = pool_param["stride"], pool_param["pool_height"], pool_param["pool_width"]
    Hprime = 1 + (H - pool_height) // stride
    Wprime = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, Hprime, Wprime))
    for i in range(Hprime):
        for j in range(Wprime):
            ay = i * stride
            by = ay + pool_height
            ax = j * stride
            bx = ax + pool_width
            out[:, :, i, j] = x[:, :, ay:by, ax:bx].max(axis=(2, 3))


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x,  pool_param = cache
    N, CHANNELS, H, W = x.shape # Pro tip: don't use C for channels. Spent an hour after a scoping problem where I set C to the # of columns locally :))))
    stride, pool_height, pool_width = pool_param["stride"], pool_param["pool_height"], pool_param["pool_width"]
    Hprime = 1 + (H - pool_height) // stride
    Wprime = 1 + (W - pool_width) // stride

    dx = np.zeros_like(x)
    for i in range(Hprime):
        for j in range(Wprime):
            ay = i * stride
            by = ay + pool_height
            ax = j * stride
            bx = ax + pool_width
            for n in range(N):
                for channel in range(CHANNELS):
                    M = x[n, channel, ay:by, ax:bx]
                    ROWS, COLS = M.shape
                    maxIndex = M.argmax() # harder to vectorize since NumPy doesn't allow multi-axis argmax :()
                    rowMax = maxIndex // ROWS + ay
                    colMax = maxIndex % COLS + ax
                    dx[n, channel, rowMax, colMax] = dout[n, channel, i, j]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    flat = x.transpose(0, 2, 3, 1).reshape((N * H * W, C))
    flat, cache = batchnorm_forward(flat, gamma, beta, bn_param)
    out = flat.reshape((N, H, W, C)).transpose(0, 3, 1, 2)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape
    dout_flat = dout.transpose(0, 2, 3, 1).reshape((N * H * W, C))
    dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
    dx = dx_flat.reshape((N, H, W, C)).transpose(0, 3, 1, 2)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Shapes
    N, C, H, W = x.shape
    D = (C // G) * H * W
    g_shape = (N * G, D)
    s_shape = (1, C, 1, 1)

    gamma_fake = np.ones(D)
    beta_fake = np.zeros(D)
    z, layernorm_cache = layernorm_forward(x.reshape(g_shape), gamma_fake, beta_fake, gn_param)
    
    z = z.reshape(x.shape)
    out = z * gamma.reshape(s_shape) + beta.reshape(s_shape)

    # Layernorm cache has the form (xflat, std, zflat, gamma). Rewrite gamma
    (_, std, _, _) = layernorm_cache
    cache = (x, std, z, gamma, G)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

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
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Shapes
    x, std, z, gamma, G = cache
    N, C, H, W = x.shape
    D = (C // G) * H * W
    g_shape = (N * G, D)
    s_shape = (1, C, 1, 1)

    dbeta = dout.sum(axis=(0, 2, 3)).reshape(s_shape)
    dgamma = (z * dout).sum(axis=(0, 2, 3)).reshape(s_shape)
    
    dx = np.zeros_like(x)
    gamma_fake = np.ones(D)

    # layernorm cache has the form (xflat, std, zflat, gamma)
    layernorm_cache = (x.reshape(g_shape), std, z.reshape(g_shape), gamma_fake)

    dx_flat, _, _ = layernorm_backward((dout * gamma).reshape(g_shape), layernorm_cache)
    dx = dx_flat.reshape(x.shape)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx

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
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
