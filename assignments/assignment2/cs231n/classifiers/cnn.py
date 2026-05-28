from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # 
        
        # self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        

        # correction: 
        C, H, W = input_dim
        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale # L5 slide 94 the weight matrix.
        # b1 = np.zeros(M,)
        # C = num_classes
        # W2 with shape (M,C) = np.Gaussian(0.0, weight_scale)
        # b2 = np.zeros(C, )

        # b1 shape: (hidden_dim,)
        # self.params['b1'] = np.zeros(hidden_dim)
        self.params['b1'] = np.zeros(num_filters)

        # outputsize of conv layer: C_out * H' * W'
        # H' = (H-K+2P)/S + 1
        '''
        The general formula for the output height of any pooling layer is: $$H_{out} = \frac{H - \text{pool_height}}{\text{stride}} + 1$$

        If we plug in our values ($\text{pool_height}=2$, $\text{stride}=2$): $$H_{out} = \frac{H - 2}{2} + 1$$ $$H_{out} = \left(\frac{H}{2} - 1\right) + 1$$ $$H_{out} = \frac{H}{2}$$
        '''
        # C_out = num_filters
        # weight matrix: C_out * C_in * filter_size

        # 

        # Layer 2: Affine (Hidden -> Classes)
        # W2 shape: (hidden_dim, num_classes)
        # self.params['W2'] = np.random.randn(hidden_dims[0], num_classes) * weight_scale

        # correction: "the width and height of the input are preserved"
        # The instructions say the Convolutional layer preserves the height (H) and width (W). So after Conv, your shape is (N, num_filters, H, W).
        # there is no N, because we look at each one of the N images. and N is not required in this initialization frame works.
        # afther the conv, relu, maxpool,  height and width are: H/2, W/2
        # number of filters: num_filters
        # so, the input size of affine layer: num_filters * H/2 * W/2

        # self.params['W2'] = np.random.randn(num_filters, H // 2, W // 2) * weight_scale
        # before entering affine, the data is flattened to a long one dim vector:
        # flattened_input = num_filters * H // 2 * W // 2
        self.params['W2'] = np.random.randn(num_filters * H // 2 * W // 2, hidden_dim) * weight_scale

        self.params['b2'] = np.zeros(hidden_dim)

        
        
        

        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale

        self.params['b3'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # 
        caches = []
        out = X        
        # Layer 1
        # can use " conv_relu_forward ", or "conv_relu_pool_forward"
        out1, cache_conv = conv_forward_im2col(X, W1, b1, conv_param)  # out1 used to be temp1
        out_relu1, cache_relu1 = relu_forward(out1)

        # out_maxpool, cache_maxpool  = maxpool(out_relu1)
        out_maxpool, cache_maxpool  = max_pool_forward_fast(out_relu1, pool_param)
        
        #---
        out_affine1, cache_aff1 = affine_forward(out_maxpool, W2, b2)
        out_relu2, cache_relu2 = relu_forward(out_affine1)

        # --  can use "affine_relu_forward"

        # advanced implementation:
    #    out, cache = conv_relu_pool_forward(x, w, b, conv_param, pool_param)

        out_affine2, cache_aff2 = affine_forward(out_relu2, W3, b3)
        # scores = softmax(out_affine2)
        # correction: 
        scores = out_affine2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # 
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)) #  + np.sum(W4**2) + np.sum(W5**2) + np.sum(W6**2)

        loss = data_loss + reg_loss

        #-- can use "affine_relu_backward"
        # conv_relu_pool_backward(dout, cache)

        dx3, dw3, db3 = affine_backward(dscores, cache_aff2)
        d_out3 = relu_backward(dx3, cache_relu2)
        
        dx2, dw2, db2 = affine_backward(d_out3, cache_aff1)
        # dx1_maxpool = maxpool_backward(dx2, cache_maxpool)
        dx1_maxpool = max_pool_backward_fast(dx2, cache_maxpool)

        # can use "conv_relu_backward"
        dx1_relu = relu_backward(dx1_maxpool, cache_relu1)
        # dx1, dw1, db1 = conv_backward(dx1_relu, cache_conv)
        dx1, dw1, db1 = conv_backward_im2col(dx1_relu, cache_conv)

        # missed these lines:
        grads['b3'] = db3 # + self.reg * b3
        grads['b2'] = db2 # + self.reg * b2
        grads['b1'] = db1 # + self.reg * b1

        grads['W3'] = dw3 + self.reg * W3
        grads['W2'] = dw2 + self.reg * W2
        grads['W1'] = dw1 + self.reg * W1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
