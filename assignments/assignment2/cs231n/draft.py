
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

    # 1. Extract parameters from the dictionary (not a list!)
    S = conv_param['stride']
    P = conv_param['pad']

    # 2. Extract dimensions from input and filter shapes
    N, C, H, W = x.shape        # input: batch, channels, height, width
    F, _, HH, WW = w.shape      # filters: num_filters, channels, filter_h, filter_w

    # 3. Compute output spatial dimensions (integer division!)
    H_prime = 1 + (H + 2 * P - HH) // S
    W_prime = 1 + (W + 2 * P - WW) // S

    # 4. Pad the input — only spatial dims H and W, not batch or channels
    #    pad_width: ((N_before, N_after), (C_before, C_after), (H_before, H_after), (W_before, W_after))
    x_padded = np.pad(x, ((0,0), (0,0), (P,P), (P,P)), mode='constant')

    # 5. Initialize the output array
    out = np.zeros((N, F, H_prime, W_prime))

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
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache
