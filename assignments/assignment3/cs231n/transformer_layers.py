import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################

        d = embed_dim

        # v1, my implementation. it works.
        # for i in range(max_len):
        #     for j in range(embed_dim):
        #         # if math.mod(j,2)==0:
        #         if j % 2 ==0:
        #             # pe[i,j] = math.sin(i*10000^(-j/d))
        #             pe[0, i, j] = math.sin(i*10000**(-j/d))
        #             '''
        #             Indexing pe: The tensor pe is initialized with shape (1, max_len, embed_dim). 
        #             Because it has 3 dimensions (with a batch dimension of 1 at index 0), you must 
        #             index it as pe[0, i, j] instead of pe[i, j].
        #             '''
        #         else:
        #             pe[0, i,j] = math.cos(i*10000**(-(j-1)/d))
                
        # v2, vectorised. it works too.
        # 1. Create a column vector of positions: shape (max_len, 1)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        
        # 2. Create the division terms for the even dimensions: shape (embed_dim / 2)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        
        # 3. Apply sine to even indices and cosine to odd indices
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################

        # x = x + self.pe
        # However, the input sequence x has a sequence 
        # length $S$ (which might be, for example, 30 words), 
        # but self.pe is precomputed with a length of 
        # max_len = 5000. If you add them directly, 
        # PyTorch will raise a shape mismatch error because $S \neq 5000$.        
        x = x + self.pe[:, :S, :]

        output = self.dropout(x)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        # split key
        H = self.n_head
        # key = torch.reshape(key, (N, T, H, E // H)) # correction: E // H
        # value = torch.reshape(value, (N, T, H, E//H))
        # see below. this is only initial implementation.

        # correction: D = self.head_dim
        D = self.head_dim # the D_H in L8 slide 67, and it is given.
        # or, the same value and format: D = E//H, because in __init__, we set self.head_dim = self.emd_dim // self.n_head.

        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # Q = X @ WQ
        # K = X @ WK
        # V = X @ WV
        # E = query @ key.T / torch.sqrt(S)
        # correction: 
        q = self.query(query)
        q = q.view(N, S, H, D).permute(0, 2, 1, 3)

        k = self.key(key)
        k = k.view(N, T, H, D).permute(0, 2, 1, 3)

        k_transpose = k.view(N, H, T, D).permute(0, 1, 3, 2)
        # or a smarter way to do it: 
        # k_transpose = k.transpose(-2, -1)  # Gives shape (N, H, D, T)

        v = self.value(value)
        v = v.view(N, T, H, D).permute(0, 2, 1, 3)

        # key = torch.reshape(key, (N, T, H, E // H)) # correction: E // H
        # value = torch.reshape(value, (N, T, H, E/H))
        # similarity:
        E_score = q @ k_transpose / math.sqrt(D)  # E (N, H, S, T) 

        # E_prime = E[:,:,S,T] torch.element_multiply attn_mask
        if attn_mask is not None:
            E_score= E_score.masked_fill(attn_mask == 0, float('-inf'))
        # assume DQ = DV = D_out, L8 slide 47.
        # A = torch.softmax(E, dim=2)
        # correction: Softmax should normalize attention weights across the 
        # target keys/values (dimension 3) so that they sum to 1.
        A = torch.softmax(E_score, dim=3)  # A (N, H, S, T) 
        
        A = self.attn_drop(A)
        Y = A @ v       # A (N, H, S, T) @ V (N, H, T, D) -> Y (N, H, S, D)

        # correction: 
        # You must permute the head dimension back and concatenate the heads to 
        # restore shape (N, S, E) before passing it to self.proj
        # Shape transition: Y (N, H, S, D) -> Y (N, S, H, D) -> Y (N, S, E)
        Y = Y.permute(0, 2, 1, 3).contiguous().view(N, S, E)
        output = self.proj(Y)  #  * Wo
        # where is Wo?
        # reply: Wo is within "self.proj()", the linear layer.
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        """
        Simple two-layer feed-forward network with dropout and ReLU activation.

        Inputs:
         - embed_dim: Dimension of input and output embeddings
         - ffn_dim: Hidden dimension in the feedforward network
         - dropout: Dropout probability
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass for the feedforward network.

        Inputs:
        - x: Input tensor of shape (N, T, D)

        Returns:
        - out: Output tensor of the same shape as input
        """
        out = torch.empty_like(x)

        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class TransformerDecoderLayer(nn.Module):
    """
    A single layer of a Transformer decoder, to be used with TransformerDecoder.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Construct a TransformerDecoderLayer instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_cross = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)


    def forward(self, tgt, memory, tgt_mask=None):
        """
        Pass the inputs (and mask) through the decoder layer.

        Inputs:
        - tgt: the sequence to the decoder layer, of shape (N, T, D) 
        Self-Attention (where tgt attends to itself with tgt_mask applied to prevent looking ahead).
        tgt contains the token embeddings of the sequence generated so far (or target sequence during training).
        - memory: the sequence from the last layer of the encoder, of shape (N, S, D)
        - tgt_mask: the parts of the target sequence to mask, of shape (T, T)

        Returns:
        - out: the Transformer features, of shape (N, T, W)
        """

        # Self-attention block (reference implementation)
        shortcut = tgt  # Line 333 stores a copy/reference of tgt in shortcut to form a residual (skip) connection (tgt = tgt + shortcut at line 336) around the multi-head self-attention layer.
        tgt = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        # Passes tgt as all three inputs: Query ($Q$), Key ($K$), and Value ($V$) because tokens attend to other tokens in the same target sequence.
        # attn_mask=tgt_mask: Applies a triangular causal mask so that position $t$ can only look at previous tokens ($\le t$) and cannot look into the future ($> t$).
        # i.e. S8P61, the top grey box.
        tgt = self.dropout_self(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_self(tgt)

        ############################################################################
        # TODO: Complete the decoder layer by implementing the remaining two       #
        # sublayers: (1) the cross-attention block using the encoder output as     #
        # memory, 

        # correction:
        shortcut = tgt

        # Cross-Attention (where tgt acts as the Query Q, while memory from the encoder acts as Key K and Value V).
        tgt = self.cross_attn(query=tgt, key=memory, value=memory)

        # correction:
        tgt = self.dropout_cross(tgt)
        tgt = tgt + shortcut

        # correction:
        tgt = self.norm_cross(tgt)
        
        # and (2) the feedforward block. Each block should follow the      #
        # same structure as self-attention implemented just above.                 #
        ############################################################################

        shortcut = tgt
        tgt = self.ffn(tgt)
        tgt = self.dropout_ffn(tgt)    
        tgt = tgt + shortcut    
        tgt = self.norm_ffn(tgt)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return tgt


class PatchEmbedding(nn.Module):
    """
    A layer that splits an image into patches and projects each patch to an embedding vector.
    Used as the input layer of a Vision Transformer (ViT).

    Inputs:
    - img_size: Integer representing the height/width of input image (assumes square image).
    - patch_size: Integer representing height/width of each patch (square patch).
    - in_channels: Number of input image channels (e.g., 3 for RGB).
    - embed_dim: Dimension of the linear embedding space.
    """
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=128):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels

        # Linear projection of flattened patches to the embedding dimension
        self.proj = nn.Linear(self.patch_dim, embed_dim)


    def forward(self, x):
        """
        Forward pass for patch embedding.

        Inputs:
        - x: Input image tensor of shape (N, C, H, W)

        Returns:
        - out: Patch embeddings with shape (N, num_patches, embed_dim)
        """
        N, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Expected image size ({self.img_size}, {self.img_size}), but got ({H}, {W})"
        out = torch.zeros(N, self.embed_dim)

        ############################################################################
        # TODO: Divide the image into non-overlapping patches of shape             #
        # (C x patch_size x patch_size), 
        
        # num_patches =  (H//self.patch_size) *  (W//self.patch_size)
        # self.num_patches is given

        # x = torch.reshape(x, (N, C, self.num_patches ,self.patch_size, self.patch_size))
        # correction:
        '''
Row 0 contains pixels from all patch columns ($P_1, P_2, P_3, \dots$) laid out horizontally.
Simply reshaping contiguous memory into blocks of size $P \times P$ takes horizontal strips of pixels from across the whole row, not 2D spatial square patches!        
        '''

        # correction:
        n_h = H//self.patch_size # how many rows do we want in a patch.
        n_w = W//self.patch_size

        P = self.patch_size

        x = torch.reshape(x, (N, C, n_h, P, n_w, P))
        # need to carefully imagine the sequence of how the pixel lies in the matrix.
        # only correct sequence can be reshaped to correct format.

        # x = torch.permute(x, [0,2,1,3,4])

        # correction:
        # Permute indices: [N, n_h, n_w, C, P, P]
        x = torch.permute(x, (0, 2, 4, 1, 3, 5))

        # Now shape is (N, n_h, n_w, C, P, P).

        # and rearrange them into a tensor of       #
        # shape (N, num_patches, patch_dim). Do not use a for-loop.                #
        # Instead, you may find torch.reshape and torch.permute helpful for this   #
        # step. 
        # x_rearranged = torch.reshape(x,(N, self.num_patches, self.patch_dim))  # aim to (N, num_patches, patch_dim)

        # correction: flatten:
        x_rearranged = torch.reshape(x,(N, self.num_patches, self.patch_dim))

        # where self.patch_dim = patch_size * patch_size * in_channels

        
        # Once the patches are flattened, embed them into latent vectors     #
        # using the projection layer.                                              #
        out = self.proj(x_rearranged)
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return out




class TransformerEncoderLayer(nn.Module):
    """
    A single layer of a Transformer encoder, to be used with TransformerEncoder.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Construct a TransformerEncoderLayer instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads.
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        Pass the inputs (and mask) through the encoder layer.

        Inputs:
        - src: the sequence to the encoder layer, of shape (N, S, D)
        - src_mask: the parts of the source sequence to mask, of shape (S, S)

        Returns:
        - out: the Transformer features, of shape (N, S, D)
        """
        ############################################################################
        # TODO: Implement the encoder layer by applying self-attention followed    #
        # by a feedforward block. This code will be very similar to decoder layer. #
        ############################################################################
        # Self-attention block (reference implementation)  copy from the "class TransformerDecoderLayer(nn.Module):"
        shortcut = src 
        src = self.self_attn(query=src, key=src, value=src, attn_mask=src_mask)
        src = self.dropout_self(src)
        src = src + shortcut
        src = self.norm_self(src)

        # no cross attention layer because nothing to cross, no target sequence avail yet.

        # feedforward block:  

        shortcut = src
        src = self.ffn(src)
        src = self.dropout_ffn(src)    
        src = src + shortcut    
        src = self.norm_ffn(src)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return src
