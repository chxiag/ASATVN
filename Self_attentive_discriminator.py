import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import torch.nn.functional as F
def generate_original_PE(length: int, d_model: int) -> torch.Tensor:
    """
       Generate positional encoding as described in original paper.  :class:`torch.Tensor`
       :param length: a higher space L of the length of input data.
       :param d_model: Dimension of the model vector.
       :return: PE: Tensor of shape (L d_model).
       """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model))
    PE[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model))

    return PE
def generate_regular_PE(length: int, d_model: int, period: Optional[int] = 24) -> torch.Tensor:
    """
    Generate positional encoding with a given period.
     :param length: a higher space L of the length of input data, i.e. L.
     :param d_model: Dimension of the model vector.
     :param period: Size of the pattern to repeat. Default is 12.
     :return: PE: Tensor of shape (L, d_model).
     """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    PE = torch.sin(pos * 2 * np.pi / period)
    PE = PE.repeat((1, d_model))

    return PE
class MultiHeadAttentiondis(nn.Module):
    def __init__(self,
                 d_model: int,#70
                 q: int,#70
                 v: int,#70
                 h: int,#1
                 attention_size: int = None):
        """
        :param d_model: Dimension of the input vector.
        :param q: Dimension of all query matrix.
        :param v: Dimension of all value matrix.
        :param h: Number of heads.
        :param attention_size: Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
        """
        super().__init__()
        self._dropout = nn.Dropout(p=0.1)
        self._h = h
        self._attention_size = attention_size

        # Query, keys and value matrices 多头
        self._W_q = nn.Linear(d_model, q * self._h)  #70 70
        self._W_k = nn.Linear(d_model, q * self._h)  #70 70
        self._W_v = nn.Linear(d_model, v * self._h)  #70 70

        # Output linear function
        self._W_o = nn.Linear(self._h * v, d_model)  #70 70

        # Score placeholder
        self._scores = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        """Propagate forward the input through the MHB. (self-attention layer)
           We compute for each head the queries, keys and values matrices,
           followed by the Scaled Dot-Product. The result is concatenated
           and returned with shape (batch_size, L, d_model).

           :param query: Input tensor with shape (batch_size, L, d_model) used to compute queries.
           :param key: Input tensor with shape (batch_size, L, d_model) used to compute keys.
           :param value: Input tensor with shape (batch_size, L, d_model) used to compute values.
           :param mask: Mask to apply on scores before computing attention. One of ``'subsequent'``, None. Default is None.
           :return: attention: Self attention tensor with shape (batch_size, L, d_model).
           """
        K = query.shape[1]
        self._scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(K)
        self._scores = F.softmax(self._scores, dim=-1)
        self._scores= self._dropout(self._scores)
        attention = torch.bmm(self._scores, value)
        return attention

    @property
    def attention_map(self) -> torch.Tensor:
        if self._scores is None:
            raise RuntimeError(
                "Evaluate the model once to generate attention map")
        return self._scores
class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module """
    def __init__(self,
                 d_model: int,
                 d_ff: Optional[int] = 90):
        super().__init__()
        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear2(F.relu(self._linear1(x)))

class Encoderdis(nn.Module):
    """Encoderdis is made up of self-attn layer and feed forward (defined below)"""
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk'):
        super().__init__()
        MHAdis = MultiHeadAttentiondis
        self._selfAttentiondis = MHAdis(d_model, q, v, h, attention_size=attention_size)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate the input through the Encoder layer.
        Apply the Multi Head Attention self-attention ayer, add residual and normalize.
        Apply the Point-wise Feed Forward block, add residual and normalize.

        :param x: Input tensor with shape (batch_size, L, d_model).
        :return: x: Output tensor with shape (batch_size, L, d_model).
        """
        residual = x
        x = self._selfAttentiondis(query=x, key=x, value=x)
        x = self._dopout(x)
        x = self._layerNorm1(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dopout(x)
        x = self._layerNorm2(x + residual)

        return x
    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map
class Self_attentive_discriminator(nn.Module):
    """
         class for self-attentive discriminator
          :param d_input: Model input dimension.
          :param d_model:Dimension of the input vector.
          :param hidden_dim: Dimension of the hidden units
          :param d_output:Model output dimension.
          :param q: Dimension of queries and keys.
          :param v: Dimension of values.
          :param h: Number of heads.
          :param M: Number of encoder layers to stack
          :parama attention_size: Number of backward elements to apply attention
          :parama dropout:Dropout probability after each MHA or PFF block.Default is ``0.3``
          :parama chunk_mode:Swict between different MultiHeadAttention blocks.One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``
          :parama pe:Type of positional encoding to add.Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
          :parama: is_discriminator: Determine whether the Transformer is a discriminator
          """
    def __init__(self, d_input: int, d_model: int, hidden_dim: int, d_output: int, q: int, v: int, h: int, M: int,
                 attention_size: int = None, dropout: float = 0.3, chunk_mode: bool = True, pe: str = 'original',is_discriminator:bool=None):
        super().__init__()
        self.is_discriminator = is_discriminator
        self.input_layer1=nn.Linear(12,hidden_dim)
        self.input_layer2 = nn.Linear(36, hidden_dim)
        self._d_model = d_model
        self.layers_encoding = nn.ModuleList(
            [Encoderdis(d_model, q, v, h, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode) for _ in
             range(M)])
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }
        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.name = 'transformer'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: class:`torch.Tensor` of shape (batch_size, x.size(1)).
        :return: out: Output tensor with shape (batch_size).
        """
        if self.is_discriminator and x.size(1) == 12:
            x = self.input_layer1(x)
        if self.is_discriminator and x.size(1) == 36:
            x = self.input_layer2(x)
        x = torch.unsqueeze(x, 2)
        K = x.shape[1]
        encoding = self._embedding(x)
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)
        for layer in self.layers_encoding:
            encoding = layer(encoding)
        output = self._linear(encoding)
        output = torch.sigmoid(output)
        num_dims = len(output.shape)
        reduction_dims = tuple(range(1, num_dims))
        out = torch.mean(output, dim=reduction_dims)
        return out