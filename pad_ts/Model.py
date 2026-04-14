import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L27
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LearnablePositionalEncoding(nn.Module):
    """
    https://github.com/Y-debug-sys/Diffusion-TS/blob/13a2186e6442669f70afe07dcd3632466f6ee10a/Models/interpretable_diffusion/model_utils.py#L66
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(1, max_len, d_model)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        # print(x.shape)
        x = x + self.pe
        return self.dropout(x)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L101C7-L101C15
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Vanilla transformer encoder block.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    
class TransformerWithGRUEncoderBlock(nn.Module):
    """
    Modified transformer encoder block with bidirectional GRU.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, gru_hidden_size=None, **block_kwargs):
        super().__init__()
        
        # Optional: Default GRU hidden size if not provided
        self.gru_hidden_size = gru_hidden_size if gru_hidden_size else hidden_size

        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # GRU Layer (Bidirectional)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=self.gru_hidden_size, 
                          num_layers=1, bidirectional=True, batch_first=True)
        
        # Second LayerNorm and MLP (unchanged from original transformer)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

        print("TransformerWithGRUEncoderBlock")

    def forward(self, x):
        # GRU Layer (Bidirectional)
        x_gru, _ = self.gru(self.norm1(x))  # x_gru is output from the GRU
        
        # Keep only the hidden state outputs from the GRU
        # GRU is bidirectional, so we concatenate the forward and backward hidden states
        x_gru = x_gru[:, :, :self.gru_hidden_size] + x_gru[:, :, self.gru_hidden_size:]

        # Apply MLP (unchanged from transformer block)
        x = x + self.mlp(self.norm2(x_gru))
        
        return x


class DiTBlockWithGRU(nn.Module):
    """
    A DiT block with BiGRU and AdaLN-Zero conditioning.
    Replaces Attention with BiGRU.
    """

    def __init__(self, hidden_size, mlp_ratio=4.0, gru_hidden_size=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Replace Attention with BiGRU
        self.gru = nn.GRU(
            input_size=hidden_size, 
            hidden_size=gru_hidden_size if gru_hidden_size else hidden_size, 
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        print("DiTBlockWithGRU initialized")

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # Modulate and apply BiGRU
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x, _ = self.gru(x)  # BiGRU output
        x = x[:, :, :x.size(2) // 2] + x[:, :, x.size(2) // 2:]  # Combine bidirectional outputs
        x = x * gate_msa.unsqueeze(1)  # Apply gating for BiGRU output

        # Apply MLP and modulate
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x



class Encoder2(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, n_layers=3, mlp_ratio=4.0):
        super().__init__()
        self.encoder_blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        for index in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[index](x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, n_layers=3, mlp_ratio=4.0, gru_hidden_size=None):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerWithGRUEncoderBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    gru_hidden_size=gru_hidden_size
                )
                for _ in range(n_layers)
            ]
        )
        print("TransformerWithGRUEncoderBlock")

    def forward(self, x):
        # Sequentially apply each GRU-enhanced transformer block
        for block in self.encoder_blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    """
    Decoder that uses BiGRU instead of Attention layers, with AdaLN modulation.
    """

    def __init__(self, hidden_size=512, num_heads=8, n_layers=3, mlp_ratio=4.0, gru_hidden_size=None):
        super().__init__()
        self.encoder_blocks = nn.Sequential(
            *[
                DiTBlockWithGRU(
                    hidden_size=hidden_size, mlp_ratio=mlp_ratio, gru_hidden_size=gru_hidden_size
                )
                for _ in range(n_layers)
            ]
        )
        self.diffusion_step_emb = TimestepEmbedder(hidden_size)

    def forward(self, x, t):
        identity = x
        toreturn = torch.zeros_like(x)
        c = self.diffusion_step_emb(t)
        for index in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[index](x, c)
            toreturn += x
            x += identity
            identity = x
        return toreturn


class Decoder2(nn.Module):
    """
    Note: Even though it is called a decoder. Each DiT blocks belongs to the transformer encoder families.
    """

    def __init__(self, hidden_size=512, num_heads=8, n_layers=3, mlp_ratio=4.0):
        super().__init__()
        self.encoder_blocks = nn.Sequential(
            *[
                DiTBlock(
                    hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(n_layers)
            ]
        )
        self.diffusion_step_emb = TimestepEmbedder(hidden_size)

    def forward(self, x, t):
        identity = x
        toreturn = torch.zeros_like(x)
        c = self.diffusion_step_emb(t)
        for index in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[index](x, c)
            toreturn += x
            x += identity
            identity = x
        return toreturn


class TimeSeries2EmbLinear(nn.Module):
    """
    Encode time series data alone with selected dimension.
    """

    def __init__(
        self,
        hidden_size=512,
        feature_last=True,
        shape=(24, 6),
        dim2emb="time",
        dropout=0,
    ):
        super().__init__()
        assert dim2emb in ["time", "feature"], "Please indicate which dim to emb"
        if feature_last:
            sequence_length, feature_size = shape
        else:
            feature_size, sequence_length = shape

        self.feature_last = feature_last
        self.dim2emb = dim2emb
        self.pos_emb = LearnablePositionalEncoding(
            d_model=hidden_size, max_len=sequence_length
        )
        if dim2emb == "time":
            self.processing = nn.Sequential(
                nn.Linear(feature_size, hidden_size), nn.Dropout(dropout)
            )
        else:
            self.processing = nn.Sequential(
                nn.Linear(sequence_length, hidden_size), nn.Dropout(dropout)
            )

    def forward(self, x):
        if not self.feature_last:
            x = x.permute(0, 2, 1)

        if self.dim2emb == "time":
            x = self.processing(x)
            return self.pos_emb(x)
        return self.processing(x.permute(0, 2, 1))


class PaD_TS(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_heads=4,
        n_encoder=2,
        n_decoder=2,
        feature_last=True,
        mlp_ratio=4.0,
        dropout=0,
        input_shape=(24, 6),
        use_gru=True,
    ):
        super().__init__()
        self.use_gru = use_gru
        self.time2emb = TimeSeries2EmbLinear(
            hidden_size=hidden_size,
            feature_last=feature_last,
            shape=input_shape, # (24,9)
            dim2emb="time",
            dropout=dropout,
        )
        self.feature2emb = TimeSeries2EmbLinear(
            hidden_size=hidden_size,
            feature_last=feature_last,
            shape=input_shape, # (24,9)
            dim2emb="feature",
            dropout=dropout,
        )

        if self.use_gru:
            self.time_encoder = Encoder(
                hidden_size=hidden_size, #256
                num_heads=num_heads, # 4
                n_layers=n_encoder, # 1
                mlp_ratio=mlp_ratio,
            )
            self.feature_encoder = Encoder(
                hidden_size=hidden_size,
                num_heads=num_heads,
                n_layers=n_encoder,
                mlp_ratio=mlp_ratio,
            )
            self.time_blocks = Decoder(
                hidden_size=hidden_size,
                num_heads=num_heads,
                n_layers=n_decoder,
                mlp_ratio=mlp_ratio,
            )
            self.feature_blocks = Decoder(
                hidden_size=hidden_size,
                num_heads=num_heads,
                n_layers=n_decoder,
                mlp_ratio=mlp_ratio,
            )
        else:
            self.time_encoder = Encoder2(
                hidden_size=hidden_size,
                num_heads=num_heads,
                n_layers=n_encoder,
                mlp_ratio=mlp_ratio,
            )
            self.feature_encoder = Encoder2(
                hidden_size=hidden_size,
                num_heads=num_heads,
                n_layers=n_encoder,
                mlp_ratio=mlp_ratio,
            )
            self.time_blocks = Decoder2(
                hidden_size=hidden_size,
                num_heads=num_heads,
                n_layers=n_decoder,
                mlp_ratio=mlp_ratio,
            )
            self.feature_blocks = Decoder2(
                hidden_size=hidden_size,
                num_heads=num_heads,
                n_layers=n_decoder,
                mlp_ratio=mlp_ratio,
            )

        self.fc_time = nn.Linear(hidden_size, input_shape[1])
        self.fc_feature = nn.Linear(hidden_size, input_shape[0])

    def forward(self, x, t):
        x_time = self.time2emb(x)
        x_time = self.time_encoder(x_time)
        x_time = self.time_blocks(x_time, t)
        x_time = self.fc_time(x_time)

        x_feature = self.feature2emb(x)
        x_feature = self.feature_encoder(x_feature)
        x_feature = self.feature_blocks(x_feature, t)
        x_feature = self.fc_feature(x_feature)
        return x_feature.permute(0, 2, 1) + x_time