import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import math
class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        self.head = nhead
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                #nn.init.orthogonal_(p)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        self.head= nhead
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                #nn.init.orthogonal_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    #     self._reset_parameters()
    
    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             # nn.init.xavier_uniform_(p)
    #             nn.init.orthogonal_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
    #     self._reset_parameters()

    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             # nn.init.xavier_uniform_(p)
    #             nn.init.orthogonal_(p)
                
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Attention(nn.Module):
    def __init__(self, head_dim: int, dim: int, n_heads: int, drop_out: float=0.0):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dim = dim
        self.drop_out = drop_out
        
        self.wq = nn.Linear(self.dim, self.dim, bias=False)
        self.wk = nn.Linear(self.dim, self.dim, bias=False)
        self.wv = nn.Linear(self.dim, self.dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

        self.attn_dropout = nn.Dropout(self.drop_out)
        self.resid_dropout = nn.Dropout(self.drop_out)


        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        assert self.flash, "Flash Attention requires PyTorch >= 2.0"
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                #nn.init.orthogonal_(p)    

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x + pos), self.wk(x + pos), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_heads, self.head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scale = 1 / math.sqrt(xq.size(-1))
        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq*scale, xk, xv, scale=1.0,attn_mask=mask, dropout_p=self.drop_out if self.training else 0.0, is_causal=False)#is_causal=True if mask is None else False)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class CAttention(nn.Module):
    def __init__(self, head_dim: int, dim: int, n_heads: int, drop_out: float=0.0):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dim = dim
        self.drop_out = drop_out
        
        self.wq = nn.Linear(self.dim, self.dim, bias=False)
        self.wk = nn.Linear(self.dim, self.dim, bias=False)
        self.wv = nn.Linear(self.dim, self.dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

        self.attn_dropout = nn.Dropout(self.drop_out)
        self.resid_dropout = nn.Dropout(self.drop_out)


        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        assert self.flash, "Flash Attention requires PyTorch >= 2.0"
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                #nn.init.orthogonal_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,

        qpos:  Optional[torch.Tensor] = None,
        kvpos:  Optional[torch.Tensor] = None,

        mask: Optional[torch.Tensor] = None,
    ):
        #print(q.shape)
        bsz, qseq_len, _ = q.shape
        bsz, kvseq_len, _ = kv.shape


        # QKV

        xq, xk, xv = self.wq(self.with_pos_embed(q,qpos)), self.wk(self.with_pos_embed(kv,kvpos)), self.wv(kv)
        xq = xq.view(bsz, qseq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, kvseq_len, self.n_heads, self.head_dim)
        xv = xv.view(bsz, kvseq_len, self.n_heads, self.head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scale = 1 / math.sqrt(xq.size(-1))

        ###################################################################
        
        L, S = xq.size(-2), xk.size(-2)
        # if S >73:
        #     attn_weight=xq*scale @ xk.transpose(-2, -1) * scale
        #     attn_bias = torch.zeros(8,1,L, S, dtype=xq.dtype, device=xq.device)
        #     attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
        #     attn_weight += attn_bias
        #     attn_weight = torch.softmax(attn_weight, dim=-1)
        #     a = attn_weight @ xv
        #     print(a[0])

        #     import numpy as np
        #     np.save('./att.npy',attn_weight.detach().cpu().numpy())
        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq*scale, xk, xv, scale=1.0,attn_mask=mask, dropout_p=self.drop_out if self.training else 0.0, is_causal=False)
        # print(output[0])
        output = output.transpose(1, 2).contiguous().view(bsz, qseq_len, -1)
        
        # assert 1==2
        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class CAttention_pure(nn.Module):
    def __init__(self, head_dim: int, dim: int, n_heads: int, drop_out: float=0.0):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dim = dim
        self.drop_out = drop_out
        
        self.wq = nn.Linear(self.dim, self.dim, bias=False)
        self.wk = nn.Linear(self.dim, self.dim, bias=False)
        #self.wv = nn.Linear(self.dim, self.dim, bias=False)
        #self.wo = nn.Linear(self.dim, self.dim, bias=False)

        self.attn_dropout = nn.Dropout(self.drop_out)
        self.resid_dropout = nn.Dropout(self.drop_out)


        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        assert self.flash, "Flash Attention requires PyTorch >= 2.0"
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                #nn.init.orthogonal_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,

        qpos:  Optional[torch.Tensor] = None,
        kvpos:  Optional[torch.Tensor] = None,

        mask: Optional[torch.Tensor] = None,
    ):
        #print(q.shape)
        bsz, qseq_len, _ = q.shape
        bsz, kvseq_len, _ = kv.shape


        # QKV

        xq, xk, xv = self.wq(self.with_pos_embed(q,qpos)), self.wk(self.with_pos_embed(kv,kvpos)), kv
        xq = xq.view(bsz, qseq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, kvseq_len, self.n_heads, self.head_dim)
        xv = xv.view(bsz, kvseq_len, self.n_heads, self.head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scale = 1 / math.sqrt(xq.size(-1))

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq*scale, xk, xv, scale=1.0,attn_mask=mask, dropout_p=self.drop_out if self.training else 0.0, is_causal=False)
        output = output.transpose(1, 2).contiguous().view(bsz, qseq_len, -1)

        # final projection into the residual stream
        #output = self.wo(output)
        output = self.resid_dropout(output)
        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, norm_eps: float, drop_out: float=0.0):
        super().__init__()
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        self.attention = Attention(
            head_dim=self.head_dim,
            dim=dim,
            n_heads=n_heads,
            drop_out=drop_out)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            drop_out=drop_out,
        )
        self.layer_id = layer_id
        #self.attention_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, xy_pos: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = x + self.attention(x, xy_pos, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class CATransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, norm_eps: float, drop_out: float=0.0):
        super().__init__()
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        self.attention = CAttention(
            head_dim=self.head_dim,
            dim=dim,
            n_heads=n_heads,
            drop_out=drop_out)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            drop_out=drop_out,
        )
        self.layer_id = layer_id
        #self.q_norm = nn.LayerNorm(dim)
        #self.kv_norm = nn.LayerNorm(dim)

        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor,kv: torch.Tensor, q_pos: Optional[torch.Tensor] = None ,kv_pos: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        h = q + self.attention(q,kv, q_pos,kv_pos, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    

class CATransformerBlock_noffn(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, norm_eps: float, drop_out: float=0.0):
        super().__init__()
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        self.attention = CAttention_pure(
            head_dim=self.head_dim,
            dim=dim,
            n_heads=n_heads,
            drop_out=drop_out)

        self.layer_id = layer_id
        #self.q_norm = nn.LayerNorm(dim)
        #self.kv_norm = nn.LayerNorm(dim)

        # self.ffn_norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(-5.0))

    def forward(self, q: torch.Tensor,kv: torch.Tensor, q_pos: Optional[torch.Tensor] = None ,kv_pos: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        h = q + self.attention(q,kv, q_pos,kv_pos, mask)*torch.sigmoid(self.scale)
        
        return h
    
class CATransformerBlockTest(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, norm_eps: float, drop_out: float=0.0):
        super().__init__()
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        self.attention = CAttention(
            head_dim=self.head_dim,
            dim=dim,
            n_heads=n_heads,
            drop_out=drop_out)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            drop_out=drop_out,
        )
        self.layer_id = layer_id
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)

        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor,kv: torch.Tensor, q_pos: Optional[torch.Tensor] = None ,kv_pos: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        h = q + self.attention(self.q_norm(q),self.kv_norm(kv), q_pos,kv_pos, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop_out: float=0.0):
        super().__init__()
        self.linear_1 = nn.Linear(dim, hidden_dim, bias=False)
        self.linear_2 = nn.Linear(hidden_dim, dim, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, resolution=None, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        # self.normalize = normalize
        self.scale = scale
        # if self.normalize is True and self.scale is None:
        #     raise ValueError("Scale should be NOT NONE when normalize is True.")
        # if self.scale is not None and self.normalize is False:
        #     raise ValueError("Normalize should be True when scale is not None.")
        self.avgpool = torch.nn.AdaptiveAvgPool2d(resolution)

    def forward(self, x, y) -> torch.Tensor:
        # if self.normalize:
        #     eps = 1e-6
        #     y = y / (img_h + eps) * self.scale
        #     x = x / (img_w + eps) * self.scale
        #bf h w
        # print(x.shape)
        x = self.avgpool(x.unsqueeze(1)).squeeze(1).flatten(1,2) #bf,q
        y = self.avgpool(y.unsqueeze(1)).squeeze(1).flatten(1,2) #bf,q
        # print(x.shape)
        y = y * self.scale
        x = x * self.scale
        dim_i = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_i = self.temperature ** (2 * (torch.div(dim_i, 2, rounding_mode="trunc")) / self.num_pos_feats)
        # print(dim_i.shape)
        pos_x = x[:,:, None] / dim_i[None,:]
        pos_y = y[:,:, None] / dim_i[None,:]
        # print(pos_x.shape)
        pos_x = torch.stack((pos_x[:,:, 0::2].sin(), pos_x[:,:, 1::2].cos()), dim=2).flatten(2)
        pos_y = torch.stack((pos_y[:,:, 0::2].sin(), pos_y[:,:, 1::2].cos()), dim=2).flatten(2)
        pos_embed = torch.cat((pos_y, pos_x), dim=2)
        return pos_embed

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, num_pos_feats, temperature=10000):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x):
        dim_i = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_i = self.temperature ** (2 * (torch.div(dim_i, 2, rounding_mode="trunc")) / self.num_pos_feats)

        pos_x = x[:, None] / dim_i
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)

        return pos_x
    
def build_xy_pe(dim,res):
    assert dim % 2 == 0, f"Hidden dim should be 2x, but get {dim}."
    num_pos_feats = dim / 2
    return PositionEmbeddingSine(num_pos_feats=num_pos_feats, resolution=res, scale=2*math.pi)

def build_frame_pe(config: dict):
    return SinusoidalPositionalEmbedding(num_pos_feats=config["DIM"], temperature=10000)