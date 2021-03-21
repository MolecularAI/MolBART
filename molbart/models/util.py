import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
# from onmt.modules import MultiHeadedAttention


class FuncLR(LambdaLR):
    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self attention block
        att = self.norm1(src)
        att = self.self_attn(att, att, att, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        att = src + self.dropout1(att)

        # Feedforward block
        out = self.norm2(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout2(out)
        return out


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None
    ):
        # Self attention block 
        query = self.norm1(tgt)
        query = self.self_attn(query, query, query, attn_mask=tgt_mask, 
                key_padding_mask=tgt_key_padding_mask)[0]
        query = tgt + self.dropout1(query)

        # Context attention block
        att = self.norm2(query)
        att = self.multihead_attn(att, memory, memory, attn_mask=memory_mask, 
                key_padding_mask=memory_key_padding_mask)[0]
        att = query + self.dropout2(att)

        # Feedforward block
        out = self.norm3(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout3(out)
        return out


# class MaskedEncoderLayer(nn.TransformerEncoderLayer):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
#         # Copied from pytorch implementation, but use OpenNMT MultiHeadedAttention

#         super().__init__(
#             d_model,
#             nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation=activation
#         )
#         self.self_attn = MultiHeadedAttention(nhead, d_model, dropout=dropout)

#     def forward(self, src, src_mask=None, src_key_padding_mask=None):
#         """ Pass input through encoder layer

#         Args:
#             src (torch.Tensor): Src tensor of shape (seq_len, batch_size, d_model)
#             src_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len, seq_len)
#             src_key_padding_mask (torch.Tensor): Padding mask of shape (batch_size, seq_len)

#         Returns:
#             (torch.Tensor): Layer output of shape (seq_len_ batch_size, d_model)
#         """

#         pad_mask = src_key_padding_mask.transpose(0, 1).unsqueeze(1)
#         att_mask = torch.gt(src_mask + pad_mask, 0)

#         # Self attention block
#         att = self.norm1(src)
#         att = att.transpose(0, 1)
#         att = self.self_attn(att, att, att, mask=att_mask, attn_type="self")[0]
#         att = att.transpose(0, 1)
#         att = src + self.dropout1(att)

#         # Feedforward block
#         out = self.norm2(att)
#         out = self.linear2(self.dropout(self.activation(self.linear1(out))))
#         out = att + self.dropout2(out)
#         return out
