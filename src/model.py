import logging
import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)


class UniHPF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.input2emb_model = DescEmb(args)
        if args.eventencoder == "transformer":
            self.eventencoder_model = TransformerEventEncoder(args)
        elif args.eventencoder == "rnn":
            self.eventencoder_model = RNNEventEncoder(args)
        else:
            raise NotImplementedError
        self.pred_model = Aggregator(args)
        self.emb2out_model = PredOutPutLayer(args)

    def get_logits(self, net_output):
        return net_output.float()

    def get_targets(self, sample):
        return sample["label"].float()

    def forward(self, **kwargs):
        all_codes_embs = self.input2emb_model(**kwargs)  # (B, S, E)
        events = self.eventencoder_model(all_codes_embs, **kwargs)
        x = self.pred_model(events, **kwargs)
        net_output = self.emb2out_model(x, **kwargs)

        return net_output


class DescEmb(nn.Module):
    def __init__(self, args, embed_dim=None):
        super().__init__()

        self.args = args

        self.input_index_size = 28119  # 28996 # bio clinical bert vocab
        self.type_index_size = 14  # mimic3 + eicu + mimic4
        self.dpe_index_size = 25

        self.dpe = args.dpe
        self.token_type = args.type_token
        self.pos_enc = args.pos_enc

        if embed_dim:
            self.args.embed_dim = embed_dim

        self.input_ids_embedding = nn.Embedding(
            self.input_index_size, self.args.embed_dim, padding_idx=0
        )

        self.type_ids_embedding = (
            nn.Embedding(self.type_index_size, self.args.embed_dim, padding_idx=0)
            if self.args.type_token
            else None
        )

        self.dpe_ids_embedding = (
            nn.Embedding(self.dpe_index_size, self.args.embed_dim, padding_idx=0)
            if self.args.dpe
            else None
        )

        max_len = args.max_word_len

        self.pos_encoder = (
            PositionalEncoding(args.embed_dim, args.dropout, max_len)
            if self.pos_enc
            else None
        )

        self.layer_norm = nn.LayerNorm(args.embed_dim, eps=1e-12)

    def forward(self, input_ids, type_ids, dpe_ids, **kwargs):
        B, S = input_ids.shape[0], input_ids.shape[1]

        x = self.input_ids_embedding(input_ids)

        if self.type_ids_embedding:  # column description mean
            x += self.type_ids_embedding(type_ids)

        if self.dpe_ids_embedding:
            x += self.dpe_ids_embedding(dpe_ids)

        x = x.view(B * S, -1, self.args.embed_dim)

        if self.pos_encoder:
            x = self.pos_encoder(x)  # (B, S, W, E) -> (B*S, W, E)
        x = self.layer_norm(x)
        return x


class RNNEventEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.pred_dim
        self.pred_dim = args.pred_dim

        self.model = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.pred_dim,
            num_layers=args.n_layers // 2,
            dropout=args.dropout,
            batch_first=True,
            bidirectional=True,
        )

        self.max_word_len = args.max_word_len

        self.post_encode_proj = nn.Linear(self.hidden_dim * 2, self.pred_dim)

    def forward(self, all_codes_embs, input_ids, **kwargs):
        B, S, _ = input_ids.size()

        lengths = torch.argmin(input_ids.view(B * S, -1), dim=1)
        lengths = torch.where(lengths > 0, lengths, 1).detach().cpu()

        packed = pack_padded_sequence(
            all_codes_embs, lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = self.model(packed)
        output_seq, _ = pad_packed_sequence(output, batch_first=True, padding_value=0)

        forward_output = output_seq[:, -1, : self.hidden_dim]
        backward_output = output_seq[:, 0, self.hidden_dim :]
        net_output = torch.cat((forward_output, backward_output), dim=-1)

        net_output = self.post_encode_proj(net_output).view(B, -1, self.pred_dim)

        return net_output


class TransformerEventEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_dim = args.pred_dim

        encoder_layers = TransformerEncoderLayer(
            args.pred_dim,
            args.n_heads,
            args.pred_dim * 4,
            args.dropout,
            batch_first=True,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, args.n_layers // 2
        )

        self.post_encode_proj = nn.Linear(args.embed_dim, self.pred_dim)

    def forward(self, all_codes_embs, input_ids, **kwargs):

        B, S, _ = input_ids.size()
        src_pad_mask = (
            input_ids.view(B * S, -1).eq(0).to(all_codes_embs.device)
        )  # (B, S, W) -> (B*S, W)
        encoder_output = self.transformer_encoder(
            all_codes_embs, src_key_padding_mask=src_pad_mask
        )
        net_output = self.post_encode_proj(encoder_output[:, 0, :]).view(
            B, -1, self.pred_dim
        )

        return net_output


class Aggregator(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        encoder_layers = TransformerEncoderLayer(
            args.pred_dim,
            args.n_heads,
            args.pred_dim * 4,
            args.dropout,
            batch_first=True,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, args.n_layers // 2
        )
        self.pos_encoder = PositionalEncoding(
            args.pred_dim, args.dropout, args.max_seq_len
        )

        self.layer_norm = nn.LayerNorm(args.embed_dim, eps=1e-12)

    def forward(self, events, input_ids, **kwargs):
        # input_ids: (B, S) (B x S, W ) -> (Bx s, W) -> (B, s, W)

        B, S = input_ids.shape[0], input_ids.shape[1]
        src_pad_mask = input_ids[:, :, 1].eq(0).to(events.device)

        src_mask = None

        """
        Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked positions. 
                    If a ByteTensor is provided, 
                        the non-zero positions are not allowed to attend
                        while the zero positions will be unchanged. 

                    If a BoolTensor is provided, 
                        positions with ``True`` are not allowed to attend 
                        while ``False`` values will be unchanged. 

                    If a FloatTensor is provided, it will be added to the attention weight. 
                    https://pytorch.org/docs/sfeature/generated/torch.nn.Transformer.html

        """
        if self.pos_encoder is not None:
            events = self.layer_norm(self.pos_encoder(events))
        encoder_output = self.transformer_encoder(
            events, mask=src_mask, src_key_padding_mask=src_pad_mask
        )

        return encoder_output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """

        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PredOutPutLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.final_proj = nn.Linear(
            args.pred_dim,
            18 if args.pred_target == "dx" else 1,
        )

    def forward(self, x, input_ids, **kwargs):
        B, S = input_ids.size(0), input_ids.size(1)
        if self.args.pred_pooling == "cls":
            x = x[:, 0, :]
        elif self.args.pred_pooling == "mean":
            mask = ~input_ids[:, :, 1].eq(102)

            mask = mask.unsqueeze(dim=2).to(x.device).expand(B, S, self.args.pred_dim)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        output = self.final_proj(x)  # B, E -> B, 1
        output = output.squeeze()
        return {"pred_output": output}
