# model.py
import torch
import torch.nn as nn

# ================ Encoder: 2-layer BiLSTM ================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim=256, hid_dim=512, n_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embed   = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm    = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                               bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.hid_dim = hid_dim
        self.n_layers = n_layers

    def forward(self, src):
        x = self.dropout(self.embed(src))
        outputs, (h, c) = self.lstm(x)
        return outputs, h, c


# ================ Bridge: BiLSTM → 4-layer decoder ================
class Bridge(nn.Module):
    def __init__(self, enc_layers=2, dec_layers=4, hid_dim=512):
        super().__init__()
        self.h_proj = nn.Linear(2*hid_dim, hid_dim)
        self.c_proj = nn.Linear(2*hid_dim, hid_dim)
        self.dec_layers = dec_layers

    def forward(self, h, c):
        n2, B, H = h.size()
        enc_layers = n2 // 2
        h = h.view(enc_layers, 2, B, H)
        c = c.view(enc_layers, 2, B, H)
        h = torch.cat([h[:,0], h[:,1]], dim=-1)
        c = torch.cat([c[:,0], c[:,1]], dim=-1)
        h = torch.tanh(self.h_proj(h))
        c = torch.tanh(self.c_proj(c))
        reps = self.dec_layers // enc_layers
        rem  = self.dec_layers %  enc_layers
        h = h.repeat_interleave(reps, dim=0)
        c = c.repeat_interleave(reps, dim=0)
        if rem:
            h = torch.cat([h, h[-1:].repeat(rem,1,1)], dim=0)
            c = torch.cat([c, c[-1:].repeat(rem,1,1)], dim=0)
        return h, c


# ================ Luong Attention ================
class LuongAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.Wa = nn.Linear(hid_dim, hid_dim, bias=False)

    def forward(self, dec_h, enc_outs, enc_mask):
        proj = self.Wa(enc_outs)
        scores = torch.bmm(proj, dec_h.unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(enc_mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), enc_outs).squeeze(1)
        return ctx, attn


# ================ Decoder with Attention ================
class AttnDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim=256, hid_dim=512, n_layers=4, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embed   = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.lstm    = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.attn    = LuongAttention(hid_dim)
        self.concat  = nn.Linear(hid_dim*2, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.hid_dim  = hid_dim
        self.output_dim = output_dim

        # weight tying
        if hid_dim == emb_dim:
            self.fc_out = nn.Linear(hid_dim, output_dim, bias=False)
            self.fc_out.weight = self.embed.weight
            self.proj = None
        else:
            self.proj  = nn.Linear(hid_dim, emb_dim, bias=False)
            self.fc_out = nn.Linear(emb_dim, output_dim, bias=False)
            self.fc_out.weight = self.embed.weight

    def forward(self, input_t, hidden, cell, enc_outs, enc_mask):
        emb = self.dropout(self.embed(input_t.unsqueeze(1)))
        lstm_out, (hidden, cell) = self.lstm(emb, (hidden, cell))
        h_t = lstm_out.squeeze(1)
        ctx, _ = self.attn(h_t, enc_outs, enc_mask)
        cat = torch.tanh(self.concat(torch.cat([h_t, ctx], dim=-1)))
        if self.proj is not None:
            cat = self.proj(cat)
        logits = self.fc_out(cat)
        return logits, hidden, cell


# ================ Seq2Seq wrapper ================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx=0, bos_idx=1, eos_idx=2, device=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx, self.bos_idx, self.eos_idx = pad_idx, bos_idx, eos_idx
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bridge = Bridge(enc_layers=encoder.n_layers, dec_layers=decoder.n_layers, hid_dim=encoder.hid_dim)

    def make_src_mask(self, src):
        return (src != self.pad_idx).long()

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5, max_len=None):
        B = src.size(0)
        T_out = (trg.size(1) if trg is not None else max_len)
        assert T_out is not None, "Provide trg or max_len"

        enc_outs2H, h, c = self.encoder(src)
        Bsz, Tsrc, H2 = enc_outs2H.size()
        H = H2 // 2
        enc_outs = enc_outs2H.view(Bsz, Tsrc, 2, H).sum(dim=2)

        h, c = self.bridge(h, c)
        src_mask = self.make_src_mask(src)

        V = self.decoder.output_dim
        outputs = torch.zeros(B, T_out, V, device=self.device, dtype=torch.float)

        input_t = (trg[:, 0] if trg is not None else
                   torch.full((B,), self.bos_idx, dtype=torch.long, device=self.device))

        hidden, cell = h, c
        for t in range(1, T_out):
            logits, hidden, cell = self.decoder(input_t, hidden, cell, enc_outs, src_mask)
            outputs[:, t, :] = logits
            use_tf = (trg is not None) and (torch.rand(1).item() < teacher_forcing_ratio)
            input_t = trg[:, t] if use_tf else logits.argmax(dim=-1)
        return outputs
