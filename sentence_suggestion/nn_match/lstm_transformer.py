import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self,
                 bert,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=55):

        super().__init__()

        self.bert = bert
        self.hid_dim = hid_dim

        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]).to(device))

        self.device = device

    def forward(self, keywords, keywords_mask):
        # keywords = [batch size, keywords len]
        # keywords_mask = [batch size, keywords len]

        batch_size = keywords.shape[0]
        keywords_len = keywords.shape[1]

        # get positional encodings
        pos = torch.arange(0, keywords_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos = [batch size, keywords len]
        # pos[1] = [0, 1, ..., keywords_len]

        # tokenize keywords by bert
        # make sure we don't update weights of pre-trained bert
        with torch.no_grad():
            embedded = self.bert(keywords)[0].to(self.device)
        # embedded = [batch_size, keywords_len, emb_dim]

        keywords = self.dropout((embedded * self.scale) + self.pos_embedding(pos))
        # keywords = [batch size, keyword len, hid dim]

        for layer in self.layers:
            keywords = layer(keywords, keywords_mask)
        # keyword = [batch size, keyword len, hid dim]

        return keywords

class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, keywords, keywords_mask):
        # keywords = [batch size, keywords len, hid dim]
        # keywords_mask = [batch size, keywords len]

        # self attention
        _keywords, _ = self.self_attention(keywords, keywords, keywords, keywords_mask)

        # dropout, residual connection and layer norm
        keywords = self.layer_norm(keywords + self.dropout(_keywords))

        # keywords = [batch size, keywords len, hid dim]

        # positionwise feedforward
        _keywords = self.positionwise_feedforward(keywords)

        # dropout, residual and layer norm
        keywords = self.layer_norm(keywords + self.dropout(_keywords))

        # keywords = [batch size, keywords len, hid dim]

        return keywords

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        # self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]).to(device))

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        # mask = [batch_size, 1, 1, keywords_len]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        # normalized weighted sum of query and key
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, seq len, seq len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # map energy into probability space
        attention = torch.softmax(energy, dim = -1)
        # attention = [batch size, n heads, query len, key len]

        # attention * values (V)
        x = torch.matmul(attention, V)
        # x = [batch size, n heads, seq len, head dim]

        # reshape x
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, seq len, n heads, head dim]

        # concatenate n heads into one big multi-head vector
        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch size, seq len, hid dim]

        # feedforward layer
        x = self.fc_o(x)
        # x = [batch size, seq len, hid dim]

        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)
        # x = [batch size, seq len, hid dim]

        return x

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, device):
        super().__init__()

        self.attn = nn.Linear(enc_hid_dim  + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

        self.device = device

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, keywords len, enc hid dim]
        # mask = [batch_size, 1, 1, keywords_len]

        batch_size = encoder_outputs.shape[0]
        keywords_len = encoder_outputs.shape[1]
        #repeat decoder hidden state keywords_len times
        hidden = hidden.unsqueeze(1).repeat(1, keywords_len, 1).to(self.device)
        # hidden = [batch size, keywords len, dec hid dim]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        # energy = [batch size, keywords len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch size, keywords len]

        mask = mask.squeeze(1).squeeze(1)
        # mask = [batch_size, keywords_len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
    def __init__(self,
                 bert,
                 emb_dim,
                 output_dim,
                 enc_hid_dim,
                 dec_hid_dim,
                 dropout,
                 device):
        super().__init__()

        self.bert = bert
        self.hid_dim = dec_hid_dim
        self.attention = Attention(enc_hid_dim, dec_hid_dim, device)

        self.rnn = nn.GRU(enc_hid_dim+emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear(enc_hid_dim+dec_hid_dim+emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.device = device

    def forward(self, input_token, dec_hidden, enc_outputs, keywords_mask):

        # input_token = [batch size]
        # dec_hidden  = [batch_size, dec hid dim]
        # enc_outputs = [batch_size, keywords_len, enc_hid_dim]
        # keywords_mask = [batch size, keywords len]

        input_token = input_token.unsqueeze(0)
        # input = [1, batch_size]

        with torch.no_grad():
            embedded = self.bert(input_token)[0].to(self.device)

        # embedded = [1, batch_size, emb_dim]

        a = self.attention(dec_hidden, enc_outputs, keywords_mask)
        # a = [batch_size, keywords_len]

        a = a.unsqueeze(1)
        # a = [batch_size, 1, keywords_len]

        weighted = torch.bmm(a, enc_outputs)
        # weighted = [batch_size, 1, enc_hid_dim]

        weighted = weighted.permute(1,0,2)
        # weighted = [1, batch_size, enc_hid_dim]

        rnn_input = torch.cat((weighted, embedded), dim=2)
        # rnn_input = [1, batch_size, enc_hid_dim+emb_dim]

        dec_hidden = dec_hidden.unsqueeze(0)
        # dec_hidden = [1, batch_size, dec_hid_dim]

        output, hidden = self.rnn(rnn_input, dec_hidden)
        # output = [1, batch_size, dec_hid_dim]
        # hidden = [1, batch_size, dec_hid_dim]

        embedded = embedded.squeeze(0)
        # embedded = [batch_size, emb_dim]
        output = output.squeeze(0)
        # output = [batch_size. dec_hidden_dim]
        weighted = weighted.squeeze(0)
        # weighted = [batch_size, enc_hid_dim]

        output = self.fc_out(torch.cat((output, weighted, embedded), dim=1) )
        # torch.cat((output, weighted, embedded), dim=1) = [batch_size, emb_dim+dec_hid_dim+enc_hid_dim]

        return output, hidden.squeeze(0)


class Seq2Seq(nn.Module):

    def __init__(self,
                 bert,
                 emb_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 output_dim,
                 dec_hid_dim,
                 dropout,
                 keywords_pad_idx,
                 device,
                 mask_idx):
        super().__init__()

        self.encoder = Encoder(
                bert=bert,
                hid_dim=emb_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                pf_dim=pf_dim,
                dropout=dropout,
                device=device
        )
        self.decoder = Decoder(
                bert=bert,
                emb_dim=emb_dim,
                output_dim=output_dim,
                enc_hid_dim=emb_dim,
                dec_hid_dim=dec_hid_dim,
                dropout=dropout,
                device=device
        )
        self.keywords_pad_idx = keywords_pad_idx
        self.mask_idx = mask_idx

        self.fc_out = nn.Linear(self.decoder.hid_dim, output_dim)
        self.device = device

    def create_mask(self, keywords):
        # keywords = [batch_size, keywords_len]
        _mask = (keywords != self.mask_idx)
        pad = (keywords != self.keywords_pad_idx)
        mask = (_mask & pad).unsqueeze(1).unsqueeze(2)
        # mask = [batch_size, 1, 1, keywords_len]
        return mask

    def forward(self, keywords, trg):
        # keywords = [batch size, keywords_len]
        # trg = [batch size, target length]

        keywords = keywords
        trg = trg

        batch_size = keywords.shape[0]
        keywords_len = keywords.shape[1]
        trg_len = trg.shape[1]

        mask = self.create_mask(keywords)
        # mask = [batch size, keywords len]
        # encoder_outputs is all hidden states of the keyword sequence
        encoder_outputs = self.encoder(keywords, mask)
        # encoder_outputs = [batch_size, keywords_len, emb_dim (bert)]

        # initialize hidden state as 0
        hidden = torch.zeros(batch_size, self.decoder.hid_dim).to(self.device)
        # hidden = [batch_size, dec_hid_dim]

        # first input to the decoder is the <sos> tokens
        input = trg[:,0]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state, all encoder hidden states and mask
            # receive output tensor (predictions) and new hidden state
            _, hidden = self.decoder(input, hidden, encoder_outputs, mask)
            # input is the next word in sentence
            input = trg[:,t]
            
        output = self.fc_out(hidden)

        return F.softmax(output, dim=1)
