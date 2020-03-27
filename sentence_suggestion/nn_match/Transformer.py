import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hid_dim,
                 n_heads,
                 n_layers,
                 dropout_rate,
                 device,
                 pad_idx):

        super().__init__()
        # specify the device (gpu or cpu)
        self.device = device
        # word embeddings
        self.token_embedding = nn.Embedding(input_dim, hid_dim).to(device)
        # transformer layers
        self.layers = nn.ModuleList(
                          [TransformerLayer(hid_dim, n_heads,
                                            dropout_rate, device)
                                     for _ in range(n_layers)])
        # Linear layer (this is where prediction happening)
        self.fc = nn.Linear(hid_dim, output_dim).to(device)

        self.dropout = nn.Dropout(dropout_rate).to(device)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.pad_idx = pad_idx

    def make_mask(self, keywords):
        # keywords = [batch size, keywords len]

        # keywords_mask = [batch size, 1, 1, keywords len]
        # if index of a token is equal to index of <pad>, the value will be 0
        # it's unsqueezed twice to be applied to multihead later, i.e. [batch size, n heads, src len, src len]
        keywords_mask = (keywords != self.pad_idx).unsqueeze(1).unsqueeze(2)

        return keywords_mask


    def forward(self, keywords):

        # keywords = [batch size, keywords len]

        # keywords_mask = [batch size, keywords len]
        keywords_mask = self.make_mask(keywords)

        batch_size = keywords.shape[0]
        keywords_len = keywords.shape[1]

        # keywords = [batch size, keyword len, hid dim]
        keywords = self.dropout((self.token_embedding(keywords) * self.scale))

        # iterate through all layers in encoder
        for layer in self.layers:
            # keywords = [batch size, keyword len, hid dim]
            keywords = layer(keywords, keywords_mask)

        # make predictions based on encoder output
        outputs = self.fc(keywords)
        # take the sume of transformer outputs
        outputs = torch.tanh(torch.sum(outputs, dim=0))

        return F.softmax(outputs, dim=1)

class TransformerLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, dropout_rate, device):

        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim).to(device)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout_rate, device)

        self.dropout = nn.Dropout(dropout_rate).to(device)

    def forward(self, keywords, keywords_mask):
        # keywords = [batch size, keywords len, hid_dim]
        # keywords_mask = [batch size, keywords len]

        # Self attention and normalizing phase
        _keywords, _ = self.self_attention(keywords,keywords,keywords,keywords_mask)
        # keywords = [batch size, keywords len, hid_dim]
        keywords = self.layer_norm(keywords + self.dropout(keywords))

        return keywords

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout_rate, device):
        super().__init__()

        assert hid_dim % n_heads == 0
        # the dimension of hidden state in a head
        self.hid_dim = hid_dim
        # the number of heads
        self.n_heads = n_heads
        # the dimension of each head, denoted as d_{k} in the tutorial
        # head_dim = 64 if hid_dim = 256 and n_heads = 4
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim).to(device)
        self.fc_k = nn.Linear(hid_dim, hid_dim).to(device)
        self.fc_v = nn.Linear(hid_dim, hid_dim).to(device)

        self.fc_o = nn.Linear(hid_dim, hid_dim).to(device)

        self.dropout = nn.Dropout(dropout_rate).to(device)
        # sqrt(head_dim)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q = [batch size, query len, hid dim], Q = XW^{Q}
        # K = [batch size, key len, hid dim], K = XW^{K}
        # V = [batch size, value len, hid dim], V = XW^{V}

        # Split Q, K, and V into multiple heads
        # e.g. if hid_dim = 256 and n_heads = 4
        # , Q = [batch size, query len, hid dim] -> [batch size, query len, 4, 64] by view -> [batch size, 4, query len, 64] by permute(0,2,1,3)

        Q = Q.view(batch_size, query.size()[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, key.size()[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, value.size()[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        # non-normalized attentional weights
        # energy = [batch size, n heads, seq len, seq len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # mask is not None for decoder
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # normalize attention weights by softmax
        # attention = [batch size, n heads, query len, key len]
        attention = torch.softmax(energy, dim = -1)

        # calculate the weighted sum of source tokens
        # x = [batch size, n heads, seq len, head dim]
        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, seq len, n heads, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, seq len, hid dim]
        x = x.view(batch_size, x.size()[1], self.hid_dim)

        # Get the context vector by concatenating all heads and multiply it by W^{O}
        # x = [batch size, seq len, hid dim]
        x = self.fc_o(x)

        return x, attention
