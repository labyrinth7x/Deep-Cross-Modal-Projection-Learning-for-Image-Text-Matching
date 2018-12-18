import torch
import torch.nn as nn
import random

seed_num = 223
torch.manual_seed(seed_num)
random.seed(seed_num)

"""
Neural Networks model : Bidirection LSTM
"""


class BiLSTM(nn.Module):

    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.hidden_dim = args.num_lstm_units

        V = args.vocab_size
        D = args.embedding_size

        # word embedding
        self.embed = nn.Embedding(V, D, padding_idx=0)

        self.bilstm = nn.LSTM(D, args.num_lstm_units, num_layers=1, dropout=1-args.lstm_dropout_ratio, bidirectional=True,
                              bias=False)

    def forward(self, text, text_length):
        embed = self.embed(text)

        '''
        _, idx_sort = torch.sort(text_length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        embed_sort = embed.index_select(0, idx_sort)
        length_list = text_length[idx_sort]
        pack = nn.utils.rnn.pack_padded_sequence(embed_sort, length_list, batch_first = True)
        
        bilstm_sort_out, _ = self.bilstm(pack)
        bilstm_sort_out = nn.utils.rnn.pad_packed_sequence(bilstm_sort_out, batch_first = True)
        bilstm_sort_out = bilstm_sort_out[0]
        
        #bilstm_out = torch.zeros(bilstm_sort_out.shape).cuda(non_blocking=True)
        #bilstm_out[idx_sort] = bilstm_sort_out
        bilstm_out = bilstm_sort_out.index_select(0, idx_unsort)
        '''
        bilstm_out = self.bilstm(embed, text_length)

        bilstm_out, _ = torch.max(bilstm_out, dim=1)
        bilstm_out = bilstm_out.unsqueeze(2).unsqueeze(2)
        return bilstm_out

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data, 1)
            nn.init.constant(m.bias.data, 0)


class Bilstm(nn.Module):
    def __init__(self, embedding_size, num_lstm_units, num_layers, dropout, bidirectional, bias):
        self.bilstm = nn.ModuleList()
        self.bilstm.append(nn.LSTM(embedding_size, num_lstm_units, num_layers, dropout, False, bias))
        self.bidirectional = bidirectional
        if bidirectional:
            self.bilstm.append(nn.LSTM(embedding_size, num_lstm_units, num_layers, dropout, False, bias))


    def forward(self, embed, text_length):

        bilstm_out = self.bilstm_out(embed, text_length, 0)
        if self.bidirectional:
            index_reverse = list(range(embed.shape[0]-1, -1, -1))
            index_reverse = torch.LongTensor(index_reverse)
            embed_reverse = embed.index_select(0, index_reverse)
            text_length_reverse = text_length.index_select(0, index_reverse)
            bilstm_out_bidirection = self.bilstm_out(embed_reverse, text_length_reverse, 1)
            bilstm_out_bidirection_reverse = bilstm_out_bidirection.index_select(0, index_reverse)
            bilstm_out = torch.cat([bilstm_out, bilstm_out_bidirection_reverse], dim=1)

        return bilstm_out

    def bilstm_out(self, embed, text_length, index):

        _, idx_sort = torch.sort(text_length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        embed_sort = embed.index_select(0, idx_sort)
        length_list = text_length[idx_sort]
        pack = nn.utils.rnn.pack_padded_sequence(embed_sort, length_list, batch_first=True)

        bilstm_sort_out, _ = self.bilstm[index](pack)
        bilstm_sort_out  = nn.utils.rnn.pad_packed_sequence(bilstm_sort_out, batch_first=True)
        bilstm_sort_out = bilstm_sort_out[0]

        bilstm_out = bilstm_sort_out.index_select(0, idx_unsort)

        return bilstm_out