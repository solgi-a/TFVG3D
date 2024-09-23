import os
import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#-------------------------------------------------------------------------------------------------------------------------------

class GruLayer(nn.Module):
    def __init__(self, num_text_classes, use_lang_classifier=True, use_bidir=False, num_layers=1, emb_size=300, hidden_size=128, out_dim=128):
        super().__init__()

        self.use_bidir = use_bidir
        self.num_bidir = 2 if self.use_bidir else 1
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.use_lang_classifier = use_lang_classifier
        
        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.use_bidir
        )

        #in_dim = hidden_size * 2 if use_bidir else hidden_size
        in_dim = 300
        self.mlps = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1),
        )

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(hidden_size, num_text_classes),
                nn.Dropout()
            )

    def forward(self, data_dict):
        """
        encode the input descriptions
        """

        word_embs = data_dict["lang_feat"]  # [B, MAX_DES_LEN, 300]
        max_des_len = word_embs.shape[1]
        lang_feat = pack_padded_sequence(word_embs, data_dict["lang_len"].cpu(), batch_first=True, enforce_sorted=False)
        
        # encode description
        lang_feat, last_feat = self.gru(lang_feat)  
        # lang_feat:[B, MAX_DES_LEN, D * hidden_size] , last_feat:[D * num_layer, B, hidden_size]
        last_feat = last_feat.view(self.num_bidir, self.num_layers, last_feat.shape[1], last_feat.shape[2])[:, -1, :, :] 
        # last_feat: [D, num_layer, B, hidden_size], choose last gru layer hidden ----> last_feat: [D, B, hidden_size]    
        last_feat = last_feat.permute(1, 0, 2).contiguous().flatten(start_dim=1)                # [B, D * hidden_size]

        lang_feat, _ = pad_packed_sequence(lang_feat, batch_first=True, total_length=max_des_len)
        
        ###########################################################################
        enhanced_lang_feat = word_embs.transpose(-1, -2)                # [B, C, N]
        enhanced_lang_feat = self.mlps(enhanced_lang_feat)
        enhanced_lang_feat = enhanced_lang_feat.transpose(-1, -2)       # [B, N, C]
        ###########################################################################

        # store the encoded language features
        data_dict["lang_emb"] = enhanced_lang_feat  # [B, N, C]
        data_dict["lang_hidden"] = last_feat        # [B, C]

        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(last_feat)

        return data_dict
    