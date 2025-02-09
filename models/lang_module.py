import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from copy import deepcopy

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.attention_modules import MultiHeadAttention

class LangModule(nn.Module):
    def __init__(self, num_text_classes, use_lang_classifier=True, use_bidir=False,
                 emb_size=300, hidden_size=256, final_size=128):
        super().__init__()

        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.use_bidir
        )
        lang_size = hidden_size * 2 if self.use_bidir else hidden_size

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(lang_size, num_text_classes),
                nn.Dropout()
            )

        self.lang_sa = MultiHeadAttention(d_model=emb_size, d_k=16, d_v=16, h=8, dropout=.1, identity_map_reordering=False,
                                        attention_module=None,
                                        attention_module_kwargs=None)
        self.ML = nn.Sequential(nn.Linear(emb_size, final_size), nn.ReLU(),nn.Dropout(p=.1),nn.LayerNorm(final_size))

    def _lang_model_forward(self, word_embs, lang_len):

        lang_feat = pack_padded_sequence(word_embs, lang_len.cpu(), batch_first=True, enforce_sorted=False)

        out, lang_last = self.gru(lang_feat)

        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        if self.use_bidir:
            cap_emb = (cap_emb[:, :, :int(cap_emb.shape[2] / 2)] + cap_emb[:, :, int(cap_emb.shape[2] / 2):]) / 2

        b_s, seq_len = cap_emb.shape[:2]
        mask_queries = torch.ones((b_s, seq_len), dtype=torch.int)
        for i in range(b_s):
            mask_queries[i, cap_len[i]:] = 0
        attention_mask = (mask_queries == 0).unsqueeze(1).unsqueeze(1).cuda()       # (b_s, 1, 1, seq_len)
                
        lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(start_dim=1)    # batch_size, hidden_size * num_dir

        return cap_emb, lang_last, attention_mask

    def forward(self, data_dict):
        """
        encode the input descriptions
        """
        
        word_embs = data_dict["lang_feat_list"]  # B * 32 * MAX_DES_LEN * LEN(300)
        lang_len = data_dict["lang_len_list"]
        batch_size, len_nun_max, max_des_len = word_embs.shape[:3]

        word_embs = word_embs.reshape(batch_size * len_nun_max, max_des_len, -1)
        lang_len = lang_len.reshape(batch_size * len_nun_max)

        #----------------------------------------------------------------------------------------------------------------------------------

        _, lang_last, _ = self._lang_model_forward(word_embs, lang_len)
        #data_dict["attention_mask"] = attention_mask
        data_dict["lang_emb"] = lang_last

        word_embs_mask = torch.ones((batch_size * len_nun_max, max_des_len), dtype=torch.int)
        for i in range(batch_size * len_nun_max):
            word_embs_mask[i, lang_len[i]:] = 0
        word_embs_mask = (word_embs_mask == 0).unsqueeze(1).unsqueeze(1).cuda()

        lang_features = self.lang_sa(word_embs, word_embs, word_embs, word_embs_mask)
        lang_features = self.ML(lang_features)
        data_dict["lang_features"] = lang_features
        data_dict["word_embs_mask"] = word_embs_mask
        
        #----------------------------------------------------------------------------------------------------------------------------------

        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])

        return data_dict

