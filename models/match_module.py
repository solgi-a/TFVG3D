import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_module import AttentionModule

#-------------------------------------------------------------------------------------------------------------------------------

class EvaluateModule(nn.Module):
    def __init__(self, num_proposals=256, object_size=128, lang_size=128, att_dim = 128):
        super().__init__() 

        self.num_proposals = num_proposals
        self.att_dim = att_dim
        self.att_dim_ff = 2*att_dim
        self.nhead = 8
        self.dropout = 0.1

        self.lang_sa         = AttentionModule(dim=self.att_dim,dim_ff=self.att_dim_ff,n_head=self.nhead,
                                              msa_dropout=self.dropout,ffn_dropout=self.dropout,ret_att=False)
        self.object_sa1      = AttentionModule(dim=self.att_dim,dim_ff=self.att_dim_ff,n_head=self.nhead,
                                              msa_dropout=self.dropout,ffn_dropout=self.dropout,ret_att=False)
        self.object_lang_ca1 = AttentionModule(dim=self.att_dim,dim_ff=self.att_dim_ff,n_head=self.nhead,
                                              msa_dropout=self.dropout,ffn_dropout=self.dropout,ret_att=False)
        self.object_sa2      = AttentionModule(dim=self.att_dim,dim_ff=self.att_dim_ff,n_head=self.nhead,
                                              msa_dropout=self.dropout,ffn_dropout=self.dropout,ret_att=False)
        self.object_lang_ca2 = AttentionModule(dim=self.att_dim,dim_ff=self.att_dim_ff,n_head=self.nhead,
                                              msa_dropout=self.dropout,ffn_dropout=self.dropout,ret_att=False)
        
        self.evaluate = nn.Sequential(
            nn.Conv1d(self.att_dim, 2*self.att_dim, 1),
            nn.BatchNorm1d(2*self.att_dim),
            nn.ReLU(),
            nn.Conv1d(2*self.att_dim, self.att_dim, 1),
            nn.BatchNorm1d(self.att_dim),
            nn.ReLU(),
            nn.Conv1d(self.att_dim, 1, 1)
        )

    def forward(self, data_dict):
        """
        Args:
            data_dict
        Returns:
            data_dict (confidences: (batch_size, num_proposals))
        """

        # unpack outputs from detection branch
        object_features = data_dict['aggregated_vote_features']                             # batch_size, num_proposal, 128
        objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2)    # batch_size, num_proposals, 1

        # unpack outputs from language branch
        lang_feat = data_dict["lang_emb"]       # batch_size, num_word, lang_size
        last_feat = data_dict["lang_hidden"]    # batch_size, lang_size
        last_feat = last_feat.unsqueeze(1)      # batch_size, 1, lang_size

        lang_sa_feat, _    = self.lang_sa(lang_feat,lang_feat,lang_feat,mask=None)                         # batch_size, num_word, lang_size

        object_sa_feat1, _ = self.object_sa1(object_features,object_features,object_features,mask=None)    # 1, num_proposal, 128

        object_sa_feat1    = object_sa_feat1.repeat(lang_sa_feat.shape[0],1,1)                             # batch_size, num_proposal, 128

        object_ca_feat1, _ = self.object_lang_ca1(object_sa_feat1,lang_sa_feat,lang_sa_feat,mask=None)     # batch_size, num_proposal, 128

        object_sa_feat2, _ = self.object_sa2(object_ca_feat1,object_ca_feat1,object_ca_feat1,mask=None)    # batch_size, num_proposal, 128

        object_ca_feat2, _ = self.object_lang_ca2(object_sa_feat2,last_feat,last_feat,mask=None)           # batch_size, num_proposal, 128

        #-------------------------------------------------------------------------------------------------------------------------------
        
        features = object_ca_feat2.permute(0,2,1) # batch_size, 128, num_proposal

        # fuse
        #features = torch.cat([features, lang_feat], dim=-1) # batch_size, num_proposals, 128 + lang_size
        #features = features.permute(0, 2, 1).contiguous() # batch_size, 128 + lang_size, num_proposals

        # mask out invalid proposals
        objectness_masks = objectness_masks.permute(0, 2, 1).contiguous()   # batch_size, 1, num_proposals
        features = features * objectness_masks                              # batch_size, 128, num_proposal

        # evaluate
        confidences = self.evaluate(features).squeeze(1)                    # batch_size, num_proposals
                
        data_dict["cluster_ref"] = confidences

        return data_dict
