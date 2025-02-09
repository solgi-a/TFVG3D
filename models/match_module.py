import torch.nn as nn
from models.attention_modules import Transformer

class MatchModule(nn.Module):
    def __init__(self, hidden_size=128, head=8):
        super().__init__()

        self.lang_last_convert = nn.Sequential(
            nn.Conv1d(256, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
        )

        self.evaluate = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),

            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),

            nn.Conv1d(hidden_size, 1, 1)
        )

        self.object_sa1 = Transformer(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) 
        
        self.object_lang_ca1 = Transformer(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) 

        self.object_sa2 = Transformer(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)   # k, q, v

        self.object_lang_ca2 = Transformer(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)  # k, q, v

    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        
        lang_features = data_dict["lang_features"]                                                          # batch_size, num_word, lang_size
        lang_last = data_dict["lang_emb"].unsqueeze(1)                                                      # batch_size, 1, lang_size

        object_features = data_dict['aggregated_vote_features']                                             # batch_size, num_proposals, obj_szie
        objectness_masks = data_dict['objectness_scores'].max(2)[1].float()                                 # batch_size, num_proposals

        #--------------------------------------------------------------------------------------------------------------------

        lang_last = self.lang_last_convert(lang_last.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()

        #--------------------------------------------------------------------------------------------------------------------

        lang_num_max = lang_features.shape[0]//object_features.shape[0]

        features = self.object_sa1(object_features, object_features, object_features)
        features = features[:,None,:,:].repeat(1,lang_num_max,1,1).reshape(lang_features.shape[0],objectness_masks.shape[1],-1)

        features = self.object_lang_ca1(features, lang_features, lang_features, data_dict["word_embs_mask"])
        features = self.object_sa2(features, features, features)
        features = self.object_lang_ca2(features, lang_last, lang_last)
        
        # evaluate
        features = features.permute(0, 2, 1).contiguous()
        objectness_masks = objectness_masks[:,None,:].repeat(1,lang_num_max,1).reshape(features.shape[0],-1)

        confidence = self.evaluate(features).squeeze(1)*objectness_masks                                    # batch_size, num_proposals
        data_dict["cluster_ref"] = confidence

        return data_dict
