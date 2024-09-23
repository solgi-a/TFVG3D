import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_module import ProposalModule
from lang_module import *
from match_module import EvaluateModule

#-------------------------------------------------------------------------------------------------------------------------------

class RefNet(nn.Module):
    def __init__(self, args, num_class, num_heading_bin, num_size_cluster, mean_size_arr, 
                input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
                use_lang_classifier=True, use_bidir=False,num_layers=1, no_reference=False,
                emb_size=300, hidden_size=128, out_dim=128):
        
        super().__init__()
        assert(mean_size_arr.shape[0] == num_size_cluster)
        self.no_reference = no_reference
        self.args = args

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(vote_factor, 256)

        # Vote aggregation and object proposal
        self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)

        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = GruLayer(num_class, use_lang_classifier, use_bidir, num_layers, emb_size, hidden_size, out_dim) ##

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            self.match = EvaluateModule(num_proposals=num_proposal, att_dim=out_dim)

    def forward(self, data_dict):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################
        
        # --------- BACKBONE POINT FEATURE LEARNING ---------
        if not self.args.use_scenes_data:
            data_dict = self.backbone_net(data_dict)
            
            # --------- HOUGH VOTING ---------
            xyz = data_dict["fp2_xyz"]
            features = data_dict["fp2_features"]
            data_dict["seed_inds"] = data_dict["fp2_inds"]
            data_dict["seed_xyz"] = xyz
            data_dict["seed_features"] = features
            
            xyz, features = self.vgen(xyz, features)
            features_norm = torch.norm(features, p=2, dim=1)
            features = features.div(features_norm.unsqueeze(1))
            data_dict["vote_xyz"] = xyz
            data_dict["vote_features"] = features
            
            # --------- PROPOSAL GENERATION ---------
            data_dict = self.proposal(xyz, features, data_dict)
        
        if not self.no_reference:
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################

            # --------- LANGUAGE ENCODING ---------
            data_dict = self.lang(data_dict) 
            
            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            data_dict = self.match(data_dict)

        return data_dict
