# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

from pointnet2_ops.pointnet2_modules import PointnetSAModuleVotes

class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling,
                 seed_feat_dim=256, dataset_config=None):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.dataset_config = dataset_config

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        dimen = 128
        self.proposal = nn.Sequential(
            nn.Conv1d(dimen,dimen,1, bias=False),
            nn.BatchNorm1d(dimen),
            nn.ReLU(),
            nn.Conv1d(dimen,dimen,1, bias=False),
            nn.BatchNorm1d(dimen),
            nn.ReLU(),
            nn.Conv1d(dimen,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1) 
        )

    def forward(self, xyz, features, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        # Farthest point sampling (FPS) on votes
        xyz, features, fps_inds = self.vote_aggregation(xyz, features)
        
        sample_inds = fps_inds

        data_dict['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        data_dict['aggregated_vote_features'] = features.permute(0, 2, 1).contiguous() # (batch_size, num_proposal, 128)
        data_dict['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = self.proposal(features) 
        ######################################################################################################
        data_dict['generated_proposals'] = net
        ######################################################################################################
        data_dict = self.decode_scores(net, data_dict, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)

        return data_dict

    def decode_scores(self, net, data_dict, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
        """
        decode the predicted parameters for the bounding boxes

        """
        net_transposed = net.transpose(2,1).contiguous() ## (batch_size, num_proposal, ..)
        batch_size = net_transposed.shape[0]
        num_proposal = net_transposed.shape[1]

        objectness_scores = net_transposed[:,:,0:2]

        base_xyz = data_dict['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
        center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)

        heading_scores = net_transposed[:,:,5:5+num_heading_bin]
        heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin:5+num_heading_bin*2]
        
        size_scores = net_transposed[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
        size_residuals_normalized = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # (B,num_proposal,num_size_cluster,3)
        
        sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:] # (B,num_proposal,num_class) 

        # store
        data_dict['objectness_scores'] = objectness_scores
        data_dict['center'] = center
        data_dict['heading_scores'] = heading_scores # (B,num_proposal,num_heading_bin)
        data_dict['heading_residuals_normalized'] = heading_residuals_normalized # (B,num_proposal,num_heading_bin) (should be -1 to 1) ##?
        data_dict['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # (B,num_proposal,num_heading_bin) ##?
        data_dict['size_scores'] = size_scores
        data_dict['size_residuals_normalized'] = size_residuals_normalized
        data_dict['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        data_dict['sem_cls_scores'] = sem_cls_scores

        return data_dict

