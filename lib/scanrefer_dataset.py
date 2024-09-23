import os
import sys
import time
import h5py
import pickle
import numpy as np
import multiprocessing as mp
import torch
from torch.utils.data import Dataset

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from data.scannet.model_util_scannet import ScannetDatasetConfig, rotate_aligned_boxes_along_axis

#-------------------------------------------------------------------------------------------------------------------------------

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")

#-------------------------------------------------------------------------------------------------------------------------------

class ScanReferDataset(Dataset):
       
    def __init__(self, 
                scanrefer, 
                scanrefer_all_scene, 
                split               ="train", 
                num_points          = 40000,
                use_scene_data      = False,
                use_color           = False, 
                use_height          = False, 
                use_normal          = False, 
                use_multiview       = False, 
                augment             = False,
                batch               = 32):

        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.num_points = num_points
        self.use_scene_data = use_scene_data
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment
        self.batch = batch

        self.use_group_free = True

        # add scannet data
        self.num_samples = len(scanrefer)
        self.scene_list = sorted(list(set([data['scene_id'] for data in self.scanrefer])))

        self.num_scene_list = len(self.scene_list)
        self.batch_scene_list = []

        for scene_index in range(0,self.num_scene_list):
            scene_prompts = []
            for data in self.scanrefer:
                if data['scene_id'] == self.scene_list[scene_index]:
                    scene_prompts.append(data)

            num_prompts = len(scene_prompts)
            scene_batch = int(np.ceil(num_prompts / self.batch))

            for i in range(scene_batch-1):

                start = self.batch*i
                end = self.batch-1 + self.batch*i
                batch_list = [self.scene_list[scene_index],start,end]
                self.batch_scene_list.append(batch_list)
            
            start = self.batch*(scene_batch-1)
            end = num_prompts-1
            batch_list = [self.scene_list[scene_index],start,end]
            self.batch_scene_list.append(batch_list)

        # load data
        self._load_data()
        self.multiview_data = {}
    
    def __len__(self):
        return len(self.batch_scene_list)

#-------------------------------------------------------------------------------------------------------------------------------
    
    def __getitem__(self, id):

        # load scene data
        start = time.time()
        
        scene = self.batch_scene_list[id][0]
        start_num = self.batch_scene_list[id][1]
        end_num = self.batch_scene_list[id][2]
        batch_num_propmts = end_num + 1 - start_num

        # get pc
        mesh_vertices   = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene)+"_aligned_vert.npy")
        instance_labels = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene)+"_ins_label.npy")
        semantic_labels = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene)+"_sem_label.npy")
        instance_bboxes = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene)+"_aligned_bbox.npy")

        if self.use_scene_data:
            
            scene_data_path = os.path.join(CONF.PATH.SCANNET, 'scenes_data',scene+"_data.p")
            with open(scene_data_path, "rb") as f:
                scene_data = pickle.load(f)

            #group_free_scene_data_path = os.path.join(CONF.PATH.SCANNET, 'group_free', scene+".npy")
            #group_free_scene_data = np.load(group_free_scene_data_path,allow_pickle=True)
            #group_free_scene_data = group_free_scene_data.tolist()
            
            #group_free_pred_bboxes = np.array(group_free_scene_data['box'])
            #group_free_num_objects = len(group_free_pred_bboxes)
            #group_free_logits = group_free_scene_data['logits']
            
            #pred_bboxes = np.zeros((MAX_NUM_OBJ,7))
            #logits = np.zeros((MAX_NUM_OBJ,485))

            #pred_bboxes[0:group_free_num_objects,0:6] = group_free_pred_bboxes
            #logits[0:group_free_num_objects,:] = group_free_logits
        
        # processing pc
        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] 
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0 
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            multiview = self.multiview_data[pid][scene]
            point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 

        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        #---------------------------------------------------------------------------------------------------
        
        self.scene_prompts = []
        for data in self.scanrefer:
            if data["scene_id"] == scene:
                self.scene_prompts.append(data)

        # --------------------------------------------- LABELS ---------------------------------------------
    
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))      ##-- detection ## center(3) & l,w,h of bboxes 
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    ##-- detection ## it shows that bbox exist or not(1 for every bbox {0:num_bbox} and 0 for others {num_bbox:MAX_NUM_OBJ})
        angle_classes = np.zeros((MAX_NUM_OBJ,))        ##-- detection ## 0 for every bbox because bboxes are "axis_aligned"! ##
        angle_residuals = np.zeros((MAX_NUM_OBJ,))      ##-- detection ## 0 for every bbox because bboxes are "axis_aligned"! ##
        size_classes = np.zeros((MAX_NUM_OBJ,))         ##-- detection ## semantic class of every bbox (Consider use size2class!)
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))     ##-- detection ## residual size of every bbox in comparison with mean size of that class in all scenes of dataset
        

        if self.split != "test": 
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox,:] = instance_bboxes[:MAX_NUM_OBJ,0:6]

            point_votes = np.zeros([self.num_points, 3])
            point_votes_mask = np.zeros(self.num_points)

            # ------------------------------- DATA AUGMENTATION ------------------------------        
            if self.augment: ## this part has been changed by me
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    point_cloud[:,0] = -1 * point_cloud[:,0]
                    target_bboxes[:,0] = -1 * target_bboxes[:,0]                
                    
                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    point_cloud[:,1] = -1 * point_cloud[:,1]
                    target_bboxes[:,1] = -1 * target_bboxes[:,1]                                

                # Rotation along X-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotx(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

                # Rotation along Y-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = roty(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")

                # Translation
                point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes, num_bbox)

            # compute votes *AFTER* augmentation
            # generate votes
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered 
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label. 
            for i_instance in np.unique(instance_labels):            
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                # find the semantic label            
                if semantic_labels[ind[0]] in DC.nyu40ids:
                    x = point_cloud[ind,:3]
                    center = 0.5*(x.min(0) + x.max(0))
                    point_votes[ind, :] = center - x
                    point_votes_mask[ind] = 1.0
            point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
            
            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]] ## 7th element of instance_bbox is semantic id / class_ind = 18 class id
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]
            
        else: 
            num_bbox = 1
            point_votes = np.zeros([self.num_points, 9]) # make 3 votes identical 
            point_votes_mask = np.zeros(self.num_points)

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))

        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
        except KeyError:
            pass
        
        #---------------------------------------------------------------------------------------------------------------------------------------

        object_id_list = []
        ann_id_list = []
        lang_feat_list = []
        lang_len_list = []
        object_cat_list = []
        unique_multiple_list = []

        
        ref_box_label = np.zeros((batch_num_propmts,MAX_NUM_OBJ))           ##-- refrence  # bbox label for reference target
        ref_center_label = np.zeros((batch_num_propmts,3))                  ##-- refrence  # bbox center for reference target
        ref_heading_class_label = np.zeros((batch_num_propmts))             ##-- refrence
        ref_heading_residual_label = np.zeros((batch_num_propmts))          ##-- refrence
        ref_size_class_label = np.zeros((batch_num_propmts))                ##-- refrence
        ref_size_residual_label = np.zeros((batch_num_propmts,3))           ##-- refrence  # bbox size residual for reference target
        

        for index in range(start_num,end_num+1):
            
            object_id = int(self.scene_prompts[index]["object_id"])
            object_name = " ".join(self.scene_prompts[index]["object_name"].split("_"))
            ann_id = self.scene_prompts[index]["ann_id"]
            
            # get language features
            lang_feat = self.lang[scene][str(object_id)][ann_id]
            lang_len = len(self.scene_prompts[index]["token"])
            lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN else CONF.TRAIN.MAX_DES_LEN
            unique_multiple = self.unique_multiple_lookup[scene][str(object_id)][ann_id]

            object_id_list.append(int(object_id))
            ann_id_list.append(int(ann_id))
            lang_feat_list.append(lang_feat)
            lang_len_list.append(lang_len)
            unique_multiple_list.append(unique_multiple)

            if self.split != "test":

                # construct the reference target label for each bbox
                ## 8th element of instance_bbox is instance id
                for i, gt_id in enumerate(instance_bboxes[:num_bbox,-1]): 

                    if gt_id == object_id:
                        ref_box_label[index - start_num,i] = 1
                        ref_center_label[index - start_num] = target_bboxes[i, 0:3]
                        ref_heading_class_label[index - start_num] = int(angle_classes[i])
                        ref_heading_residual_label[index - start_num] = int(angle_residuals[i])
                        ref_size_class_label[index - start_num] = int(size_classes[i])
                        ref_size_residual_label[index - start_num] = size_residuals[i]
                
            ## converting object's "raw" name to nyu40class "label" (only 40 class) for example monitor --> otherprop
            object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17
            object_cat_list.append(object_cat)
            
        data_dict = {}

        ######################################################################################################
        if self.use_scene_data:

            #data_dict["group_free"] = self.use_group_free
            #data_dict["group_free_pred_bboxes"] = pred_bboxes
            #data_dict["group_free_logits"] = logits
            #data_dict['group_free_num_objects'] = group_free_num_objects

            save = ['aggregated_vote_xyz','aggregated_vote_features','aggregated_vote_inds','generated_proposals',
                    'seed_inds','seed_xyz','seed_features','vote_xyz','vote_features']

            for key in save:
                data_dict[key] = scene_data[key]

            data_dict = self._decode_scores(data_dict['generated_proposals'],data_dict,DC.num_class,DC.num_heading_bin,DC.num_size_cluster,DC.mean_size_arr)

            key_list = ['aggregated_vote_xyz','aggregated_vote_features','aggregated_vote_inds','generated_proposals',
                        'seed_inds','seed_xyz','seed_features','vote_xyz','vote_features',
                        'objectness_scores','center','heading_scores','heading_residuals_normalized','heading_residuals',
                        'size_scores','size_residuals_normalized','size_residuals','sem_cls_scores']

            for key in key_list:
                data_dict[key] = data_dict[key].squeeze(0)

        ######################################################################################################
        
        #---------------------------------------------------det data_dict---------------------------------------------------

        data_dict["point_clouds"] = point_cloud.astype(np.float32)                  # point cloud data including features
        data_dict["pcl_color"] = pcl_color

        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3]         # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64)           # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32)    # (MAX_NUM_OBJ,) 
        data_dict["size_class_label"] = size_classes.astype(np.int64)               # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER ## (semantic 18 class id)
        data_dict["size_residual_label"] = size_residuals.astype(np.float32)        # (MAX_NUM_OBJ, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64)          # (MAX_NUM_OBJ,) semantic class index 
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32)         # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict["scan_idx"] = np.array(id).astype(np.int64)

        #---------------------------------------------------ref data_dict---------------------------------------------------

        data_dict["lang_feat"] = np.array(lang_feat_list).astype(np.float32)                     # language feature vectors ## (MAX_DES_LEN, 300)
        data_dict["lang_len"] = np.array(lang_len_list).astype(np.int64)                         # length of each description

        data_dict["ref_box_label"] = ref_box_label.astype(np.int64)                              #  0/1 reference labels for each object bbox
        data_dict["ref_center_label"] = ref_center_label.astype(np.float32)                      ## (X,Y,Z) of reference target (object of interest)
        data_dict["ref_heading_class_label"] = ref_heading_class_label.astype(np.int64)          ## ref_heading_class_label = 0
        data_dict["ref_heading_residual_label"] = ref_heading_residual_label.astype(np.int64)    ## ref_heading_residual_label = 0
        data_dict["ref_size_class_label"] = ref_size_class_label.astype(np.int64)                ## semantic 18 class id
        data_dict["ref_size_residual_label"] = ref_size_residual_label.astype(np.float32)        ## size residual of reference target (object of interest)
        
        data_dict["object_id"] = np.array(object_id_list).astype(np.int64)
        data_dict["ann_id"] = np.array(ann_id_list).astype(np.int64)
        data_dict["object_cat"] = np.array(object_cat_list).astype(np.int64)
        data_dict["unique_multiple"] = np.array(unique_multiple_list).astype(np.int64)
        data_dict["load_time"] = time.time() - start
        
        return data_dict
    
    ######################################################################################################
    def _decode_scores(self, net, data_dict, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
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
        ######################################################################################################
#-------------------------------------------------------------------------------------------------------------------------------

    ## converting object's "raw" name to "0-17" "label" (18 class) for example monitor --> 17
    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

#-------------------------------------------------------------------------------------------------------------------------------
    
    ## 1 -> multiple -- 0 -> unique
    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

#-------------------------------------------------------------------------------------------------------------------------------
    
    def _tranform_des(self):
        with open(GLOVE_PICKLE, "rb") as f:
            glove = pickle.load(f)

        lang = {}

        for data in self.scanrefer:

            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if scene_id not in lang:
                lang[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}

            # tokenize the description
            tokens = data["token"]
            embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN, 300)) 
            for token_id in range(CONF.TRAIN.MAX_DES_LEN):
                if token_id < len(tokens):
                    token = tokens[token_id]
                    if token in glove:
                        embeddings[token_id] = glove[token]
                    else:
                        embeddings[token_id] = glove["unk"]

            # store
            lang[scene_id][object_id][ann_id] = embeddings

        return lang

#-------------------------------------------------------------------------------------------------------------------------------
    
    def _load_data(self):

        print("loading data...")
        # load language features
        self.lang = self._tranform_des()

        # prepare class mapping / ## converting object's "raw" name to nyu40id (an integer between [1-40])
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid                                          ## 40 id mapping (object name to [1-40])
        self.raw2label = self._get_raw2label()                              ## 18 class mapping (object name to label) (for example monitor --> 17)
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

#-------------------------------------------------------------------------------------------------------------------------------
    
    def _translate(self, point_set, bbox, num_bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:num_bbox, :3] += factor #########################################################################################

        return point_set, bbox
