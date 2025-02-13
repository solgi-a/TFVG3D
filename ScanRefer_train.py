import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
import torch.nn as nn
import numpy as np
import pickle

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.config import CONF
from models.refnet import RefNet
from utils.AdamW import AdamW
from utils.script_utils import set_params_lr_dict

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
DC = ScannetDatasetConfig()

#print(sys.path, '<< sys path')

def comp_weight(our_model,weight):

    our_model_state_dict = our_model.state_dict()
    our_model_state_dict_keys = list(our_model_state_dict.keys())
    weight_keys = list(weight.keys())

    for i in weight_keys:
        if i in our_model_state_dict_keys:
            if weight[i].shape == our_model_state_dict[i].shape:
                our_model_state_dict[i] = weight[i]

    our_model.load_state_dict(our_model_state_dict)


def get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, split, config, augment, shuffle=True):
    dataset = ScannetReferenceDataset(
        args = args,
        scanrefer=scanrefer[split],
        scanrefer_new=scanrefer_new[split],
        scanrefer_all_scene=all_scene_list,
        split=split,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        lang_num_max=args.lang_num_max,
        augment=augment,
        shuffle=shuffle
    )
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    return dataset, dataloader

def get_model(args):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = RefNet(
        args=args,
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=args.no_reference,
        dataset_config=DC
    )

    # trainable model
    if args.use_pretrained:
        # load model
          
        print("\nloading pretrained VoteNet...")

        pretrained_model = RefNet(
            args=args,
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            input_feature_dim=input_channels,
            num_proposal=args.num_proposals,
            use_lang_classifier=(not args.no_lang_cls),
            use_bidir=args.use_bidir,
            no_reference=args.no_reference,
            dataset_config=DC
        )

        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_criteria_25.pth")
        pretrained_model.load_state_dict(torch.load(pretrained_path, weights_only=False), strict=False)

        # mount
        model.backbone_net = pretrained_model.backbone_net
        model.vgen = pretrained_model.vgen
        model.proposal = pretrained_model.proposal

        if args.no_detection:
            # freeze pointnet++ backbone
            for param in model.backbone_net.parameters():
                param.requires_grad = False

            # freeze voting
            for param in model.vgen.parameters():
                param.requires_grad = False

            # freeze detector
            for param in model.proposal.parameters():
                param.requires_grad = False
                    
    # to CUDA
    model = model.cuda()

    return model


def get_num_params(model):
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader):

    model = get_model(args)

    # different lr for various modules.
    weight_dict = {'lang': {'lr': args.lang_lr}, 'match': {'lr': args.match_lr}}

    # scheduler parameters for training solely the detection pipeline
    lr_decay_rate = 0.1 if args.no_reference else None
    bn_decay_step = 20 if args.no_reference else None
    bn_decay_rate = 0.5 if args.no_reference else None
    lr_decay_step = [80, 120, 160] if args.no_reference else None

    if args.coslr:
        lr_decay_step = {
            'type': 'cosine',
            'T_max': args.epoch,
            'eta_min': 1e-6,
        }

    params = set_params_lr_dict(model, base_lr=args.lr, weight_decay=args.wd, weight_dict=weight_dict)
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad)

    # lr scheduler
    if lr_decay_step:
        if isinstance(lr_decay_step, list):
            lr_scheduler = MultiStepLR(optimizer, lr_decay_step, lr_decay_rate)
        elif isinstance(lr_decay_step, dict):
            if lr_decay_step['type'] != 'cosine':
                raise NotImplementedError('lr dict type should be cosine (other not implemented)')
            print(lr_decay_step, '<< lr_decay_step dict', flush=True)  # TODO
            config = lr_decay_step
            config['optimizer'] = optimizer
            config.pop('type')
            lr_scheduler = CosineAnnealingLR(**config)
        else:
            lr_scheduler = StepLR(optimizer, lr_decay_step, lr_decay_rate)
    else:
        lr_scheduler = None

    if args.use_checkpoint:
        print("\nloading checkpoint {}...".format(args.use_checkpoint))

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

        best_model_weights = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "model_criteria_25.pth"), weights_only=False)
        comp_weight(model,best_model_weights)
        #model.load_state_dict(best_model_weights, strict=False)
  
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)


    #print('LR&BN_DECAY', lr_decay_step, lr_decay_rate, bn_decay_step, bn_decay_rate, flush=True)

    solver = Solver(args=args,
        model=model,
        config=DC,
        dataloader=dataloader,
        optimizer=optimizer,
        stamp=stamp,
        val_step=args.val_step,
        detection=not args.no_detection,
        reference=not args.no_reference,
        use_lang_classifier=not args.no_lang_cls,
        lr_decay_step=lr_decay_step,
        lr_decay_rate=lr_decay_rate,
        bn_decay_step=bn_decay_step,
        bn_decay_rate=bn_decay_rate,
        lr_scheduler=lr_scheduler
    )

    num_params = get_num_params(model)
    num_params_lang = get_num_params(model.lang)
    num_params_match = get_num_params(model.match)
    num_params_detector = num_params - num_params_match - num_params_lang


    print("\n")
    print(f"num_params: {num_params}")
    print(f"num_params_detector: {num_params_detector}")
    print(f"num_params_lang: {num_params_lang}")
    print(f"num_params_match: {num_params_match}")

    return solver, num_params, root


def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def get_scannet_scene_list(split):
    scene_list = sorted(
        [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list


def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes, lang_num_max):
    if args.no_reference:
        train_scene_list = get_scannet_scene_list("train")
        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        val_scene_list = get_scannet_scene_list("val")
        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_VAL[0])
            data["scene_id"] = scene_id
            new_scanrefer_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        if num_scenes == -1:
            num_scenes = len(train_scene_list)
        else:
            assert len(train_scene_list) >= num_scenes

        # slice train_scene_list
        train_scene_list = train_scene_list[:num_scenes]

        # filter data in chosen scenes
        new_scanrefer_train = []
        scanrefer_train_new = []
        scanrefer_train_new_scene = []
        scene_id = ""
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)
                if scene_id != data["scene_id"]:
                    scene_id = data["scene_id"]
                    if len(scanrefer_train_new_scene) > 0:
                        scanrefer_train_new.append(scanrefer_train_new_scene)
                    scanrefer_train_new_scene = []
                if len(scanrefer_train_new_scene) >= lang_num_max:
                    scanrefer_train_new.append(scanrefer_train_new_scene)
                    scanrefer_train_new_scene = []
                scanrefer_train_new_scene.append(data)
                """
                if data["scene_id"] not in scanrefer_train_new:
                    scanrefer_train_new[data["scene_id"]] = []
                scanrefer_train_new[data["scene_id"]].append(data)
                """
        scanrefer_train_new.append(scanrefer_train_new_scene)

        new_scanrefer_val = scanrefer_val
        scanrefer_val_new = []
        scanrefer_val_new_scene = []
        scene_id = ""
        for data in scanrefer_val:
            # if data["scene_id"] not in scanrefer_val_new:
            # scanrefer_val_new[data["scene_id"]] = []
            # scanrefer_val_new[data["scene_id"]].append(data)
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(scanrefer_val_new_scene) > 0:
                    scanrefer_val_new.append(scanrefer_val_new_scene)
                scanrefer_val_new_scene = []
            if len(scanrefer_val_new_scene) >= lang_num_max:
                scanrefer_val_new.append(scanrefer_val_new_scene)
                scanrefer_val_new_scene = []
            scanrefer_val_new_scene.append(data)
        scanrefer_val_new.append(scanrefer_val_new_scene)

    print("\nscanrefer_train_new", len(scanrefer_train_new), len(scanrefer_val_new), len(scanrefer_train_new[0]))  # 4819 1253 8
    sum = 0
    for i in range(len(scanrefer_train_new)):
        sum += len(scanrefer_train_new[i])
    print("training sample numbers", sum)  # 36665
    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples\n".format(len(new_scanrefer_train), len(new_scanrefer_val)))  # 36665 9508

    return new_scanrefer_train, new_scanrefer_val, all_scene_list, scanrefer_train_new, scanrefer_val_new


def train(args):
    # init training dataset
    print("\npreparing data...")
    scanrefer_train, scanrefer_val, all_scene_list, scanrefer_train_new, scanrefer_val_new = get_scanrefer(
        SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes, args.lang_num_max)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }
    scanrefer_new = {
        "train": scanrefer_train_new,
        "val": scanrefer_val_new
    }

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, "train", DC, augment=True)
    val_dataset, val_dataloader = get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, "val", DC, augment=False)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("\ninitializing...")
    solver, num_params, root = get_solver(args, dataloader)

    print("\nStart training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=14)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=5000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--lang_num_max", type=int, help="lang num max", default=32)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--coslr", action='store_true', help="cosine learning rate")
    parser.add_argument("--amsgrad", action='store_true', help="optimizer with amsgrad")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use augment on trainingset (not used)")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--detection", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_pretrained", type=str, help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--lang_lr", type=float, help="lang module learning rate", default=0.0005)
    parser.add_argument("--match_lr", type=float, help="match module learning rate", default=0.0005)

    #----------------------------------------------------------------------------------------------------------------------------------
    
    args = parser.parse_args()
    args.detection = args.no_reference

    debug = False

    if debug:
        args.use_color = True
        args.use_normal = True
        args.batch_size = 8
        args.lang_num_max = 32
        args.epoch = 50
        args.lr = 0.002
        args.coslr = True
        args.tag = '3DVG-AISP'
        args.val_step = 50

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)

