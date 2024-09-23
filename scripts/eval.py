import os
import sys
import json
import pickle
import argparse
import torch
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config import CONF
from lib.dataset import ScannetReferenceDataset
from lib.scanrefer_dataset import ScanReferDataset
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from models.refnet import RefNet
from data.scannet.model_util_scannet import ScannetDatasetConfig

#-------------------------------------------------------------------------------------------------------------------------------

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

#-------------------------------------------------------------------------------------------------------------------------------

def get_dataloader(args, scanrefer, all_scene_list, split, device, augment = False):
    
    dataset = ScanReferDataset(
        scanrefer=scanrefer[split], 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        num_points=args.num_points, 
        use_scene_data=args.use_scenes_data,
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        augment=augment
    )

    print("evaluate on {} samples".format(len(dataset)))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    return dataset, dataloader

#-------------------------------------------------------------------------------------------------------------------------------

def get_model(args, config):
    # load model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = RefNet(
        args=args,
        num_class=config.num_class,
        num_heading_bin=config.num_heading_bin,
        num_size_cluster=config.num_size_cluster,
        mean_size_arr=config.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir
    ).to(device=args.device)


    model_name = "model_last.pth" if args.detection else "model.pth"
    path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()

    return model

#-------------------------------------------------------------------------------------------------------------------------------

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

#-------------------------------------------------------------------------------------------------------------------------------

def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes):

    ## train detection modules with scene0000_00, scene0000_01, ...
    if args.detection: 
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

    ## train all modules with scene0000_00, scene0001_00, ...
    else: 
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))

        ######################################################################################################
        if num_scenes == -1: 
            num_scenes = len(train_scene_list)
            new_scanrefer_train = scanrefer_train
        else:
            assert len(train_scene_list) >= num_scenes
        
            # slice train_scene_list
            train_scene_list = train_scene_list[:num_scenes]

            # filter data in chosen scenes
            new_scanrefer_train = []
            for data in scanrefer_train:
                if data["scene_id"] in train_scene_list:
                    new_scanrefer_train.append(data)
        ######################################################################################################

        new_scanrefer_val = scanrefer_val

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, train_scene_list, val_scene_list

#-------------------------------------------------------------------------------------------------------------------------------

def eval_ref(args):
    print("evaluate localization...")
    # constant
    DC = ScannetDatasetConfig() 

    ######################################################################################################
    # init training dataset
    print("preparing data...")
    # scanrefer, scene_list = get_scanrefer(args)
    scanrefer_train, scanrefer_val, train_scene_list, val_scene_list = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes)

    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }
    
    # dataloader
    if args.use_train :
        _, dataloader = get_dataloader(args, scanrefer, train_scene_list, "train", args.device)
        scene_list = train_scene_list
    else: 
        _, dataloader = get_dataloader(args, scanrefer, val_scene_list, "val", args.device)
        scene_list = val_scene_list
    ######################################################################################################
    
    # model
    model = get_model(args, DC)

    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    } if not args.no_nms else None

    # random seeds
    seeds = [args.seed] + [2 * i for i in range(args.repeat - 1)]

    # evaluate
    print("evaluating...")
    score_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "scores.p")
    pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "predictions.p")
    gen_flag = (not os.path.exists(score_path)) or args.force or args.repeat > 1
    if gen_flag:
        ref_acc_all = []
        ious_all = []
        masks_all = []
        others_all = []
        lang_acc_all = []
        for seed in seeds:
            # reproducibility
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)

            print("generating the scores for seed {}...".format(seed))
            ref_acc = []
            ious = []
            masks = []
            others = []
            lang_acc = []
            predictions = {}
            
            ######################################################################################################
            #data_dict = {}
            ######################################################################################################

            for data in tqdm(dataloader):

                ######################################################################################################
                ref_realted = [  
                    'lang_feat','lang_len','ref_box_label','ref_center_label','ref_heading_class_label',
                    'ref_heading_residual_label','ref_size_class_label','ref_size_residual_label',
                    'object_id','ann_id','object_cat','unique_multiple'
                            ]
                
                for label in ref_realted:
                    data[label] = data[label].squeeze(0)
                ######################################################################################################

                for key in data:
                    data[key] = data[key].to(device=args.device)

                # feed
                data = model(data)
                _, data = get_loss(
                    data_dict=data, 
                    config=DC, 
                    device=args.device,
                    detection=args.detection,
                    reference=True, 
                    use_lang_classifier=not args.no_lang_cls
                )
                data = get_eval(
                    data_dict=data, 
                    config=DC,
                    reference=True, 
                    use_lang_classifier=not args.no_lang_cls,
                    use_oracle=args.use_oracle,
                    use_cat_rand=args.use_cat_rand,
                    use_best=args.use_best,
                    post_processing=POST_DICT
                )

                ref_acc += data["ref_acc"]
                ious += data["ref_iou"]
                masks += data["ref_multiple_mask"]
                others += data["ref_others_mask"]
                lang_acc.append(data["lang_acc"].item())

                ######################################################################################################
                # store predictions
                scan_idx = data["scan_idx"].detach().cpu().numpy()[0]
                scene_id = scene_list[scan_idx]
 
                for idx in range(len(data["object_id"])):

                    object_id = data["object_id"][idx]
                    ann_id = data["ann_id"][idx]

                    if scene_id not in predictions:
                        predictions[scene_id] = {}

                    if object_id not in predictions[scene_id]:
                        predictions[scene_id][object_id] = {}

                    if ann_id not in predictions[scene_id][object_id]:
                        predictions[scene_id][object_id][ann_id] = {}

                    predictions[scene_id][object_id][ann_id]["pred_bbox"] = data["pred_bboxes"][idx]
                    predictions[scene_id][object_id][ann_id]["gt_bbox"] = data["gt_bboxes"][idx]
                    predictions[scene_id][object_id][ann_id]["iou"] = data["ref_iou"][idx] 
                
                ######################################################################################################

                ######################################################################################################
                if args.use_scenes_data:
                    save = ['aggregated_vote_xyz','aggregated_vote_features','aggregated_vote_inds','generated_proposals',
                            'seed_inds','seed_xyz','seed_features','vote_xyz','vote_features']
                    data_dict = {}
                    for key in save:
                        data_dict[key] = data[key]

                    with open(os.path.join(CONF.PATH.OUTPUT, args.folder, 'scenes_data', "{a}_data.p".format(a = scene_id)), "wb") as f:
                        pickle.dump(data_dict, f) 
                    
                    #torch.save(data_dict,CONF.PATH.OUTPUT, args.folder, 'scenes_data_with_pt', "{a}_data.pt".format(a = scene_id))

                ######################################################################################################

            # save the last predictions
            with open(pred_path, "wb") as f:
                pickle.dump(predictions, f) 
            
            # save to global
            ref_acc_all.append(ref_acc)
            ious_all.append(ious)
            masks_all.append(masks)
            others_all.append(others)
            lang_acc_all.append(lang_acc)

        # convert to numpy array
        ref_acc = np.array(ref_acc_all)
        ious = np.array(ious_all)
        masks = np.array(masks_all)
        others = np.array(others_all)
        lang_acc = np.array(lang_acc_all)

        # save the global scores
        with open(score_path, "wb") as f:
            scores = {
                "ref_acc": ref_acc_all,
                "ious": ious_all,
                "masks": masks_all,
                "others": others_all,
                "lang_acc": lang_acc_all
            }
            pickle.dump(scores, f)

    else:
        print("loading the scores...")
        with open(score_path, "rb") as f:
            scores = pickle.load(f)

            # unpack
            ref_acc = np.array(scores["ref_acc"])
            ious = np.array(scores["ious"])
            masks = np.array(scores["masks"])
            others = np.array(scores["others"])
            lang_acc = np.array(scores["lang_acc"])

    multiple_dict = {
        "unique": 0,
        "multiple": 1
    }
    others_dict = {
        "not_in_others": 0,
        "in_others": 1
    }

    # evaluation stats
    stats = {}
    for k, v in multiple_dict.items():
        stats[k] = {}
        for k_o, v_o in others_dict.items():
            stats[k][k_o] = np.sum(np.logical_and(masks[0] == v, others[0] == v_o))

        stats[k]["overall"] = np.sum(masks[0] == v)

    stats["overall"] = {}
    for k_o, v_o in others_dict.items():
        stats["overall"][k_o] = np.sum(others[0] == v_o)
    
    stats["overall"]["overall"] = masks[0].shape[0]

    # aggregate scores
    scores = {}
    for k, v in multiple_dict.items():
        for k_o in others_dict.keys():
            ref_accs, acc_025ious, acc_05ious = [], [], []
            for i in range(masks.shape[0]):
                running_ref_acc = np.mean(ref_acc[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])]) \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_025iou = ious[i][np.logical_and(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.25)].shape[0] \
                    / ious[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_05iou = ious[i][np.logical_and(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.5)].shape[0] \
                    / ious[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0

                # store
                ref_accs.append(running_ref_acc)
                acc_025ious.append(running_acc_025iou)
                acc_05ious.append(running_acc_05iou)

            if k not in scores:
                scores[k] = {k_o: {} for k_o in others_dict.keys()}

            scores[k][k_o]["ref_acc"] = np.mean(ref_accs)
            scores[k][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
            scores[k][k_o]["acc@0.5iou"] = np.mean(acc_05ious)

        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][masks[i] == multiple_dict[k]]) if np.sum(masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.25)].shape[0] \
                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.5)].shape[0] \
                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(masks[i] == multiple_dict[k]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        scores[k]["overall"] = {}
        scores[k]["overall"]["ref_acc"] = np.mean(ref_accs)
        scores[k]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
        scores[k]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    scores["overall"] = {}
    for k_o in others_dict.keys():
        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][others[i] == others_dict[k_o]]) if np.sum(others[i] == others_dict[k_o]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.25)].shape[0] \
                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(others[i] == others_dict[k_o]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.5)].shape[0] \
                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(others[i] == others_dict[k_o]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        # aggregate
        scores["overall"][k_o] = {}
        scores["overall"][k_o]["ref_acc"] = np.mean(ref_accs)
        scores["overall"][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
        scores["overall"][k_o]["acc@0.5iou"] = np.mean(acc_05ious)
   
    ref_accs, acc_025ious, acc_05ious = [], [], []
    for i in range(masks.shape[0]):
        running_ref_acc = np.mean(ref_acc[i])
        running_acc_025iou = ious[i][ious[i] >= 0.25].shape[0] / ious[i].shape[0]
        running_acc_05iou = ious[i][ious[i] >= 0.5].shape[0] / ious[i].shape[0]

        # store
        ref_accs.append(running_ref_acc)
        acc_025ious.append(running_acc_025iou)
        acc_05ious.append(running_acc_05iou)

    # aggregate
    scores["overall"]["overall"] = {}
    scores["overall"]["overall"]["ref_acc"] = np.mean(ref_accs)
    scores["overall"]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
    scores["overall"]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    # report
    print("\nstats:")
    for k_s in stats.keys():
        for k_o in stats[k_s].keys():
            print("{} | {}: {}".format(k_s, k_o, stats[k_s][k_o]))

    for k_s in scores.keys():
        print("\n{}:".format(k_s))
        for k_m in scores[k_s].keys():
            for metric in scores[k_s][k_m].keys():
                print("{} | {} | {}: {}".format(k_s, k_m, metric, scores[k_s][k_m][metric]))

    print("\nlanguage classification accuracy: {}".format(np.mean(lang_acc)))

#-------------------------------------------------------------------------------------------------------------------------------

def eval_det(args):
    print("evaluate detection...")
    # constant
    DC = ScannetDatasetConfig()
    
    ######################################################################################################
    # init training dataset
    print("preparing data...")
    # scanrefer, scene_list = get_scanrefer(args)
    scanrefer_train, scanrefer_val, train_scene_list, val_scene_list = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes)

    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }
    
    # dataloader
    if args.use_train :
        _, dataloader = get_dataloader(args, scanrefer, train_scene_list, "train", args.device)
        scene_list = train_scene_list
    else: 
        _, dataloader = get_dataloader(args, scanrefer, val_scene_list, "val", args.device)
        scene_list = val_scene_list
    ######################################################################################################

    # model
    model = get_model(args, DC)

    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    }
    AP_IOU_THRESHOLDS = [0.25, 0.5]
    AP_CALCULATOR_LIST = [APCalculator(iou_thresh, DC.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

    sem_acc = []
    for data in tqdm(dataloader):

        ######################################################################################################
        ref_realted = [  
            'lang_feat','lang_len','ref_box_label','ref_center_label','ref_heading_class_label',
            'ref_heading_residual_label','ref_size_class_label','ref_size_residual_label',
            'object_id','ann_id','object_cat','unique_multiple'
                    ]
        
        for label in ref_realted:
            data[label] = data[label].squeeze(0)
        ######################################################################################################

        for key in data:
            data[key] = data[key].to(device=args.device)

        # feed
        with torch.no_grad():
            data = model(data)
            _, data = get_loss(
                data_dict=data, 
                config=DC, 
                device=args.device,
                detection=True,
                reference=False
            )
            data = get_eval(
                data_dict=data, 
                config=DC, 
                reference=False,
                post_processing=POST_DICT
            )

            scan_idx = data["scan_idx"].detach().cpu().numpy()[0]
            scene_id = scene_list[scan_idx]
            ######################################################################################################
            if not args.use_scenes_data:
                save = ['aggregated_vote_xyz','aggregated_vote_features','aggregated_vote_inds','generated_proposals',
                        'seed_inds','seed_xyz','seed_features','vote_xyz','vote_features']
                data_dict = {}
                for key in save:
                    data_dict[key] = data[key]

                with open(os.path.join(CONF.PATH.OUTPUT, args.folder, 'scenes_data', "{a}_data.p".format(a = scene_id)), "wb") as f:
                    pickle.dump(data_dict, f) 
                
                #torch.save(data_dict,CONF.PATH.OUTPUT, args.folder, 'scenes_data_with_pt', "{a}_data.pt".format(a = scene_id))
            ######################################################################################################

        sem_acc.append(data["sem_acc"].item())

        batch_pred_map_cls = parse_predictions(data, POST_DICT) 
        batch_gt_map_cls = parse_groundtruths(data, POST_DICT) 
        for ap_calculator in AP_CALCULATOR_LIST:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # aggregate object detection results and report
    print("\nobject detection sem_acc: {}".format(np.mean(sem_acc)))
    for i, ap_calculator in enumerate(AP_CALCULATOR_LIST):
        print()
        print("-"*10, "iou_thresh: %f"%(AP_IOU_THRESHOLDS[i]), "-"*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            print("eval %s: %f"%(key, metrics_dict[key]))

#-------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--device", type=str, help="device", default="cpu")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--force", action="store_true", help="enforce the generation of results")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times for evaluation")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
    parser.add_argument("--use_oracle", action="store_true", help="Use ground truth bounding boxes.")
    parser.add_argument("--use_cat_rand", action="store_true", help="Use randomly selected bounding boxes from correct categories as outputs.")
    parser.add_argument("--use_best", action="store_true", help="Use best bounding boxes as outputs.")
    parser.add_argument("--reference", action="store_true", help="evaluate the reference localization results")
    parser.add_argument("--detection", action="store_true", help="evaluate the object detection results")
    parser.add_argument("--no_detection", action="store_true", help="")
    parser.add_argument("--use_scenes_data", action="store_true")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    #os.environ['KMP_DUPLICATE_LIB_OK']='True' ## added by me
    
    # evaluate
    if args.reference: eval_ref(args)
    if args.detection: eval_det(args)

