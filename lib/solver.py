'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from lib.config import CONF
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from utils.eta import decode_eta
from pointnet2_ops.pt_utils import BNMomentumScheduler
import json

ITER_REPORT_TEMPLATE = """
-------------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[loss] train_loss: {train_loss}
[loss] train_ref_loss: {train_ref_loss}
[loss] train_lang_loss: {train_lang_loss}
[loss] train_objectness_loss: {train_objectness_loss}
[loss] train_vote_loss: {train_vote_loss}
[loss] train_box_loss: {train_box_loss}

[loss] train_lang_acc: {train_lang_acc}
[sco.] train_ref_acc: {train_ref_acc}
[sco.] train_obj_acc: {train_obj_acc}
[sco.] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[sco.] train_iou_rate_0.25: {train_iou_rate_25}, train_iou_rate_0.5: {train_iou_rate_5}

[sco.] train_iou_max_rate_0.25: {train_iou_max_rate_25}, train_iou_max_rate_0.5: {train_iou_max_rate_5}
[sco.] best25_iou_rate_0.25:{cur25_25_best}, [sco.] best25_iou_rate_0.5:{cur25_5_best}
[sco.] best5_iou_rate_0.25:{cur5_25_best}, [sco.] best5_iou_rate_0.5:{cur5_5_best}

[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_eval_time: {mean_eval_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

EPOCH_REPORT_TEMPLATE = """
---------------------------------summary---------------------------------
[train] train_loss: {train_loss}
[train] train_ref_loss: {train_ref_loss}
[train] train_lang_loss: {train_lang_loss}
[train] train_objectness_loss: {train_objectness_loss}
[train] train_vote_loss: {train_vote_loss}
[train] train_box_loss: {train_box_loss}

[train] train_lang_acc: {train_lang_acc}
[train] train_ref_acc: {train_ref_acc}
[train] train_obj_acc: {train_obj_acc}
[train] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[train] train_iou_rate_0.25: {train_iou_rate_25}, train_iou_rate_0.5: {train_iou_rate_5}
[train] train_max_iou_rate_0.25: {train_max_iou_rate_25}, train_max_iou_rate_0.5: {train_max_iou_rate_5}

[val]   val_loss: {val_loss}
[val]   val_ref_loss: {val_ref_loss}
[val]   val_lang_loss: {val_lang_loss}
[val]   val_objectness_loss: {val_objectness_loss}
[val]   val_vote_loss: {val_vote_loss}
[val]   val_box_loss: {val_box_loss}

[val]   val_lang_acc: {val_lang_acc}
[val]   val_ref_acc: {val_ref_acc}
[val]   val_obj_acc: {val_obj_acc}
[val]   val_pos_ratio: {val_pos_ratio}, val_neg_ratio: {val_neg_ratio}
[val]   val_iou_rate_0.25: {val_iou_rate_25}, val_iou_rate_0.5: {val_iou_rate_5}
[val]   val_max_iou_rate_0.25: {val_max_iou_rate_25}, val_max_iou_rate_0.5: {val_max_iou_rate_5}
"""

BEST_REPORT_TEMPLATE = """
--------------------------------------best--------------------------------------
[best] epoch: {epoch}
[loss] loss: {loss}
[loss] ref_loss: {ref_loss}
[loss] lang_loss: {lang_loss}
[loss] objectness_loss: {objectness_loss}
[loss] vote_loss: {vote_loss}
[loss] box_loss: {box_loss}
[loss] lang_acc: {lang_acc}
[sco.] ref_acc: {ref_acc}
[sco.] obj_acc: {obj_acc}
[sco.] pos_ratio: {pos_ratio}, neg_ratio: {neg_ratio}
[sco.] iou_rate_0.25: {iou_rate_25}, iou_rate_0.5: {iou_rate_5}
"""

class Solver():
    def __init__(self, args, model, config, dataloader, optimizer, stamp, val_step=10, 
    detection=True, reference=True, use_lang_classifier=True,
    lr_decay_step=None, lr_decay_rate=None, bn_decay_step=None, bn_decay_rate=None, lr_scheduler=None):

        self.epoch = 0                    # set in __call__
        self.verbose = 0                  # set in __call__

        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.stamp = stamp
        self.val_step = val_step

        self.detection = detection
        self.reference = reference
        self.use_lang_classifier = use_lang_classifier

        self.args = args
        self.lr_scheduler = lr_scheduler

        self.best = {
            "epoch": 0,
            "loss": float("inf"),
            "ref_loss": float("inf"),
            "lang_loss": float("inf"),
            "objectness_loss": float("inf"),
            "vote_loss": float("inf"),
            "box_loss": float("inf"),
            "lang_acc": -float("inf"),
            "ref_acc": -float("inf"),
            "obj_acc": -float("inf"),
            "pos_ratio": -float("inf"),
            "neg_ratio": -float("inf"),
            "iou_rate_0.25": -float("inf"),
            "iou_rate_0.5": -float("inf"),
            "max_iou_rate_0.25": -float("inf"),
            "max_iou_rate_0.5": -float("inf")
        }

        self.my_best = {'criteria_25':{"iou_rate_0.25": -float("inf"), "iou_rate_0.5": -float("inf")},
                        'criteria_5':{"iou_rate_0.25": -float("inf"), "iou_rate_0.5": -float("inf")}}

        self.my_log = {'train':{'25':[],'5':[]}, 'val':{'25':[],'5':[]}}
        # init log
        # contains all necessary info for all phases
        self.log = {
            "train": {},
            "val": {}
        }

        # tensorboard
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train"), exist_ok=True)
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"), exist_ok=True)
        self._log_writer = {
            "train": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train")),
            "val": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"))
        }

        # training log
        log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        self.log_fout = open(log_path, "a")

        eval_path = os.path.join(CONF.PATH.OUTPUT, stamp, "eval.txt")
        self.eval_fout = open(eval_path, "a")

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._total_iter = {}             # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE

        # bn scheduler
        if bn_decay_step and bn_decay_rate:
            it = -1
            start_epoch = 0
            BN_MOMENTUM_INIT = 0.5
            BN_MOMENTUM_MAX = 0.001
            bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * bn_decay_rate**(int(it / bn_decay_step)), BN_MOMENTUM_MAX)
            self.bn_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)
        else:
            self.bn_scheduler = None

    def __call__(self, epoch, verbose):
        # setting
        self.epoch = epoch
        self.verbose = verbose
        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = len(self.dataloader["val"]) * self.val_step
        # base_lr = self.lr_scheduler.get_lr()[0]
        # base_group_lr = [param['lr'] for param in self.optimizer.param_groups]
        for epoch_id in range(epoch):
            try:
                self._log("epoch {} starting...".format(epoch_id + 1))

                if self.lr_scheduler:
                    # self.lr_scheduler.step()
                    print("learning rate --> {}\n".format(self.lr_scheduler.get_lr()), flush=True)
                    # now_lr = self.lr_scheduler.get_lr()[0]
                    for (idx, param_group) in enumerate(self.optimizer.param_groups):
                        # print(param_group.keys(), '<< param key shape')
                        print('[LR Param Group]', param_group['Param_Name'], param_group['lr'], '<< should', flush=True)
                        # param_group['lr'] = base_group_lr[idx] / base_lr * now_lr

                # feed 
                self.dataloader['train'].dataset.shuffle_data()
                self._feed(self.dataloader["train"], "train", epoch_id)

                # save model
                self._log("saving last models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

                # update lr scheduler
                if self.lr_scheduler:
                    print("update learning rate --> {}\n".format(self.lr_scheduler.get_lr()))
                    self.lr_scheduler.step()

                # update bn scheduler
                if self.bn_scheduler:
                    print("update batch normalization momentum --> {}\n".format(self.bn_scheduler.lmbd(self.bn_scheduler.last_epoch)))
                    self.bn_scheduler.step()
                
            except KeyboardInterrupt:
                # finish training
                self._finish(epoch_id)
                exit()

        # finish training
        self._finish(epoch_id)

    def _log(self, info_str):
        self.log_fout.write(info_str + "\n")
        self.log_fout.flush()
        print(info_str, flush=True)

    def _log_eval(self, info_str):
        self.eval_fout.write(info_str + "\n")
        self.eval_fout.flush()
        print(info_str, flush=True)

    def _reset_log(self, phase):
        self.log[phase] = {
            # info
            "forward": [],
            "backward": [],
            "eval": [],
            "fetch": [],
            "iter_time": [],
            # loss (float, not torch.cuda.FloatTensor)
            "loss": [],
            "ref_loss": [],
            "lang_loss": [],
            "objectness_loss": [],
            "vote_loss": [],
            "box_loss": [],
            # scores (float, not torch.cuda.FloatTensor)
            "lang_acc": [],
            "ref_acc": [],
            "obj_acc": [],
            "pos_ratio": [],
            "neg_ratio": [],
            "ref_iou": [],
            "iou_rate_0.25": [],
            "iou_rate_0.5": [],
            "max_iou_rate_0.25": [],
            "max_iou_rate_0.5": []
        }

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _forward(self, data_dict):
        data_dict = self.model(data_dict)

        return data_dict

    def _backward(self):
        # optimize
        self.optimizer.zero_grad()
        self._running_log["loss"].backward()
        self.optimizer.step()

    def _compute_loss(self, data_dict):
        _, data_dict = get_loss(args=self.args,
            data_dict=data_dict,
            config=self.config,
            detection=self.detection,
            reference=self.reference, 
            use_lang_classifier=self.use_lang_classifier
        )

        # dump
        self._running_log["ref_loss"] = data_dict["ref_loss"]
        self._running_log["lang_loss"] = data_dict["lang_loss"]
        self._running_log["objectness_loss"] = data_dict["objectness_loss"]
        self._running_log["vote_loss"] = data_dict["vote_loss"]
        self._running_log["box_loss"] = data_dict["box_loss"]
        self._running_log["loss"] = data_dict["loss"]

    def _eval(self, data_dict):
        data_dict = get_eval(
            data_dict=data_dict,
            config=self.config,
            reference=self.reference,
            use_lang_classifier=self.use_lang_classifier
        )

        # dump
        self._running_log["lang_acc"] = data_dict["lang_acc"].item()
        self._running_log["ref_acc"] = np.mean(data_dict["ref_acc"])
        self._running_log["obj_acc"] = data_dict["obj_acc"].item()
        self._running_log["pos_ratio"] = data_dict["pos_ratio"].item()
        self._running_log["neg_ratio"] = data_dict["neg_ratio"].item()

        self._running_log["ref_iou"] = data_dict["ref_iou"]
        self._running_log["iou_rate_0.25"] = np.mean(data_dict["ref_iou_rate_0.25"])
        self._running_log["iou_rate_0.5"] = np.mean(data_dict["ref_iou_rate_0.5"])
        self._running_log["max_iou_rate_0.25"] = np.mean(data_dict["max_iou_rate_0.25"])
        self._running_log["max_iou_rate_0.5"] = np.mean(data_dict["max_iou_rate_0.5"])

    def _feed(self, dataloader, phase, epoch_id):
        # switch mode
        self._set_phase(phase)

        # re-init log
        self._reset_log(phase)

        # change dataloader
        #dataloader = dataloader if phase == "train" else tqdm(dataloader)
        dataloader = tqdm(dataloader)
        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                data_dict[key] = data_dict[key].cuda()

            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                "ref_loss": 0,
                "lang_loss": 0,
                "objectness_loss": 0,
                "vote_loss": 0,
                "box_loss": 0,
                # acc
                "lang_acc": 0,
                "ref_acc": 0,
                "obj_acc": 0,
                "pos_ratio": 0,
                "neg_ratio": 0,
                "ref_iou": 0,
                "iou_rate_0.25": 0,
                "iou_rate_0.5": 0,
                "max_iou_rate_0.25": 0,
                "max_iou_rate_0.5": 0
            }

            # load
            self.log[phase]["fetch"].append(data_dict["load_time"].sum().item())

            # with torch.autograd.set_detect_anomaly(True):
            # forward
            data_dict["epoch_id"] = epoch_id
            start = time.time()
            data_dict = self._forward(data_dict)
            self._compute_loss(data_dict)
            self.log[phase]["forward"].append(time.time() - start)

            # backward
            if phase == "train":
                start = time.time()
                self._backward()
                self.log[phase]["backward"].append(time.time() - start)
            
            # eval
            start = time.time()
            self._eval(data_dict)
            self.log[phase]["eval"].append(time.time() - start)

            # record log
            self.log[phase]["loss"].append(self._running_log["loss"].item())
            self.log[phase]["ref_loss"].append(self._running_log["ref_loss"].item())
            self.log[phase]["lang_loss"].append(self._running_log["lang_loss"].item())
            self.log[phase]["objectness_loss"].append(self._running_log["objectness_loss"].item())
            self.log[phase]["vote_loss"].append(self._running_log["vote_loss"].item())
            self.log[phase]["box_loss"].append(self._running_log["box_loss"].item())

            self.log[phase]["lang_acc"].append(self._running_log["lang_acc"])
            self.log[phase]["ref_acc"].append(self._running_log["ref_acc"])
            self.log[phase]["obj_acc"].append(self._running_log["obj_acc"])
            self.log[phase]["pos_ratio"].append(self._running_log["pos_ratio"])
            self.log[phase]["neg_ratio"].append(self._running_log["neg_ratio"])

            self.log[phase]["ref_iou"] += self._running_log["ref_iou"]
            self.log[phase]["iou_rate_0.25"].append(self._running_log["iou_rate_0.25"])
            self.log[phase]["iou_rate_0.5"].append(self._running_log["iou_rate_0.5"])
            self.log[phase]["max_iou_rate_0.25"].append(self._running_log["max_iou_rate_0.25"])
            self.log[phase]["max_iou_rate_0.5"].append(self._running_log["max_iou_rate_0.5"])

            # report
            if phase == "train":
                iter_time = self.log[phase]["fetch"][-1]
                iter_time += self.log[phase]["forward"][-1]
                iter_time += self.log[phase]["backward"][-1]
                iter_time += self.log[phase]["eval"][-1]
                self.log[phase]["iter_time"].append(iter_time)
                if (self._global_iter_id + 1) % self.verbose == 0:
                    self._train_report(epoch_id)
                    ious = self.log[phase]["ref_iou"]
                    train_iou_25 = round((np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]),5)
                    train_iou_5 = round((np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]),5)
                    self.my_log['train']['25'].append(train_iou_25)
                    self.my_log['train']['5'].append(train_iou_5)

                # evaluation
                if self._global_iter_id % self.val_step == 0 and self._global_iter_id != 0:
                    print("evaluating...")
                    # val
                    self._feed(self.dataloader["val"], "val", epoch_id)
                    self._dump_log("val")
                    self._set_phase("train")
                    self._epoch_report(epoch_id)

                # dump log
                if self._global_iter_id % 50 == 0:
                    self._dump_log("train")
                self._global_iter_id += 1


        # check best
        if phase == "val":
            
            ious = self.log[phase]["ref_iou"]
            cur_criterion = "iou_rate_0.5"
            cur_criterion_25 = "iou_rate_0.25"

            #cur_best = np.mean(self.log[phase][cur_criterion])
            #cur_best_25 = np.mean(self.log[phase][cur_criterion_25])

            cur_best_25 = np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]
            cur_best = np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]
            
            self.my_log['val']['25'].append(cur_best_25)
            self.my_log['val']['5'].append(cur_best)

            if cur_best + cur_best_25 > self.best[cur_criterion] + self.best[cur_criterion_25]:
                self._log("best {} achieved: {}".format(cur_criterion, cur_best))
                self._log("best {} achieved: {}".format(cur_criterion_25, cur_best_25))
                self._log("current train_loss: {}".format(np.mean(self.log["train"]["loss"])))
                self._log("current val_loss: {}".format(np.mean(self.log["val"]["loss"])))
                self.best["epoch"] = epoch_id + 1
                self.best["loss"] = np.mean(self.log[phase]["loss"])
                self.best["ref_loss"] = np.mean(self.log[phase]["ref_loss"])
                self.best["lang_loss"] = np.mean(self.log[phase]["lang_loss"])
                self.best["objectness_loss"] = np.mean(self.log[phase]["objectness_loss"])
                self.best["vote_loss"] = np.mean(self.log[phase]["vote_loss"])
                self.best["box_loss"] = np.mean(self.log[phase]["box_loss"])
                self.best["lang_acc"] = np.mean(self.log[phase]["lang_acc"])
                self.best["ref_acc"] = np.mean(self.log[phase]["ref_acc"])
                self.best["obj_acc"] = np.mean(self.log[phase]["obj_acc"])
                self.best["pos_ratio"] = np.mean(self.log[phase]["pos_ratio"])
                self.best["neg_ratio"] = np.mean(self.log[phase]["neg_ratio"])

                #self.best["iou_rate_0.25"] = np.mean(self.log[phase]["iou_rate_0.25"])
                #self.best["iou_rate_0.5"] = np.mean(self.log[phase]["iou_rate_0.5"])

                #ious = self.log[phase]["ref_iou"]
                self.best["iou_rate_0.25"] = np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]
                self.best["iou_rate_0.5"] = np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]

                # save model
                self._log("saving best models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))


            #iou_5_result = np.mean(self.log[phase]["iou_rate_0.5"])
            #iou_25_result = np.mean(self.log[phase]["iou_rate_0.25"])

            #ious = self.log[phase]["ref_iou"]
            iou_5_result = np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]
            iou_25_result = np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]

            if iou_25_result > self.my_best['criteria_25']["iou_rate_0.25"]:
                self.my_best['criteria_25']['iou_rate_0.25'] = iou_25_result
                self.my_best['criteria_25']['iou_rate_0.5'] = iou_5_result

                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model_criteria_25.pth"))

            if iou_5_result > self.my_best['criteria_5']["iou_rate_0.5"]:
                self.my_best['criteria_5']['iou_rate_0.25'] = iou_25_result
                self.my_best['criteria_5']['iou_rate_0.5'] = iou_5_result

                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model_criteria_5.pth"))

            det_cur_criterion = "max_iou_rate_0.5"
            det_cur_best = np.mean(self.log[phase][det_cur_criterion])
            if det_cur_best > self.best[det_cur_criterion]:
                self.best["max_iou_rate_0.25"] = np.mean(self.log[phase]["max_iou_rate_0.25"])
                self.best["max_iou_rate_0.5"] = np.mean(self.log[phase]["max_iou_rate_0.5"])
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model_det_cur.pth"))

            model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
            with open(f"{model_root}/my_log.json",'w') as f:
                json.dump(self.my_log,f)
            
    def _dump_log(self, phase):
        log = {
            "loss": ["loss", "ref_loss", "lang_loss", "objectness_loss", "vote_loss", "box_loss"],
            "score": ["lang_acc", "ref_acc", "obj_acc", "pos_ratio", "neg_ratio", "iou_rate_0.25", "iou_rate_0.5", "max_iou_rate_0.25", "max_iou_rate_0.5"]
        }
        for key in log:
            for item in log[key]:
                self._log_writer[phase].add_scalar(
                    "{}/{}".format(key, item),
                    np.mean([v for v in self.log[phase][item]]),
                    self._global_iter_id
                )

    def _finish(self, epoch_id):
        # print best
        self._best_report()

        # save check point
        self._log("saving checkpoint...\n")
        save_dict = {
            "epoch": epoch_id,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        checkpoint_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(save_dict, os.path.join(checkpoint_root, "checkpoint.tar"))

        # save model
        self._log("saving last models...\n")
        model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

        # export
        for phase in ["train", "val"]:
            self._log_writer[phase].export_scalars_to_json(os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))

    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = self.log["train"]["fetch"]
        forward_time = self.log["train"]["forward"]
        backward_time = self.log["train"]["backward"]
        eval_time = self.log["train"]["eval"]
        iter_time = self.log["train"]["iter_time"]

        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
        eta_sec = (self._total_iter["train"] - self._global_iter_id - 1) * mean_train_time
        eta_sec += len(self.dataloader["val"]) * np.ceil(self._total_iter["train"] / self.val_step) * mean_est_val_time
        eta = decode_eta(eta_sec)

        # print report
        ious = self.log["train"]["ref_iou"]
        iter_report = self.__iter_report_template.format(
            epoch_id=epoch_id + 1,
            iter_id=self._global_iter_id + 1,
            total_iter=self._total_iter["train"],
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_ref_loss=round(np.mean([v for v in self.log["train"]["ref_loss"]]), 5),
            train_lang_loss=round(np.mean([v for v in self.log["train"]["lang_loss"]]), 5),
            train_objectness_loss=round(np.mean([v for v in self.log["train"]["objectness_loss"]]), 5),
            train_vote_loss=round(np.mean([v for v in self.log["train"]["vote_loss"]]), 5),
            train_box_loss=round(np.mean([v for v in self.log["train"]["box_loss"]]), 5),
            train_lang_acc=round(np.mean([v for v in self.log["train"]["lang_acc"]]), 5),
            train_ref_acc=round(np.mean([v for v in self.log["train"]["ref_acc"]]), 5),
            train_obj_acc=round(np.mean([v for v in self.log["train"]["obj_acc"]]), 5),
            train_pos_ratio=round(np.mean([v for v in self.log["train"]["pos_ratio"]]), 5),
            train_neg_ratio=round(np.mean([v for v in self.log["train"]["neg_ratio"]]), 5),
            
            
            #train_iou_rate_25=round(np.mean([v for v in self.log["train"]["iou_rate_0.25"]]), 5),
            #train_iou_rate_5=round(np.mean([v for v in self.log["train"]["iou_rate_0.5"]]), 5),
            
            train_iou_rate_25 = round((np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]),5),
            train_iou_rate_5 = round((np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]),5),


            train_iou_max_rate_25=round(np.mean([v for v in self.log["train"]["max_iou_rate_0.25"]]), 5),
            train_iou_max_rate_5=round(np.mean([v for v in self.log["train"]["max_iou_rate_0.5"]]), 5),

            cur25_25_best = round(self.my_best['criteria_25']["iou_rate_0.25"], 5),
            cur25_5_best = round(self.my_best['criteria_25']["iou_rate_0.5"], 5),
            
            cur5_25_best = round(self.my_best['criteria_5']["iou_rate_0.25"], 5),
            cur5_5_best = round(self.my_best['criteria_5']["iou_rate_0.5"], 5),

            mean_fetch_time=round(np.mean(fetch_time), 5),
            mean_forward_time=round(np.mean(forward_time), 5),
            mean_backward_time=round(np.mean(backward_time), 5),
            mean_eval_time=round(np.mean(eval_time), 5),
            mean_iter_time=round(np.mean(iter_time), 5),
            eta_h=eta["h"],
            eta_m=eta["m"],
            eta_s=eta["s"]
        )
        self._log(iter_report)

    def _epoch_report(self, epoch_id):
        self._log("epoch [{}/{}] done...".format(epoch_id+1, self.epoch))
        self._log_eval("epoch [{}/{}] done...".format(epoch_id + 1, self.epoch))

        train_ious = self.log["train"]["ref_iou"]
        val_ious = self.log["val"]["ref_iou"]

        epoch_report = self.__epoch_report_template.format(
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_ref_loss=round(np.mean([v for v in self.log["train"]["ref_loss"]]), 5),
            train_lang_loss=round(np.mean([v for v in self.log["train"]["lang_loss"]]), 5),
            train_objectness_loss=round(np.mean([v for v in self.log["train"]["objectness_loss"]]), 5),
            train_vote_loss=round(np.mean([v for v in self.log["train"]["vote_loss"]]), 5),
            train_box_loss=round(np.mean([v for v in self.log["train"]["box_loss"]]), 5),
            train_lang_acc=round(np.mean([v for v in self.log["train"]["lang_acc"]]), 5),
            train_ref_acc=round(np.mean([v for v in self.log["train"]["ref_acc"]]), 5),
            train_obj_acc=round(np.mean([v for v in self.log["train"]["obj_acc"]]), 5),
            train_pos_ratio=round(np.mean([v for v in self.log["train"]["pos_ratio"]]), 5),
            train_neg_ratio=round(np.mean([v for v in self.log["train"]["neg_ratio"]]), 5),
            
            #train_iou_rate_25=round(np.mean([v for v in self.log["train"]["iou_rate_0.25"]]), 5),
            #train_iou_rate_5=round(np.mean([v for v in self.log["train"]["iou_rate_0.5"]]), 5),
            
            train_iou_rate_25 = round((np.array(train_ious)[np.array(train_ious) >= 0.25].shape[0] / np.array(train_ious).shape[0]),5),
            train_iou_rate_5 = round((np.array(train_ious)[np.array(train_ious) >= 0.5].shape[0] / np.array(train_ious).shape[0]),5),

            
            train_max_iou_rate_25=round(np.mean([v for v in self.log["train"]["max_iou_rate_0.25"]]), 5),
            train_max_iou_rate_5=round(np.mean([v for v in self.log["train"]["max_iou_rate_0.5"]]), 5),
            val_loss=round(np.mean([v for v in self.log["val"]["loss"]]), 5),
            val_ref_loss=round(np.mean([v for v in self.log["val"]["ref_loss"]]), 5),
            val_lang_loss=round(np.mean([v for v in self.log["val"]["lang_loss"]]), 5),
            val_objectness_loss=round(np.mean([v for v in self.log["val"]["objectness_loss"]]), 5),
            val_vote_loss=round(np.mean([v for v in self.log["val"]["vote_loss"]]), 5),
            val_box_loss=round(np.mean([v for v in self.log["val"]["box_loss"]]), 5),
            val_lang_acc=round(np.mean([v for v in self.log["val"]["lang_acc"]]), 5),
            val_ref_acc=round(np.mean([v for v in self.log["val"]["ref_acc"]]), 5),
            val_obj_acc=round(np.mean([v for v in self.log["val"]["obj_acc"]]), 5),
            val_pos_ratio=round(np.mean([v for v in self.log["val"]["pos_ratio"]]), 5),
            val_neg_ratio=round(np.mean([v for v in self.log["val"]["neg_ratio"]]), 5),
            
            #val_iou_rate_25=round(np.mean([v for v in self.log["val"]["iou_rate_0.25"]]), 5),
            #val_iou_rate_5=round(np.mean([v for v in self.log["val"]["iou_rate_0.5"]]), 5),
            
            val_iou_rate_25 = round((np.array(val_ious)[np.array(val_ious) >= 0.25].shape[0] / np.array(val_ious).shape[0]),5),
            val_iou_rate_5 = round((np.array(val_ious)[np.array(val_ious) >= 0.5].shape[0] / np.array(val_ious).shape[0]),5),

            
            val_max_iou_rate_25=round(np.mean([v for v in self.log["val"]["max_iou_rate_0.25"]]), 5),
            val_max_iou_rate_5=round(np.mean([v for v in self.log["val"]["max_iou_rate_0.5"]]), 5),
        )
        self._log(epoch_report)
        self._log_eval(epoch_report)
    
    def _best_report(self):
        self._log("training completed...")
        best_report = self.__best_report_template.format(
            epoch=self.best["epoch"],
            loss=round(self.best["loss"], 5),
            ref_loss=round(self.best["ref_loss"], 5),
            lang_loss=round(self.best["lang_loss"], 5),
            objectness_loss=round(self.best["objectness_loss"], 5),
            vote_loss=round(self.best["vote_loss"], 5),
            box_loss=round(self.best["box_loss"], 5),
            lang_acc=round(self.best["lang_acc"], 5),
            ref_acc=round(self.best["ref_acc"], 5),
            obj_acc=round(self.best["obj_acc"], 5),
            pos_ratio=round(self.best["pos_ratio"], 5),
            neg_ratio=round(self.best["neg_ratio"], 5),
            iou_rate_25=round(self.best["iou_rate_0.25"], 5),
            iou_rate_5=round(self.best["iou_rate_0.5"], 5),
        )
        self._log(best_report)
        with open(os.path.join(CONF.PATH.OUTPUT, self.stamp, "best.txt"), "w") as f:
            f.write(best_report)
