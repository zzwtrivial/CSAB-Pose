import pickle

from pycocotools.coco_custom import COCO
from pycocotools.cocoeval_custom import COCOeval

from allconfig.test_config_all import cfg_test_all
from allconfig.test_config_extreme import cfg_test_extreme
from allconfig.test_config_hard import cfg_test_hard
from allconfig.test_config_normal import cfg_test_normal
from allconfig.test_config_ricoh3 import cfg_test as cfg_ricoh
from train_ll_cpn_config import cfg

import json
import os
import cv2
from tqdm import tqdm

import torch
from utils.evaluation import AverageMeter
from utils.imutils import *
from utils.logger import Logger
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from dataloader.loader_eval_LL import EvalLLLmdb

import argparse
import json
import sys
from datetime import datetime

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
from pycocotools.coco_custom import COCO
from pycocotools.cocoeval_custom import COCOeval
from tqdm import tqdm

from dataloader.loader_eval_LL import EvalLLData, EvalLLLmdb
from dataloader.LMDB_train_ll import LmdbDataset
from dataloader.loader_ExLPoseOC import ExLPoseOC
from networks import network

from train_ll_cpn_config import cfg
from utils.evaluation import AverageMeter
from utils.imutils import *
from utils.logger import Logger
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join

import warnings

warnings.filterwarnings('ignore')
model_type = "E_CPN50"
save_name = model_type


def test(i_cfg, model, prefix, db_path):
    test_loader = torch.utils.data.DataLoader(
        ExLPoseOC(i_cfg, False),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)

    print(f"{prefix} testing...")
    full_result = []
    for i, (inputs, meta) in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs.cuda())

            flip_inputs = inputs.clone()
            for i, finp in enumerate(flip_inputs):
                finp = im_to_numpy(finp)
                finp = cv2.flip(finp, 1)
                flip_inputs[i] = im_to_torch(finp)
            flip_input_var = torch.autograd.Variable(flip_inputs.cuda())
            # compute output
            if model_type == "LSBN_CPN50":
                *_, global_outputs, refine_output = model(input_var, 0 * torch.ones(input_var.shape[0],
                                                                                    dtype=torch.long).cuda())
            elif model_type == "E_CPN50":
                global_outputs, refine_output, _ = model(input_var)
            else:
                global_outputs, refine_output = model(input_var)
            score_map = refine_output.data.cpu()
            score_map = score_map.numpy()

            if model_type == "LSBN_CPN50":
                *_, flip_global_outputs, flip_output = model(flip_input_var, 0 * torch.ones(flip_input_var.shape[0],
                                                                                            dtype=torch.long).cuda())
            elif model_type == "E_CPN50":
                flip_global_outputs, flip_output, _ = model(flip_input_var)
            else:
                flip_global_outputs, flip_output = model(flip_input_var)
            flip_score_map = flip_output.data.cpu()
            flip_score_map = flip_score_map.numpy()

            for i, fscore in enumerate(flip_score_map):
                fscore = fscore.transpose((1, 2, 0))
                fscore = cv2.flip(fscore, 1)
                fscore = list(fscore.transpose((2, 0, 1)))
                for (q, w) in i_cfg.symmetry:
                    fscore[q], fscore[w] = fscore[w], fscore[q]
                fscore = np.array(fscore)
                score_map[i] += fscore
                score_map[i] /= 2

            ids = meta['imgID'].numpy()
            det_scores = meta['det_scores']
            for b in range(inputs.size(0)):
                details = meta['augmentation_details']
                single_result_dict = {}
                single_result = []

                single_map = score_map[b]
                r0 = single_map.copy()
                r0 /= 255
                r0 += 0.5
                v_score = np.zeros(14)
                for p in range(14):
                    single_map[p] /= np.amax(single_map[p])
                    border = 10
                    dr = np.zeros(
                        (i_cfg.output_shape[0] + 2 * border, i_cfg.output_shape[1] + 2 * border))
                    dr[border:-border, border:-border] = single_map[p].copy()
                    dr = cv2.GaussianBlur(dr, (21, 21), 0)
                    lb = dr.argmax()
                    y, x = np.unravel_index(lb, dr.shape)
                    dr[y, x] = 0
                    lb = dr.argmax()
                    py, px = np.unravel_index(lb, dr.shape)
                    y -= border
                    x -= border
                    py -= border + y
                    px -= border + x
                    ln = (px ** 2 + py ** 2) ** 0.5
                    delta = 0.25
                    if ln > 1e-3:
                        x += delta * px / ln
                        y += delta * py / ln
                    x = max(0, min(x, i_cfg.output_shape[1] - 1))
                    y = max(0, min(y, i_cfg.output_shape[0] - 1))
                    resy = float(
                        (4 * y + 2) / i_cfg.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                    resx = float(
                        (4 * x + 2) / i_cfg.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                    v_score[p] = float(r0[p, int(round(y) + 1e-10), int(round(x) + 1e-10)])
                    single_result.append(resx)
                    single_result.append(resy)
                    single_result.append(1)
                if len(single_result) != 0:
                    single_result_dict['image_id'] = int(ids[b])
                    single_result_dict['category_id'] = 1
                    single_result_dict['keypoints'] = single_result
                    single_result_dict['score'] = float(det_scores[b]) * v_score.mean()
                    full_result.append(single_result_dict)

    f_name = os.path.join("uni_test/OC", f'{save_name}-ll-{prefix}.json')
    with open(f_name, 'w') as wf:
        json.dump(full_result, wf)

    eval_gt = COCO(i_cfg.ori_gt_path)
    eval_dt = eval_gt.loadRes(f_name)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    result = cocoEval.summarize()
    return result


def create_model(m_type):
    from config import cfg as cpn_cfg0
    if m_type == "LSBN_CPN50":
        return network.__dict__[m_type](cfg.output_shape, cfg.num_class, pretrained=True)
    elif m_type == "CID_CPN50":
        return network.__dict__[m_type](cfg.output_shape, cfg.num_class,
                                        "/home/wjm/clone_projects/HVI-CIDNet/weights/LOLv1/w_perc.pth",
                                        False, pretrained=True)
    elif m_type == "E_CPN50":
        return network.__dict__[m_type](cfg.output_shape, cfg.num_class, pretrained=True, active=[True, True, False])
    elif m_type == "CIConv_CPN50":
        return network.__dict__[m_type](cfg.output_shape, cfg.num_class, pretrained=False)
    else:
        return network.__dict__[m_type](cfg.output_shape, cfg.num_class, False, pretrained=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str,
                        help='Model name or path')
    parser.add_argument('--path', type=str, required=True,
                        help='Experiment name')
    parser.add_argument('--save', type=str, required=True,
                        help='Experiment name')
    args = parser.parse_args()

    save_name = args.save
    model_type = args.model
    model = create_model(args.model)
    model = torch.nn.DataParallel(model).cuda()

    model_path = args.path
    checkpoint = torch.load(model_path)
    pretrained_dict = checkpoint['state_dict']
    model.load_state_dict(pretrained_dict)
    cudnn.benchmark = True
    model.eval()

    test(cfg_ricoh, model, "", "")

    # pass
