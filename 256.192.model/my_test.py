import pickle
import logging
from pycocotools.coco_custom import COCO
from pycocotools.cocoeval_custom import COCOeval

from allconfig.test_config_all import cfg_test_all
from allconfig.test_config_extreme import cfg_test_extreme
from allconfig.test_config_hard import cfg_test_hard
from allconfig.test_config_normal import cfg_test_normal
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

from test_config_all import cfg_test_all
from dataloader.loader_eval_LL import EvalLLData, EvalLLLmdb
from dataloader.LMDB_train_ll import LmdbDataset

from networks import network

# from allconfig import cfg
from train_ll_cpn_config import cfg
from utils.evaluation import AverageMeter
from utils.imutils import *
from utils.logger import Logger
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join

import warnings

warnings.filterwarnings('ignore')


def main(args):
    # create model
    file_name = 'Ours_model'

    # wandb.init(project="dark_project", name=file_name, entity="poseindark")

    model = network.__dict__["LSBN_CPN50"](cfg_test_all.output_shape, cfg_test_all.num_class, in_features=0,
                                           num_conditions=2,
                                           pretrained=True)
    model = torch.nn.DataParallel(model).cuda()

    # load training weights
    checkpoint_file = os.path.join(args.checkpoint, args.test + '.pth.tar')
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    # change to evaluation mode
    model.eval()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    show = logging.StreamHandler()
    show.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s-%(levelname)s: %(message)s")
    show.setFormatter(formatter)
    logger.addHandler(show)

    test_loader = torch.utils.data.DataLoader(
        EvalLLLmdb("/ExLPose/test-ll-normal.lmdb", cfg_test_normal),
        batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    print('testing...')
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
            ll_idx = 0
            # compute output
            f0, f1, f2, f3, f4, global_outputs, refine_output = model(input_var, ll_idx * torch.ones(input_var.shape[0],
                                                                                                     dtype=torch.long).cuda())
            score_map = refine_output.data.cpu()
            score_map = score_map.numpy()

            f0, f1, f2, f3, f4, flip_global_outputs, flip_output = model(flip_input_var,
                                                                         ll_idx * torch.ones(flip_input_var.shape[0],
                                                                                             dtype=torch.long).cuda())
            flip_score_map = flip_output.data.cpu()
            flip_score_map = flip_score_map.numpy()

            for i, fscore in enumerate(flip_score_map):
                fscore = fscore.transpose((1, 2, 0))
                fscore = cv2.flip(fscore, 1)
                fscore = list(fscore.transpose((2, 0, 1)))
                for (q, w) in cfg_test_normal.symmetry:
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
                        (cfg_test_normal.output_shape[0] + 2 * border, cfg_test_normal.output_shape[1] + 2 * border))
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
                    x = max(0, min(x, cfg_test_normal.output_shape[1] - 1))
                    y = max(0, min(y, cfg_test_normal.output_shape[0] - 1))
                    resy = float(
                        (4 * y + 2) / cfg_test_normal.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                    resx = float(
                        (4 * x + 2) / cfg_test_normal.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
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

    result_path = 'result_LL_normal'
    if not isdir(result_path):
        mkdir_p(result_path)
    result_file = os.path.join(result_path, 'result.json')
    with open(result_file, 'w') as wf:
        json.dump(full_result, wf)

    eval_gt = COCO(cfg_test_normal.ori_gt_path)
    eval_dt = eval_gt.loadRes(result_file)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    result = cocoEval.summarize()

    # wandb.log({"LL_normalsplit_AP": result[0]})
    logger.info({"LL_normalsplit_AP": result[0]})

    test_loader = torch.utils.data.DataLoader(
        EvalLLLmdb( "/ExLPose/test-ll-hard.lmdb",cfg_test_hard),
        batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print('testing...')
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
            ll_idx = 0
            # compute output
            f0, f1, f2, f3, f4, global_outputs, refine_output = model(input_var, ll_idx * torch.ones(input_var.shape[0],
                                                                                                     dtype=torch.long).cuda())
            score_map = refine_output.data.cpu()
            score_map = score_map.numpy()

            f0, f1, f2, f3, f4, flip_global_outputs, flip_output = model(flip_input_var,
                                                                         ll_idx * torch.ones(flip_input_var.shape[0],
                                                                                             dtype=torch.long).cuda())
            flip_score_map = flip_output.data.cpu()
            flip_score_map = flip_score_map.numpy()

            for i, fscore in enumerate(flip_score_map):
                fscore = fscore.transpose((1, 2, 0))
                fscore = cv2.flip(fscore, 1)
                fscore = list(fscore.transpose((2, 0, 1)))
                for (q, w) in cfg_test_hard.symmetry:
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
                        (cfg_test_hard.output_shape[0] + 2 * border, cfg_test_hard.output_shape[1] + 2 * border))
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
                    x = max(0, min(x, cfg_test_hard.output_shape[1] - 1))
                    y = max(0, min(y, cfg_test_hard.output_shape[0] - 1))
                    resy = float(
                        (4 * y + 2) / cfg_test_hard.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                    resx = float(
                        (4 * x + 2) / cfg_test_hard.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
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

    result_path = 'result_LL_hard'
    if not isdir(result_path):
        mkdir_p(result_path)
    result_file = os.path.join(result_path, 'result.json')
    with open(result_file, 'w') as wf:
        json.dump(full_result, wf)

    eval_gt = COCO(cfg_test_hard.ori_gt_path)
    eval_dt = eval_gt.loadRes(result_file)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    result = cocoEval.summarize()
    # wandb.log({"LL_hardsplit_AP": result[0]})
    logger.info({"LL_hardsplit_AP": result[0]})

    test_loader = torch.utils.data.DataLoader(
        EvalLLLmdb("/ExLPose/test-ll-extreme.lmdb",cfg_test_extreme),
        batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print('testing...')
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
            ll_idx = 0
            # compute output
            f0, f1, f2, f3, f4, global_outputs, refine_output = model(input_var, ll_idx * torch.ones(input_var.shape[0],
                                                                                                     dtype=torch.long).cuda())
            score_map = refine_output.data.cpu()
            score_map = score_map.numpy()

            f0, f1, f2, f3, f4, flip_global_outputs, flip_output = model(flip_input_var,
                                                                         ll_idx * torch.ones(flip_input_var.shape[0],
                                                                                             dtype=torch.long).cuda())
            flip_score_map = flip_output.data.cpu()
            flip_score_map = flip_score_map.numpy()

            for i, fscore in enumerate(flip_score_map):
                fscore = fscore.transpose((1, 2, 0))
                fscore = cv2.flip(fscore, 1)
                fscore = list(fscore.transpose((2, 0, 1)))
                for (q, w) in cfg_test_extreme.symmetry:
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
                        (cfg_test_extreme.output_shape[0] + 2 * border, cfg_test_extreme.output_shape[1] + 2 * border))
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
                    x = max(0, min(x, cfg_test_extreme.output_shape[1] - 1))
                    y = max(0, min(y, cfg_test_extreme.output_shape[0] - 1))
                    resy = float(
                        (4 * y + 2) / cfg_test_extreme.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                    resx = float(
                        (4 * x + 2) / cfg_test_extreme.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
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

    result_path = 'result_LL_extreme'
    if not isdir(result_path):
        mkdir_p(result_path)
    result_file = os.path.join(result_path, 'result.json')
    with open(result_file, 'w') as wf:
        json.dump(full_result, wf)
    # evaluate on COCO

    eval_gt = COCO(cfg_test_extreme.ori_gt_path)
    eval_dt = eval_gt.loadRes(result_file)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    result = cocoEval.summarize()

    # wandb.log({"LL_extremesplit_AP": result[0]})
    logger.info({"LL_extremesplit_AP": result[0]})

    test_loader = torch.utils.data.DataLoader(
        EvalLLLmdb("/ExLPose/test-ll-all.lmdb",cfg_test_all),
        batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print('testing...')
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
            ll_idx = 0
            # compute output
            f0, f1, f2, f3, f4, global_outputs, refine_output = model(input_var, ll_idx * torch.ones(input_var.shape[0],
                                                                                                     dtype=torch.long).cuda())
            score_map = refine_output.data.cpu()
            score_map = score_map.numpy()

            f0, f1, f2, f3, f4, flip_global_outputs, flip_output = model(flip_input_var,
                                                                         ll_idx * torch.ones(flip_input_var.shape[0],
                                                                                             dtype=torch.long).cuda())
            flip_score_map = flip_output.data.cpu()
            flip_score_map = flip_score_map.numpy()

            for i, fscore in enumerate(flip_score_map):
                fscore = fscore.transpose((1, 2, 0))
                fscore = cv2.flip(fscore, 1)
                fscore = list(fscore.transpose((2, 0, 1)))
                for (q, w) in cfg_test_all.symmetry:
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
                        (cfg_test_all.output_shape[0] + 2 * border, cfg_test_all.output_shape[1] + 2 * border))
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
                    x = max(0, min(x, cfg_test_all.output_shape[1] - 1))
                    y = max(0, min(y, cfg_test_all.output_shape[0] - 1))
                    resy = float(
                        (4 * y + 2) / cfg_test_all.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                    resx = float(
                        (4 * x + 2) / cfg_test_all.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
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

    result_path = 'result_LL_all'
    if not isdir(result_path):
        mkdir_p(result_path)
    result_file = os.path.join(result_path, 'result.json')
    with open(result_file, 'w') as wf:
        json.dump(full_result, wf)

    eval_gt = COCO(cfg_test_all.ori_gt_path)
    eval_dt = eval_gt.loadRes(result_file)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    result = cocoEval.summarize()

    # wandb.log({"LL_all_AP": result[0]})
    logger.info({"LL_all_AP": result[0]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Test')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')
    parser.add_argument('-c', '--checkpoint', default='with-awd-conti', type=str, metavar='PATH',
                        help='path to load checkpoint (default: checkpoint)')
    parser.add_argument('-f', '--flip', default=True, type=bool,
                        help='flip input image during test (default: True)')
    parser.add_argument('-b', '--batch', default=8, type=int,
                        help='test batch size (default: 128)')
    parser.add_argument('-t', '--test', default='epoch24checkpoint', type=str,
                        help='using which checkpoint to be tested (default: CPN256x192')
    parser.add_argument('-r', '--result', default='result', type=str,
                        help='path to save save result (default: result)')
    main(parser.parse_args())
