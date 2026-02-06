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

from allconfig.test_config_normal import cfg_test_normal
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
from uni_test import test_4
import warnings
warnings.filterwarnings('ignore')


def gram_matrix(tensor):
    batch_size, d, h, w = tensor.size()
    tensor = tensor.view(batch_size, d, h * w)  # 转换为 (B, d, h*w)
    gram = torch.bmm(tensor, tensor.transpose(1, 2))  # 批量矩阵乘法 (B, d, d)
    return gram


def main(args):
    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    if model_type == "CID_CPN50":
        model = network.__dict__[args.model](cfg.output_shape, cfg.num_class, "/home/wjm/clone_projects/HVI-CIDNet/weights/LOLv1/w_perc.pth", args.freeze, pretrained=True)
    else:
        model = network.__dict__[args.model](cfg.output_shape, cfg.num_class, args.freeze, pretrained=True)
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion1 = torch.nn.MSELoss().cuda()  # for Global loss
    criterion2 = torch.nn.MSELoss(reduce=False).cuda()  # for refine loss
    # gram_criterion = torch.nn.MSELoss(reduction='none').cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.lr,
                                weight_decay=cfg.weight_decay,
                                momentum=0.9)
    logger = None
    if args.resume:
        if isfile(args.resume):
            print("=> loading all model checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            pretrained_dict = checkpoint['state_dict']
            model.load_state_dict(pretrained_dict)
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, exp_name + '_log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    elif args.baseline:
        if isfile(args.baseline):
            print("=> loading cpn model checkpoint '{}'".format(args.baseline))
            checkpoint = torch.load(args.baseline)
            pretrained_dict = checkpoint['state_dict']
            pretrained_dict = {
                k.replace('module.', ''): v
                for k, v in checkpoint['state_dict'].items()
            }
            model.module.cpn.load_state_dict(pretrained_dict)
            print("=> loaded CPN checkpoint '{}' (epoch {})"
                  .format(args.baseline, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, exp_name + '_log.txt'))
            logger.set_names(['Epoch', 'LR', 'Train Loss'])
        else:
            print("=> no checkpoint found at '{}'".format(args.baseline))
    else:
        logger = Logger(join(args.checkpoint, exp_name + '_log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss'])

    cudnn.benchmark = True
    print('    Total params: %.2fMB %s' % (
        (sum(p.numel() for p in model.parameters()) / (1024 * 1024) * 4), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    if model_type == "CID_CPN50":
        train_loader = torch.utils.data.DataLoader(
            LmdbDataset("/home/wjm/MyFinalProject/train2.lmdb", cfg),
            batch_size=cfg.batch_size * args.num_gpus, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            LmdbDataset(r"E:\Final\train.lmdb", cfg),
            batch_size=cfg.batch_size * args.num_gpus, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, cfg.lr_dec_epoch, cfg.lr_gamma)
        cur_time = datetime.now()
        print('\n%s Epoch: %d | LR: %.8f' % (cur_time.strftime("%Y-%m-%d %H:%M:%S"), epoch + 1, lr))

        # train for one epoch
        train_loss = train(train_loader, model, [criterion1, criterion2], optimizer)
        print('train_loss: ', train_loss)

        # append logger file
        logger.append([epoch + 1, lr, train_loss])

        save_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint=args.checkpoint, filename=f'{exp_name}-{epoch + 1}.pth.tar')

        '''
            model testing
        '''
        model.eval()
        test_loader_all = torch.utils.data.DataLoader(
            EvalLLLmdb("/ExLPose/test-ll-normal.lmdb", cfg_test_normal),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)
        result = test(test_loader_all, model, args.checkpoint)
        print(f"Epoch {epoch + 1} Test acc: {result[0]}\n")
        logger.file.write(f"Epoch {epoch + 1} Test acc: {result[0]}\n")

    logger.close()


def train(train_loader, model, criterions, optimizer):
    # prepare for refine loss
    def ohkm(loss, top_k):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(sub_loss, k=top_k, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / top_k
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    criterion1, criterion2 = criterions

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for i, (inputs, targets, valid) in enumerate(train_loader):
        input_var = torch.autograd.Variable(inputs.cuda())

        target15, target11, target9, target7 = targets
        refine_target_var = torch.autograd.Variable(target7.cuda(non_blocking=True))
        valid_var = torch.autograd.Variable(valid.cuda(non_blocking=True))

        # compute output
        if model_type == "E_CPN50":
            global_outputs, refine_output, enhanced_x = model(input_var)
        else:
            global_outputs, refine_output = model(input_var)
        score_map = refine_output.data.cpu()

        loss = 0.
        global_loss_record = 0.
        refine_loss_record = 0.

        # comput global loss and refine loss
        for global_output, label in zip(global_outputs, targets):
            num_points = global_output.size()[1]
            global_label = label * (valid > 1.1).type(torch.FloatTensor).view(-1, num_points, 1, 1)
            global_loss = criterion1(global_output, torch.autograd.Variable(global_label.cuda(non_blocking=True))) / 2.0
            loss += global_loss
            global_loss_record += global_loss.data.item()
        refine_loss = criterion2(refine_output, refine_target_var)
        refine_loss = refine_loss.mean(dim=3).mean(dim=2)
        refine_loss *= (valid_var > 0.1).type(torch.cuda.FloatTensor)
        refine_loss = ohkm(refine_loss, 8)
        loss += refine_loss
        refine_loss_record = refine_loss.data.item()

        # record loss
        losses.update(loss.data.item(), inputs.size(0))

        # compute gradient and do Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i % 100 == 0 and i != 0):
            print('iteration {} | loss: {}, global loss: {}, refine loss: {}, avg loss: {}, time: {}'
                  .format(i, loss.data.item(), global_loss_record,
                          refine_loss_record, losses.avg, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    return losses.avg


def test(test_loader, model, prefix):
    print("testing...")
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
            if model_type == "E_CPN50":
                global_outputs, refine_output, _ = model(input_var)
            else:
                global_outputs, refine_output = model(input_var)
            score_map = refine_output.data.cpu()
            score_map = score_map.numpy()
            if model_type == "E_CPN50":
                flip_global_outputs, flip_output, _ = model(flip_input_var)
            else:
                flip_global_outputs, flip_output = model(flip_input_var)
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

    f_name = os.path.join(prefix, f'{exp_name}-test-result.json')
    with open(f_name, 'w') as wf:
        json.dump(full_result, wf)

    eval_gt = COCO(cfg_test_all.ori_gt_path)
    eval_dt = eval_gt.loadRes(f_name)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    result = cocoEval.summarize()
    return result


exp_name = "freeze-cpn"
model_type = ""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint')
    parser.add_argument('--baseline', type=str, metavar='PATH',
                        help='Path to CPN checkpoint')
    parser.add_argument('--model', required=True, type=str,
                        help='Model name or path')
    parser.add_argument('--exp', type=str, required=True,
                        help='Experiment name')
    parser.add_argument('--freeze', action='store_true',
                        help='Freeze model parameters')

    parser.set_defaults(start_epoch=0, workers=0, num_gpus=1, epochs=32, checkpoint='checkpoint')
    args = parser.parse_args()  # 解析参数
    exp_name = args.exp  # 覆盖全局变量
    model_type = args.model

    main(args)
