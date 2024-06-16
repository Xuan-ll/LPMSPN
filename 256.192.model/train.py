jimport os
import sys
sys.path.append('../')

import torch.nn.parallel
import torch.optim
import cv2
import json
import numpy as np

import argparse
import torch
import torch.backends.cudnn as cudnn
import wandb
import torchvision.utils as vutils

from test_config_all import cfg_test
from test_config_normal import cfg_test_normal
from test_config_hard import cfg_test_hard
from test_config_extreme import cfg_test_extreme
from pycocotools.coco_custom import COCO
from pycocotools.cocoeval_custom import COCOeval
from tqdm import tqdm
# from LLMSPN import MSPNetworks
from MSPNetworks import MSPNetwork
from train_config import cfg
from datetime import datetime
from DataLoader.loader_eval_LL import EvalLLData
from DataLoader.loader_eval_WL import EvalWLData
from DataLoader.loader_training_pair import TrainingData
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.misc import save_model, adjust_learning_rate, adjust_lupi_wight
from utils.imutils import im_to_numpy, im_to_torch
from utils.logger import Logger


def get_optimizer_params(modules, lr, weight_decay=0.0005, double_bias_lr=True, base_weight_factor=1):
    weights = []
    biases = []
    base_weights = []
    base_biases = []
    if isinstance(modules, list):
        for module in modules:
            for key, value in dict(module.named_parameters()).items():
                if value.requires_grad:
                    if 'fc' in key or 'score' in key:
                        if 'bias' in key:
                            biases += [value]
                        else:
                            weights += [value]
                    else:
                        if 'bias' in key:
                            base_biases += [value]
                        else:
                            base_weights += [value]
    else:
        module = modules
        for key, value in dict(module.named_parameters()).items():
            if value.requires_grad:
                if 'fc' in key or 'score' in key:
                    if 'bias' in key:
                        biases += [value]
                    else:
                        weights += [value]
                else:
                    if 'bias' in key:
                        base_biases += [value]
                    else:
                        base_weights += [value]
    if base_weight_factor:
        params = [
            {'params': weights, 'lr': lr, 'weight_decay': weight_decay},
            {'params': biases, 'lr': lr},
            {'params': base_weights, 'lr': lr * base_weight_factor, 'weight_decay': weight_decay},
            {'params': base_biases, 'lr': lr * base_weight_factor},
        ]
    else:
        params = [
            {'params': base_weights + weights, 'lr': lr, 'weight_decay': weight_decay},
            {'params': base_biases + biases, 'lr': lr },
        ]
    return params

def gram_matrix(tensor):
    d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def plot_images_to_wandb(images: list, name: str, step=None):
    # images are should be list of RGB images tensors in shape (C, H, W)
    images = vutils.make_grid(images, normalize=True, range=(-2.11785, 2.64005))

    if images.dim() == 3:
        images = images.permute(1, 2, 0)
    images = images.detach().cpu().numpy()
    images = wandb.Image(images, caption=name)

    wandb.log({name: images}, step=step)

def main(args):
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    now = datetime.now().strftime('%m-%d-%H-%M')
    print(now)

    run_name = "best4_xuan2_2"
    wandb.init(project="Dark_project_stage_4", name=run_name, entity="3058174047")


    # create model
    model = MSPNetwork.__dict__[cfg.model](output_shape=cfg.output_shape, num_class=cfg.num_class, pretrained=True, stage_num=cfg.num_stage)
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
#     criterions = list()
#     lupi_criterion = torch.nn.MSELoss().cuda()
#     criterions.append(lupi_criterion)
#     criterion = torch.nn.MSELoss().cuda()
#     criterions.append(criterion)
#     criterion = torch.nn.MSELoss(reduction='none').cuda()
#     criterions.append(criterion)
    criterions = list()
    for i in range(cfg.num_stage):
        lupi_criterion = torch.nn.MSELoss().cuda()
        criterions.append(lupi_criterion)
    for i in range(cfg.num_stage):
        if i==0:
            criterion = torch.nn.MSELoss().cuda()
            criterions.append(criterion)
        else:
            method = 'none' if cfg.has_ohkm else 'mean'
            criterion = torch.nn.MSELoss(reduction=method).cuda()
            criterions.append(criterion)

    params = get_optimizer_params(model, cfg.lr, weight_decay=cfg.weight_decay)
    optimizer = torch.optim.Adam(params)

    wandb.config = {
        "learning_rate": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "epochs": args.epochs,
        "batch_size": cfg.batch_size
    }

    if args.resume:
        # if isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint_file = os.path.join(args.checkpoint, args.resume + '.pth.tar')
        checkpoint = torch.load(checkpoint_file)
        pretrained_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_dict)
        args.start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        logger1 = Logger(join(args.checkpoint, 'log2.txt'))
        logger1.set_names(['Epoch', 'LR', 'Train Loss'])
        # args.start_epoch = 0
        # else:
        #     print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger1 = Logger(join(args.checkpoint, 'log2.txt'))
        logger1.set_names(['Epoch', 'LR', 'Train Loss'])

    cudnn.benchmark = True

    print('    Total params: %.2fMB' % (sum(p.numel() for p in model.parameters())/(1024*1024)*4))

    train_loader = torch.utils.data.DataLoader(
        TrainingData(cfg, cfg),
        batch_size=cfg.batch_size*args.num_gpus, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    #logger
    logger = Logger(join(args.checkpoint, 'train_log2.txt'))
    logger.set_names(['train_epoch', 'DataType', 'AP@IoU=0.50:0.95', 'AP@IoU=0.50', 'AP@IoU=0.75', 'AR@IoU=0.50:0.95', 'AR@IoU=0.50', 'AR@IoU=0.75'])

    length = len(train_loader)

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, cfg.lr_dec_epoch, cfg.lr_gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
        # wandb.log({"learning_rate": lr}, step=epoch)
        wandb.log({"learning_rate": lr})
        # train for one epoch
        train_loss = train(length, train_loader, model, criterions, optimizer, epoch)
        print('train_loss: ',train_loss)
        # wandb.log({"train_loss": train_loss}, step=epoch)
        now = datetime.now().strftime('%m-%d-%H-%M')
        print('\nEpoch: %d end at ' % (epoch + 1))
        print(now)
        logger1.append([epoch + 1, lr, train_loss])

        if epoch > 2:
            save_model({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, checkpoint=args.checkpoint)

            # change to evaluation mode
            model.eval()

            test_loader = torch.utils.data.DataLoader(EvalLLData(cfg_test_normal, train=False),batch_size=16*args.num_gpus, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

            print('\nEpoch: %d testing with the condition of normal lighting' %(epoch+1))
            full_result = []
            for i, (inputs, meta) in tqdm(enumerate(test_loader)):
                with torch.no_grad():
                    input_var = inputs.cuda()
                    flip_inputs = inputs.clone()
                    for j, finp in enumerate(flip_inputs):
                        finp = im_to_numpy(finp)
                        finp = cv2.flip(finp, 1)
                        flip_inputs[j] = im_to_torch(finp)
                    flip_input_var = flip_inputs.cuda()
                    ll_idx = 0
                    # compute output
                    features, outputs = model(input_var, ll_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
                    score_map = outputs[-1][-1].data.cpu()
                    score_map = score_map.numpy()

                    flip_features, flip_outputs = model(flip_input_var, ll_idx * torch.ones(flip_input_var.shape[0], dtype=torch.long).cuda())
                    flip_score_map = flip_outputs[-1][-1].data.cpu()
                    flip_score_map = flip_score_map.numpy()

                    for ii, fscore in enumerate(flip_score_map):
                        fscore = fscore.transpose((1,2,0))
                        fscore = cv2.flip(fscore, 1)
                        fscore = list(fscore.transpose((2,0,1)))
                        for (q, w) in cfg_test_normal.symmetry:
                            fscore[q], fscore[w] = fscore[w], fscore[q]
                        fscore = np.array(fscore)
                        score_map[ii] += fscore
                        score_map[ii] /= 2

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
                            dr = np.zeros((cfg_test_normal.output_shape[0] + 2*border, cfg_test_normal.output_shape[1]+2*border))
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
                            resy = float((4 * y + 2) / cfg_test_normal.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                            resx = float((4 * x + 2) / cfg_test_normal.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                            v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])       # 从原始分数图r0中获取的第p个关键点的置信度分数
                            single_result.append(resx)
                            single_result.append(resy)
                            single_result.append(1)
                        if len(single_result) != 0:
                            single_result_dict['image_id'] = int(ids[b])
                            single_result_dict['category_id'] = 1
                            single_result_dict['keypoints'] = single_result
                            single_result_dict['score'] = float(det_scores[b])*v_score.mean()
                            full_result.append(single_result_dict)

            result_path = 'result_LL_normal'
            if not isdir(result_path):
                mkdir_p(result_path)
            result_file = os.path.join(result_path, 'result.json')
            with open(result_file,'w') as wf:
                json.dump(full_result, wf)

            eval_gt = COCO(cfg_test_normal.ori_gt_path)
            eval_dt = eval_gt.loadRes(result_file)
            cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
            cocoEval.evaluate()
            cocoEval.accumulate()
            result = cocoEval.summarize()

            logger.append([epoch + 1, 1.0, result[0], result[1], result[2], result[3], result[4], result[5]])
            wandb.log({"LL_normalsplit_AP": result[0]})

            test_loader = torch.utils.data.DataLoader(EvalLLData(cfg_test_hard, train=False),
                        batch_size=16*args.num_gpus, shuffle=False,num_workers=args.workers, pin_memory=True)
            print('\ntesting with the condition of hard lighting')
            full_result = []
            for i, (inputs, meta) in tqdm(enumerate(test_loader)):
                with torch.no_grad():
                    input_var = inputs.cuda()
                    flip_inputs = inputs.clone()
                    for j, finp in enumerate(flip_inputs):
                        finp = im_to_numpy(finp)
                        finp = cv2.flip(finp, 1)
                        flip_inputs[j] = im_to_torch(finp)
                    flip_input_var = flip_inputs.cuda()
                    ll_idx = 0
                    # compute output
                    features, outputs = model(input_var, ll_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
                    score_map = outputs[-1][-1].data.cpu()
                    score_map = score_map.numpy()

                    flip_features, flip_outputs = model(flip_input_var, ll_idx * torch.ones(flip_input_var.shape[0], dtype=torch.long).cuda())
                    flip_score_map = flip_outputs[-1][-1].data.cpu()
                    flip_score_map = flip_score_map.numpy()

                    for ii, fscore in enumerate(flip_score_map):
                        fscore = fscore.transpose((1,2,0))
                        fscore = cv2.flip(fscore, 1)
                        fscore = list(fscore.transpose((2,0,1)))
                        for (q, w) in cfg_test_hard.symmetry:
                            fscore[q], fscore[w] = fscore[w], fscore[q]
                        fscore = np.array(fscore)
                        score_map[ii] += fscore
                        score_map[ii] /= 2

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
                            dr = np.zeros((cfg_test_hard.output_shape[0] + 2*border, cfg_test_hard.output_shape[1]+2*border))
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
                            resy = float((4 * y + 2) / cfg_test_hard.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                            resx = float((4 * x + 2) / cfg_test_hard.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                            v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])
                            single_result.append(resx)
                            single_result.append(resy)
                            single_result.append(1)
                        if len(single_result) != 0:
                            single_result_dict['image_id'] = int(ids[b])
                            single_result_dict['category_id'] = 1
                            single_result_dict['keypoints'] = single_result
                            single_result_dict['score'] = float(det_scores[b])*v_score.mean()
                            full_result.append(single_result_dict)

            result_path = 'result_LL_hard'
            if not isdir(result_path):
                mkdir_p(result_path)
            result_file = os.path.join(result_path, 'result.json')
            with open(result_file,'w') as wf:
                json.dump(full_result, wf)

            eval_gt = COCO(cfg_test_hard.ori_gt_path)
            eval_dt = eval_gt.loadRes(result_file)
            cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
            cocoEval.evaluate()
            cocoEval.accumulate()
            result = cocoEval.summarize()

            logger.append([epoch+1, 2.0, result[0], result[1], result[2], result[3], result[4], result[5]])
            wandb.log({"LL_hardsplit_AP": result[0]})

            test_loader = torch.utils.data.DataLoader(EvalLLData(cfg_test_extreme, train=False), batch_size=16*args.num_gpus, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

            print('\ntesting with the condition of extreme lighting')
            full_result = []
            for i, (inputs, meta) in tqdm(enumerate(test_loader)):
                with torch.no_grad():
                    input_var = inputs.cuda()

                    flip_inputs = inputs.clone()
                    for j, finp in enumerate(flip_inputs):
                        finp = im_to_numpy(finp)
                        finp = cv2.flip(finp, 1)
                        flip_inputs[j] = im_to_torch(finp)
                    flip_input_var = flip_inputs.cuda()
                    ll_idx = 0
                    # compute output
                    features, outputs = model(input_var, ll_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
                    score_map = outputs[-1][-1].data.cpu()
                    score_map = score_map.numpy()

                    flip_features, flip_outputs = model(flip_input_var, ll_idx * torch.ones(flip_input_var.shape[0], dtype=torch.long).cuda())
                    flip_score_map = flip_outputs[-1][-1].data.cpu()
                    flip_score_map = flip_score_map.numpy()

                    for ii, fscore in enumerate(flip_score_map):
                        fscore = fscore.transpose((1,2,0))
                        fscore = cv2.flip(fscore, 1)
                        fscore = list(fscore.transpose((2,0,1)))
                        for (q, w) in cfg_test_extreme.symmetry:
                            fscore[q], fscore[w] = fscore[w], fscore[q]
                        fscore = np.array(fscore)
                        score_map[ii] += fscore
                        score_map[ii] /= 2

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
                            dr = np.zeros((cfg_test_extreme.output_shape[0] + 2*border, cfg_test_extreme.output_shape[1]+2*border))
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
                            resy = float((4 * y + 2) / cfg_test_extreme.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                            resx = float((4 * x + 2) / cfg_test_extreme.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                            v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])
                            single_result.append(resx)
                            single_result.append(resy)
                            single_result.append(1)
                        if len(single_result) != 0:
                            single_result_dict['image_id'] = int(ids[b])
                            single_result_dict['category_id'] = 1
                            single_result_dict['keypoints'] = single_result
                            single_result_dict['score'] = float(det_scores[b])*v_score.mean()
                            full_result.append(single_result_dict)

            result_path = 'result_LL_extreme'
            if not isdir(result_path):
                mkdir_p(result_path)
            result_file = os.path.join(result_path, 'result.json')
            with open(result_file,'w') as wf:
                json.dump(full_result, wf)
            # evaluate on COCO

            eval_gt = COCO(cfg_test_extreme.ori_gt_path)
            eval_dt = eval_gt.loadRes(result_file)
            cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
            cocoEval.evaluate()
            cocoEval.accumulate()
            result = cocoEval.summarize()

            logger.append([epoch+1, 3.0, result[0], result[1], result[2], result[3], result[4], result[5]])
            wandb.log({"LL_extremesplit_AP": result[0]})

            test_loader = torch.utils.data.DataLoader(EvalLLData(cfg_test, train=False), batch_size=16*args.num_gpus, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

            print('\ntesting with the condition of all lighting')
            full_result = []
            for i, (inputs, meta) in tqdm(enumerate(test_loader)):
                with torch.no_grad():
                    input_var = inputs.cuda()

                    flip_inputs = inputs.clone()
                    for j, finp in enumerate(flip_inputs):
                        finp = im_to_numpy(finp)
                        finp = cv2.flip(finp, 1)
                        flip_inputs[j] = im_to_torch(finp)
                    flip_input_var = flip_inputs.cuda()
                    ll_idx = 0
                    # compute output
                    features, outputs = model(input_var, ll_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
                    score_map = outputs[-1][-1].data.cpu()
                    score_map = score_map.numpy()

                    flip_features, flip_outputs = model(flip_input_var, ll_idx * torch.ones(flip_input_var.shape[0], dtype=torch.long).cuda())
                    flip_score_map = flip_outputs[-1][-1].data.cpu()
                    flip_score_map = flip_score_map.numpy()


                    for ii, fscore in enumerate(flip_score_map):
                        fscore = fscore.transpose((1,2,0))
                        fscore = cv2.flip(fscore, 1)
                        fscore = list(fscore.transpose((2,0,1)))
                        for (q, w) in cfg_test.symmetry:
                            fscore[q], fscore[w] = fscore[w], fscore[q]
                        fscore = np.array(fscore)
                        score_map[ii] += fscore
                        score_map[ii] /= 2

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
                            dr = np.zeros((cfg_test.output_shape[0] + 2*border, cfg_test.output_shape[1]+2*border))
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
                            x = max(0, min(x, cfg_test.output_shape[1] - 1))
                            y = max(0, min(y, cfg_test.output_shape[0] - 1))
                            resy = float((4 * y + 2) / cfg_test.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                            resx = float((4 * x + 2) / cfg_test.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                            v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])
                            single_result.append(resx)
                            single_result.append(resy)
                            single_result.append(1)
                        if len(single_result) != 0:
                            single_result_dict['image_id'] = int(ids[b])
                            single_result_dict['category_id'] = 1
                            single_result_dict['keypoints'] = single_result
                            single_result_dict['score'] = float(det_scores[b])*v_score.mean()
                            full_result.append(single_result_dict)

            result_path = 'result_LL_all'
            if not isdir(result_path):
                mkdir_p(result_path)
            result_file = os.path.join(result_path, 'result.json')
            with open(result_file,'w') as wf:
                json.dump(full_result, wf)

            eval_gt = COCO(cfg_test.ori_gt_path)
            eval_dt = eval_gt.loadRes(result_file)
            cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
            cocoEval.evaluate()
            cocoEval.accumulate()
            result = cocoEval.summarize()

            logger.append([epoch+1, 4.0, result[0], result[1], result[2], result[3], result[4], result[5]])
            wandb.log({"LL_all_AP": result[0]})

            test_loader = torch.utils.data.DataLoader(EvalWLData(cfg_test, train=False), batch_size=16*args.num_gpus, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

            print('testing with the condition of well lighting')
            full_result = []
            for i, (inputs, meta) in tqdm(enumerate(test_loader)):
                with torch.no_grad():
                    input_var = inputs.cuda()
                    flip_inputs = inputs.clone()
                    for j, finp in enumerate(flip_inputs):
                        finp = im_to_numpy(finp)
                        finp = cv2.flip(finp, 1)
                        flip_inputs[j] = im_to_torch(finp)
                    flip_input_var = flip_inputs.cuda()
                    wl_idx = 1
                    # compute output
                    features, outputs = model(input_var, wl_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
                    score_map = outputs[-1][-1].data.cpu()
                    score_map = score_map.numpy()

                    flip_features, flip_outputs = model(flip_input_var, wl_idx * torch.ones(flip_input_var.shape[0], dtype=torch.long).cuda())
                    flip_score_map = flip_outputs[-1][-1].data.cpu()
                    flip_score_map = flip_score_map.numpy()

                    for ii, fscore in enumerate(flip_score_map):
                        fscore = fscore.transpose((1,2,0))
                        fscore = cv2.flip(fscore, 1)
                        fscore = list(fscore.transpose((2,0,1)))
                        for (q, w) in cfg_test.symmetry:
                            fscore[q], fscore[w] = fscore[w], fscore[q]
                        fscore = np.array(fscore)
                        score_map[ii] += fscore
                        score_map[ii] /= 2

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
                            dr = np.zeros((cfg_test.output_shape[0] + 2*border, cfg_test.output_shape[1]+2*border))
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
                            x = max(0, min(x, cfg_test.output_shape[1] - 1))
                            y = max(0, min(y, cfg_test.output_shape[0] - 1))
                            resy = float((4 * y + 2) / cfg_test.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                            resx = float((4 * x + 2) / cfg_test.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                            v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])
                            single_result.append(resx)
                            single_result.append(resy)
                            single_result.append(1)
                        if len(single_result) != 0:
                            single_result_dict['image_id'] = int(ids[b])
                            single_result_dict['category_id'] = 1
                            single_result_dict['keypoints'] = single_result
                            single_result_dict['score'] = float(det_scores[b])*v_score.mean()
                            full_result.append(single_result_dict)

            result_path = 'result_WL'
            if not isdir(result_path):
                mkdir_p(result_path)
            result_file = os.path.join(result_path, 'result.json')
            with open(result_file,'w') as wf:
                json.dump(full_result, wf)

            eval_gt = COCO(cfg_test.ori_gt_path)
            eval_dt = eval_gt.loadRes(result_file)
            cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
            cocoEval.evaluate()
            cocoEval.accumulate()
            result = cocoEval.summarize()

            logger.append([epoch+1, 5.0, result[0], result[1], result[2], result[3], result[4], result[5]])
            wandb.log({"WL_AP": result[0]})
            logger.append([0, 0, 0, 0, 0, 0, 0, 0])

def train(length, train_loader, model, criterions, optimizer, epoch):

    def ohkm(loss, top_k):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(sub_loss, k=top_k, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / top_k
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    train_loader_iter = enumerate(train_loader)
    losses = 0
    count = 0
    # switch to train mode
    model.train()

    for i in range(length):
        loss_lupi = 0
        ll_idx = 0
        wl_idx = 1

        loss = 0.
        global_loss_record = 0.
        global_loss_record_wl = 0.
        loss_lupi_record = 0.

        _, batch = train_loader_iter.__next__()
        inputs_ll, inputs_wl, targets, valid, meta = batch
        input_var = inputs_ll.cuda()
        input_var_wl = inputs_wl.cuda()

#         plot_images_to_wandb([input_var[0], input_var_wl[0]], "Comparison")
#         target15, target11, target9, target7 = targets
        valid_var = valid.cuda(non_blocking =True)

        features, outputs = model(input_var, ll_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
        features_wl, outputs_wl = model(input_var_wl, wl_idx * torch.ones(input_var_wl.shape[0], dtype=torch.long).cuda())
        for stage in range(cfg.num_stage):
            if stage == 0:
                low_features = {'layer0':features[stage][0], 'layer1':features[stage][1], 'layer2':features[stage][2], 'layer3':features[stage][3], 'layer4':features[stage][4]}
                well_features = {'layer0':features_wl[stage][0], 'layer1':features_wl[stage][1], 'layer2':features_wl[stage][2],'layer3':features_wl[stage][3], 'layer4':features_wl[stage][4]}
                lupi_weights = {'layer0':0.2, 'layer1':0.2, 'layer2':0.2, 'layer3':0.2, 'layer4':0.2}
            else:
                low_features = {'layer1': features[stage][1],'layer2': features[stage][2], 'layer3': features[stage][3],'layer4': features[stage][4]}
                well_features = {'layer1': features_wl[stage][1],'layer2': features_wl[stage][2], 'layer3': features_wl[stage][3],'layer4': features_wl[stage][4]}
                lupi_weights = {'layer1': 0.2, 'layer2': 0.2, 'layer3': 0.2, 'layer4': 0.2}
            for idx, layer in enumerate(lupi_weights):
                well_feature = well_features[layer]
                low_feature = low_features[layer]
                layer_lupi_loss = 0

                for batch_idx in range(low_feature.size(0)):
                    low_gram = gram_matrix(low_feature[batch_idx])
                    well_gram = gram_matrix(well_feature[batch_idx])
                    n,d,h,w = well_feature.size()
                    layer_lupi_loss += lupi_weights[layer]*criterions[stage](well_gram.detach(), low_gram)/(h*h*w*w)

                loss_lupi += layer_lupi_loss / 4.
        #  print('lupi loss: {}'.format(loss_lupi.data.item()))
        loss_lupi_record += loss_lupi.data.item()
        for k, output_stage in zip(range(cfg.num_stage),outputs):
            for index, (output, label) in enumerate(zip(output_stage, targets)):
                num_points = output.size()[1]
                label = label * (valid > 1.1).type(torch.FloatTensor).view(-1, num_points, 1, 1)
                if k==0:
                    global_loss = criterions[k+cfg.num_stage](output, label.cuda(non_blocking =True)) /2.0
                elif k>=1 and cfg.has_ohkm:
                    global_loss = criterions[k+cfg.num_stage](output, label.cuda(non_blocking =True)) /2.0
                    global_loss = global_loss.mean(dim=3).mean(dim=2)
                    global_loss *= (valid_var > 0.1).type(torch.cuda.FloatTensor)
                    global_loss = ohkm(global_loss, cfg.topk)
                else:
                    global_loss = criterions[k+cfg.num_stage](output, label.cuda(non_blocking =True)) /2.0
                if index < 3:
                    global_loss = global_loss/4
                loss += global_loss
                global_loss_record += global_loss.data.item()
        # print('global_loss: {}'.format(global_loss.data.item()))

        for k, output_stage_wl in zip(range(cfg.num_stage),outputs_wl):
            for index,(output, label) in enumerate(zip(output_stage_wl, targets)):
                num_points = output.size()[1]
                label = label * (valid > 1.1).type(torch.FloatTensor).view(-1, num_points, 1, 1)
                if k==0:
                    global_loss = criterions[k+cfg.num_stage](output, label.cuda(non_blocking =True)) /2.0
                elif k>=1 and cfg.has_ohkm:
                    global_loss = criterions[k+cfg.num_stage](output, label.cuda(non_blocking =True)) /2.0
                    global_loss = global_loss.mean(dim=3).mean(dim=2)
                    global_loss *= (valid_var > 0.1).type(torch.cuda.FloatTensor)
                    global_loss = ohkm(global_loss, cfg.topk)
                else:
                    global_loss = criterions[k+cfg.num_stage](output, label.cuda(non_blocking =True)) /2.0
                if index < 3:
                    global_loss = global_loss/4
                loss += global_loss
                global_loss_record_wl += global_loss.data.item()

        # print('global_loss_wl: {}'.format(global_loss.data.item()))
        lupi_w = adjust_lupi_wight(epoch)
        loss += lupi_w*loss_lupi
        wandb.log({"loss": loss.data.item()})
        wandb.log({"lupi_loss": lupi_w * loss_lupi.data.cpu().numpy()})

        # # record loss
        losses = losses + loss.data.item() * inputs_ll.size(0)
        count = count + inputs_ll.size(0)

        wandb.log({"global_loss_record_wl": global_loss_record_wl})
        wandb.log({"global_loss_record_ll": global_loss_record})

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if(i%100==0 and i!=0):
            avg = losses/count
            print('iteration {} | loss: {},lupi loss: {}, lupi_w:{}, global loss: {}, global loss wl: {}, avg loss: {}'
                .format(i, loss.data.item(), loss_lupi_record, lupi_w, global_loss_record, global_loss_record_wl, avg))

    return losses/count

NAME = 'Ours'
PATH_NAME = os.path.join('./checkpoint/', NAME)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MSPN_LSBN Training')

    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')
    parser.add_argument('--epochs', default=32, type=int, metavar='N',
                        help='number of total epochs to run (default: 32)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default=NAME, type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--file-name', default=NAME, type=str)

    main(parser.parse_args())


