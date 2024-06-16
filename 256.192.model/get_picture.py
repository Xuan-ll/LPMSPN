import os
import sys
sys.path.append('../')

import argparse
import torch
import torch.nn.parallel
import torch.optim
import cv2
import json
import wandb
import numpy as np
import torchvision.utils as vutils

from test_config_all import cfg_test
from test_config_normal import cfg_test_normal
from test_config_hard import cfg_test_hard
from test_config_extreme import cfg_test_extreme
from pycocotools.coco_custom import COCO
from pycocotools.cocoeval_custom import COCOeval

from utils.osutils import mkdir_p, isfile, isdir, join
from utils.imutils import im_to_numpy, im_to_torch
from utils.logger import Logger

from MSPNetworks import MSPNetwork
from DataLoader.loader_eval_LL import EvalLLData
from DataLoader.loader_eval_WL import EvalWLData
from datetime import datetime
from tqdm import tqdm

def main(args):
    model = MSPNetwork.__dict__[cfg_test.model](output_shape=cfg_test.output_shape, num_class=cfg_test.num_class, pretrained=True, stage_num=cfg_test.num_stage)
    model = torch.nn.DataParallel(model).cuda()

    # load trainning weights
    checkpoint_file = os.path.join(args.checkpoint, args.test+'.pth.tar')
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    run_name = "get picture all"
    wandb.init(project="Dark_project_MSPN", name=run_name, entity="3058174047")

    connections= [[0, 2], [1, 3], [2, 4], [3, 5], [6, 8], [8, 10], [7, 9], [9, 11], [12, 13], [0, 13], [1, 13],
                                 [6,13],[7, 13]]
    # change to evaluation mode
    model.eval()

    test_loader = torch.utils.data.DataLoader(EvalLLData(cfg_test_normal, train=False),batch_size=4*args.num_gpus, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    print('get picture of the condition of normal lighting')
    for i, (inputs, meta) in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            input_var = inputs.cuda()
            flip_inputs = inputs.clone()
            for k, finp in enumerate(flip_inputs):
                finp = im_to_numpy(finp)
                finp = cv2.flip(finp, 1)
                flip_inputs[k] = im_to_torch(finp)
            flip_input_var = flip_inputs.cuda()
            ll_idx = 0
            # compute output
            features, outputs = model(input_var, ll_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
            score_map = outputs[-1][-1].data.cpu()
            score_map = score_map.numpy()

            flip_features, flip_outputs = model(flip_input_var, ll_idx * torch.ones(flip_input_var.shape[0], dtype=torch.long).cuda())
            flip_score_map = flip_outputs[-1][-1].data.cpu()
            flip_score_map = flip_score_map.numpy()

            for j, fscore in enumerate(flip_score_map):
                fscore = fscore.transpose((1,2,0))
                fscore = cv2.flip(fscore, 1)
                fscore = list(fscore.transpose((2,0,1)))
                for (q, w) in cfg_test_normal.symmetry:
                    fscore[q], fscore[w] = fscore[w], fscore[q]
                fscore = np.array(fscore)
                score_map[j] += fscore
                score_map[j] /= 2

            for b in range(inputs.size(0)):
                if b == 0:
                    details = meta['augmentation_details']
                    single_map = score_map[b]
                    #画图
                    img_tensor = inputs[b].detach().cpu()
                    min_value, max_value = -2.11785, 2.64005
                    img_tensor = (img_tensor - 0.5) * (max_value - min_value) + min_value
                    img_np = img_tensor.numpy().transpose((1, 2, 0))
                    img_mat = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # 确保颜色空间正确
                    keypoints = list()
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
#                         resy = float((4 * y + 2) / cfg_test_normal.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
#                         resx = float((4 * x + 2) / cfg_test_normal.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                        resx = int(x*4)
                        resy = int(y*4)
                        res = [resx,resy]
                        keypoints.append(res)
                        # 使用OpenCV绘制关键点
                        cv2.circle(img_mat, (resx, resy), 3, (0,0,4), -1)
                    for connection in connections:
                        (p1, p2) = (keypoints[connection[0]], keypoints[connection[1]])
                        cv2.line(img_mat, p1, p2, (0, 4, 0), 2)
                    name = 'LL_normal'+ str(i)
                    image = wandb.Image(img_mat, caption=name)

                    wandb.log({name: image})

    test_loader = torch.utils.data.DataLoader(EvalLLData(cfg_test_hard, train=False),
                batch_size=64*args.num_gpus, shuffle=False,num_workers=args.workers, pin_memory=True)
    print('get picture of the condition of hard lighting')
    for i, (inputs, meta) in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            input_var = inputs.cuda()
            flip_inputs = inputs.clone()
            for ii, finp in enumerate(flip_inputs):
                finp = im_to_numpy(finp)
                finp = cv2.flip(finp, 1)
                flip_inputs[ii] = im_to_torch(finp)
            flip_input_var = flip_inputs.cuda()
            ll_idx = 0
            # compute output
            features, outputs = model(input_var, ll_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
            score_map = outputs[-1][-1].data.cpu()
            score_map = score_map.numpy()

            flip_features, flip_outputs = model(flip_input_var, ll_idx * torch.ones(flip_input_var.shape[0], dtype=torch.long).cuda())
            flip_score_map = flip_outputs[-1][-1].data.cpu()
            flip_score_map = flip_score_map.numpy()

            for j, fscore in enumerate(flip_score_map):
                fscore = fscore.transpose((1,2,0))
                fscore = cv2.flip(fscore, 1)
                fscore = list(fscore.transpose((2,0,1)))
                for (q, w) in cfg_test_hard.symmetry:
                    fscore[q], fscore[w] = fscore[w], fscore[q]
                fscore = np.array(fscore)
                score_map[j] += fscore
                score_map[j] /= 2

            for b in range(inputs.size(0)):
                if b == 0:
                    details = meta['augmentation_details']
                    single_map = score_map[b]

                    #画图
                    img_tensor = inputs[b].detach().cpu()
                    min_value, max_value = -2.11785, 2.64005
                    img_tensor = (img_tensor - 0.5) * (max_value - min_value) + min_value
                    img_np = img_tensor.numpy().transpose((1, 2, 0))
                    img_mat = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # 确保颜色空间正确
                    keypoints = list()
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
#                         resy = float((4 * y + 2) / cfg_test_hard.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
#                         resx = float((4 * x + 2) / cfg_test_hard.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                        resx = int(x*4)
                        resy = int(y*4)
                        res = [resx,resy]
                        keypoints.append(res)
                        # 使用OpenCV绘制关键点
                        cv2.circle(img_mat, (resx, resy), 3, (0,0,4), -1)
                    for connection in connections:
                        (p1, p2) = (keypoints[connection[0]], keypoints[connection[1]])
                        cv2.line(img_mat, p1, p2, (0, 4, 0), 2)
                    name = 'LL_hard'+ str(i)
                    image = wandb.Image(img_mat, caption=name)

                    wandb.log({name: image})




    test_loader = torch.utils.data.DataLoader(EvalLLData(cfg_test_extreme, train=False),batch_size=64*args.num_gpus, shuffle=False,
                num_workers=args.workers, pin_memory=True)

    print('get picture of the condition of extreme lighting')
    for i, (inputs, meta) in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            input_var = inputs.cuda()

            flip_inputs = inputs.clone()
            for ii, finp in enumerate(flip_inputs):
                finp = im_to_numpy(finp)
                finp = cv2.flip(finp, 1)
                flip_inputs[ii] = im_to_torch(finp)
            flip_input_var = flip_inputs.cuda()
            ll_idx = 0
            # compute output
            features, outputs = model(input_var, ll_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
            score_map = outputs[-1][-1].data.cpu()
            score_map = score_map.numpy()

            flip_features, flip_outputs = model(flip_input_var, ll_idx * torch.ones(flip_input_var.shape[0], dtype=torch.long).cuda())
            flip_score_map = flip_outputs[-1][-1].data.cpu()
            flip_score_map = flip_score_map.numpy()

            for j, fscore in enumerate(flip_score_map):
                fscore = fscore.transpose((1,2,0))
                fscore = cv2.flip(fscore, 1)
                fscore = list(fscore.transpose((2,0,1)))
                for (q, w) in cfg_test_extreme.symmetry:
                    fscore[q], fscore[w] = fscore[w], fscore[q]
                fscore = np.array(fscore)
                score_map[j] += fscore
                score_map[j] /= 2

            for b in range(inputs.size(0)):
                if b == 0:
                    details = meta['augmentation_details']
                    single_map = score_map[b]
                    r0 = single_map.copy()
                    #画图
                    img_tensor = inputs[b].detach().cpu()
                    min_value, max_value = -2.11785, 2.64005
                    img_tensor = (img_tensor - 0.5) * (max_value - min_value) + min_value
                    img_np = img_tensor.numpy().transpose((1, 2, 0))
                    img_mat = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # 确保颜色空间正确
                    keypoints = list()
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
#                         resy = float((4 * y + 2) / cfg_test_extreme.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
#                         resx = float((4 * x + 2) / cfg_test_extreme.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                        # 使用OpenCV绘制关键点
                        resx = int(x*4)
                        resy = int(y*4)
                        res = [resx,resy]
                        keypoints.append(res)
                        # 使用OpenCV绘制关键点
                        cv2.circle(img_mat, (resx, resy), 3, (0,4,0), -1)
                    for connection in connections:
                        (p1, p2) = (keypoints[connection[0]], keypoints[connection[1]])
                        cv2.line(img_mat, p1, p2, (0, 0, 4), 2)
                    name = 'LL_extreme'+ str(i)
                    image = wandb.Image(img_mat, caption=name)

                    wandb.log({name: image})


    test_loader = torch.utils.data.DataLoader(EvalLLData(cfg_test, train=False),batch_size=64*args.num_gpus, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    imgs_LL = list()
    print('get picture of the condition of all lighting')
    for i, (inputs, meta) in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            input_var = inputs.cuda()
            flip_inputs = inputs.clone()
            for ii, finp in enumerate(flip_inputs):
                finp = im_to_numpy(finp)
                finp = cv2.flip(finp, 1)
                flip_inputs[ii] = im_to_torch(finp)
            flip_input_var = flip_inputs.cuda()
            ll_idx = 0
            # compute output
            features, outputs = model(input_var, ll_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
            score_map = outputs[-1][-1].data.cpu()
            score_map = score_map.numpy()

            flip_features, flip_outputs = model(flip_input_var, ll_idx * torch.ones(flip_input_var.shape[0], dtype=torch.long).cuda())
            flip_score_map = flip_outputs[-1][-1].data.cpu()
            flip_score_map = flip_score_map.numpy()


            for j, fscore in enumerate(flip_score_map):
                fscore = fscore.transpose((1,2,0))
                fscore = cv2.flip(fscore, 1)
                fscore = list(fscore.transpose((2,0,1)))
                for (q, w) in cfg_test.symmetry:
                    fscore[q], fscore[w] = fscore[w], fscore[q]
                fscore = np.array(fscore)
                score_map[j] += fscore
                score_map[j] /= 2

            for b in range(inputs.size(0)):
                if b == 0:
                    details = meta['augmentation_details']
                    single_map = score_map[b]
                    #画图
                    img_tensor = inputs[b].detach().cpu()
                    min_value, max_value = -2.11785, 2.64005
                    img_tensor = (img_tensor - 0.5) * (max_value - min_value) + min_value
                    img_np = img_tensor.numpy().transpose((1, 2, 0))
                    img_mat = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # 确保颜色空间正确
                    keypoints = list()
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
#                         resy = float((4 * y + 2) / cfg_test.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
#                         resx = float((4 * x + 2) / cfg_test.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                        # 使用OpenCV绘制关键点
                        resx = int(x*4)
                        resy = int(y*4)
                        res = [resx,resy]
                        keypoints.append(res)
                        # 使用OpenCV绘制关键点
                        cv2.circle(img_mat, (resx, resy), 3, (0,0,4), -1)
                    for connection in connections:
                        (p1, p2) = (keypoints[connection[0]], keypoints[connection[1]])
                        cv2.line(img_mat, p1, p2, (0, 4, 0), 2)
                    imgs_LL.append(img_mat)

    test_loader = torch.utils.data.DataLoader(EvalWLData(cfg_test, train=False),batch_size=64*args.num_gpus, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    imgs_WL = list()
    print('get picture of the condition of well lighting')
    for i, (inputs, meta) in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            input_var = inputs.cuda()
            flip_inputs = inputs.clone()
            for ii, finp in enumerate(flip_inputs):
                finp = im_to_numpy(finp)
                finp = cv2.flip(finp, 1)
                flip_inputs[ii] = im_to_torch(finp)
            flip_input_var = flip_inputs.cuda()
            wl_idx = 1
            # compute output
            features, outputs = model(input_var, wl_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
            score_map = outputs[-1][-1].data.cpu()
            score_map = score_map.numpy()

            flip_features, flip_outputs = model(flip_input_var, wl_idx * torch.ones(flip_input_var.shape[0], dtype=torch.long).cuda())
            flip_score_map = flip_outputs[-1][-1].data.cpu()
            flip_score_map = flip_score_map.numpy()

            for j, fscore in enumerate(flip_score_map):
                fscore = fscore.transpose((1,2,0))
                fscore = cv2.flip(fscore, 1)
                fscore = list(fscore.transpose((2,0,1)))
                for (q, w) in cfg_test.symmetry:
                    fscore[q], fscore[w] = fscore[w], fscore[q]
                fscore = np.array(fscore)
                score_map[j] += fscore
                score_map[j] /= 2

            for b in range(inputs.size(0)):
                if b == 0:
                    details = meta['augmentation_details']
                    single_map = score_map[b]
                    #画图
                    img_tensor = inputs[b].detach().cpu()
                    min_value, max_value = -2.11785, 2.64005
                    img_tensor = (img_tensor - 0.5) * (max_value - min_value) + min_value
                    img_np = img_tensor.numpy().transpose((1, 2, 0))
                    img_mat = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # 确保颜色空间正确
                    keypoints = list()
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
#                         resy = float((4 * y + 2) / cfg_test.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
#                         resx = float((4 * x + 2) / cfg_test.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                        # 使用OpenCV绘制关键点
                        resx = int(x*4)
                        resy = int(y*4)
                        res = [resx,resy]
                        keypoints.append(res)
                        # 使用OpenCV绘制关键点
                        cv2.circle(img_mat, (resx, resy), 3, (0,0,4), -1)
                    for connection in connections:
                        (p1, p2) = (keypoints[connection[0]], keypoints[connection[1]])
                        cv2.line(img_mat, p1, p2, (0, 4, 0), 2)
                    imgs_WL.append(img_mat)

    for t in range(len(imgs_LL)):
        images = cv2.hconcat([imgs_LL[t], imgs_WL[t]])
        name = 'Compare'+ str(t)
        images = wandb.Image(images, caption=name)
        wandb.log({name: images})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch get picture')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to load checkpoint (default: checkpoint)')
    parser.add_argument('-f', '--flip', default=True, type=bool,
                        help='flip input image during test (default: True)')
    parser.add_argument('-b', '--batch', default=128, type=int,
                        help='test batch size (default: 128)')
    parser.add_argument('-t', '--test', default='CPN256x192', type=str,
                        help='using which checkpoint to be tested (default: CPN256x192')
    parser.add_argument('-r', '--result', default='result', type=str,
                        help='path to save save result (default: result)')
    main(parser.parse_args())

