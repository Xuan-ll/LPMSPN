import os
import os.path
import sys
import numpy as np

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')

    model = 'MSPN_lsbn_50'
    num_stage = 4

    lr = 5e-4
    lr_gamma = 0.5
    lr_dec_epoch = list(range(6,40,6))
    base_weight_factor = 0.5

    has_ohkm = True
    topk=8

    batch_size = 16
    weight_decay = 1e-5  #正则化，防止过拟合

    num_class = 14
    img_path = os.path.join(root_dir, '../ExLPose/')
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
    bbox_extend_factor = (0.1, 0.15) # x, y

    # data augmentation setting
    scale_factor=(0.7, 1.35)
    rot_factor=45

    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB
    data_shape = (256, 192)
    output_shape = (64, 48)
    gaussain_kernel = (7, 7)
    gk15 = (15, 15)
    gk11 = (11, 11)
    gk9 = (9, 9)
    gk7 = (7, 7)
    # stage = 4:
#     gk_0 = list()
#     gk_0_0 = (21, 21)
#     gk_0.append(gk_0_0)
#     gk_0_1 = (17, 17)
#     gk_0.append(gk_0_1)
#     gk_0_2 = (15, 15)
#     gk_0.append(gk_0_2)
#     gk_0_3 = (13, 13)
#     gk_0.append(gk_0_3)
#
#     gk_1 = list()
#     gk_1_0 = (19, 19)
#     gk_1.append(gk_1_0)
#     gk_1_1 = (15, 15)
#     gk_1.append(gk_1_1)
#     gk_1_2 = (13, 13)
#     gk_1.append(gk_1_2)
#     gk_1_3 = (11, 11)
#     gk_1.append(gk_1_3)
#
#     gk_2 = list()
#     gk_2_0 = (17, 17)
#     gk_2.append(gk_2_0)
#     gk_2_1 = (13, 13)
#     gk_2.append(gk_2_1)
#     gk_2_2 = (11, 11)
#     gk_2.append(gk_2_2)
#     gk_2_3 = (9, 9)
#     gk_2.append(gk_2_3)
#
#     gk_3 = list()
#     gk_3_0 = (15, 15)
#     gk_3.append(gk_3_0)
#     gk_3_1 = (11, 11)
#     gk_3.append(gk_3_1)
#     gk_3_2 = (9, 9)
#     gk_3.append(gk_3_2)
#     gk_3_3 = (7, 7)
#     gk_3.append(gk_3_3)
#
#     gk = list()
#     gk.append(gk_0)
#     gk.append(gk_1)
#     gk.append(gk_2)
#     gk.append(gk_3)

    gt_path = os.path.join(root_dir, '../Annotations/ExLPose','ExLPose_train_trans.json')

cfg = Config()
add_pypath(cfg.root_dir)

