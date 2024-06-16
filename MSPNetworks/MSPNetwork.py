from .resnetLsbn import *                                                   
import torch.nn as nn
import torch
from .globalNet import globalNet
from .PRM import PRM

class Single_stage_module(nn.Module):

    def __init__(self, resnetLsbn, output_shape, num_class,  gen_skip=True, gen_cross_conv=True):
        super(Single_stage_module, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        self.resnet = resnetLsbn
        self.global_net = globalNet(channel_settings, output_shape=output_shape, num_class=num_class, gen_skip=gen_skip, gen_cross_conv=gen_cross_conv)
    def forward(self, x, condition_label, skip1, skip2):
        x0, x1, x2, x3, x4 = self.resnet(x, condition_label, skip1, skip2)
        res_out = [x4, x3, x2, x1]
        global_res, skip1s, skip2s, cross_conv = self.global_net(res_out)
        return x0, x1, x2, x3, x4, global_res, skip1s, skip2s, cross_conv
    

class LPMSPN(nn.Module):

    def __init__(self, resnet_first, resnet_lsbn_1, resnet_lsbn_2, resnet_lsbn_3, PRM0,PRM1,PRM2,PRM3, has_attention, output_shape, num_class, stage_num=4):
        super(LPMSPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        self.stage_num = stage_num
        self.first_stage = Single_stage_module(resnetLsbn=resnet_first, output_shape=output_shape, num_class=num_class, gen_skip=True, gen_cross_conv=True)
        self.PRM0 = PRM0
        self.PRM1 = PRM1
        self.PRM2 = PRM2
        self.PRM3 = PRM3
        self.has_attention = has_attention
        self.mspn_modules = list()
        self.resnet = list()
        self.resnet.append(resnet_lsbn_1)
        self.resnet.append(resnet_lsbn_2)
        self.resnet.append(resnet_lsbn_3)
        for i in range(self.stage_num-1):
            if i != self.stage_num - 2:
                gen_s = True
                gen_c = True
            else:
                gen_s = False
                gen_c = False
            self.mspn_modules.append(
                    Single_stage_module(resnetLsbn=self.resnet[i], output_shape=output_shape, num_class=num_class, gen_skip=gen_s, gen_cross_conv=gen_c)
                    )
            setattr(self, 'stage%d' % i, self.mspn_modules[i])
    def forward(self, img, lighting_condition):
        x0,x1,x2,x3,x4, res, skip1, skip2, x = self.first_stage(img, lighting_condition, None, None)
        feature = [x0, x1, x2, x3, x4]
        features= list()
        features.append(feature)
        outputs = list()
        if self.has_attention:
            resP = []
            r0 = self.PRM0[0](res[0])
            resP.append(r0)
            r1 = self.PRM1[0](res[1])
            resP.append(r1)
            r2 = self.PRM2[0](res[2])
            resP.append(r2)
            r3 = self.PRM3[0](res[3])
            resP.append(r3)
            outputs.append(resP)
        else:
            outputs.append(res)
        for i in range(self.stage_num-1):
            x0, x1, x2, x3, x4, res, skip1, skip2, x = eval('self.stage' + str(i))(x, lighting_condition, skip1, skip2)
            feature = [x0, x1, x2, x3, x4]
            features.append(feature)
            if self.has_attention:
                if i<1:
                    resP = []
                    r0 = self.PRM0[0](res[0])
                    resP.append(r0)
                    r1 = self.PRM1[0](res[1])
                    resP.append(r1)
                    r2 = self.PRM2[0](res[2])
                    resP.append(r2)
                    r3 = self.PRM3[0](res[3])
                    resP.append(r3)
                    outputs.append(resP)
                else:
                    resP = []
                    r0 = self.PRM0[1](res[0])
                    resP.append(r0)
                    r1 = self.PRM1[1](res[1])
                    resP.append(r1)
                    r2 = self.PRM2[1](res[2])
                    resP.append(r2)
                    r3 = self.PRM3[1](res[3])
                    resP.append(r3)
                    outputs.append(resP)
            else:
                outputs.append(res)
        return features, outputs

def MSPN_lsbn_50(output_shape, num_class,pretrained=True,stage_num=4, num_conditions=2):
    resnet_first = resnet50lsbn(pretrained=pretrained, gen_top=True, has_skip=False, num_conditions=num_conditions)
    resnet_lsbn_1 = resnet50lsbn(pretrained=pretrained, gen_top=False, has_skip=True, num_conditions=num_conditions)
    resnet_lsbn_2 = resnet50lsbn(pretrained=pretrained, gen_top=False, has_skip=True, num_conditions=num_conditions)
    resnet_lsbn_3 = resnet50lsbn(pretrained=pretrained, gen_top=False, has_skip=True, num_conditions=num_conditions)
    prm0 = nn.ModuleList()
    prm0_0 = PRM(num_class)
    prm0.append(prm0_0)
    prm0_1 = PRM(num_class)
    prm0.append(prm0_1)

    prm1 = nn.ModuleList()
    prm1_0 = PRM(num_class)
    prm1.append(prm1_0)
    prm1_1 = PRM(num_class)
    prm1.append(prm1_1)

    prm2 = nn.ModuleList()
    prm2_0 = PRM(num_class)
    prm2.append(prm2_0)
    prm2_1 = PRM(num_class)
    prm2.append(prm2_1)

    prm3 = nn.ModuleList()
    prm3_0 = PRM(num_class)
    prm3.append(prm3_0)
    prm3_1 = PRM(num_class)
    prm3.append(prm3_1)
    model = MSPN_lsbn(resnet_first=resnet_first, resnet_lsbn_1=resnet_lsbn_1, resnet_lsbn_2=resnet_lsbn_2, resnet_lsbn_3=resnet_lsbn_3,
                PRM0=prm0,  PRM1=prm1, PRM2=prm2, PRM3=prm3,  has_attention=True, output_shape=output_shape, num_class=num_class, stage_num=stage_num )
    return model

