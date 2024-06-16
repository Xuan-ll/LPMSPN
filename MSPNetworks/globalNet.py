import torch.nn as nn
import torch
import math

class conv_bn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, 
            has_bn=True, has_relu=True, efficient=False):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.efficient = efficient
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x
            return func 

        func = _func_factory(
                self.conv, self.bn, self.relu, self.has_bn, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x

class globalNet(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class, gen_skip=False, gen_cross_conv=False, efficient=False):
        super(globalNet, self).__init__()
        self.channel_settings = channel_settings
        laterals, upsamples, predict = [], [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            predict.append(self._predict(output_shape, num_class))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = nn.ModuleList(predict)
        self.gen_skip = gen_skip
        if self.gen_skip:
            skip1_s, skip2_s = [],[]
            for i in range(len(channel_settings)):
                skip1_s.append(conv_bn_relu(channel_settings[i], channel_settings[i], kernel_size=1,
                        stride=1, padding=0, has_bn=True, has_relu=True,
                        efficient=efficient))
                skip2_s.append(conv_bn_relu(256, channel_settings[i], kernel_size=1,
                        stride=1, padding=0, has_bn=True, has_relu=True,
                        efficient=efficient))
            self.skip1_s = nn.ModuleList(skip1_s)
            self.skip2_s = nn.ModuleList(skip2_s)
        self.gen_cross_conv = gen_cross_conv
        if self.gen_cross_conv:
            self.cross_conv = conv_bn_relu(256, 64, kernel_size=1,
                        stride=1, padding=0, has_bn=True, has_relu=True,
                        efficient=efficient)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))

        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):
        layers = []
        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(num_class))

        return nn.Sequential(*layers)


    def forward(self, x):
        global_fms, global_res = [], []
        skip1s, skip2s = [], []
        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[i](x[i])
            else:
                feature = self.laterals[i](x[i]) + up
            global_fms.append(feature)
            
            skip1 = None
            skip2 = None
            if self.gen_skip:
                skip1 = self.skip1_s[i](x[i])
                skip2 = self.skip2_s[i](feature)        # skip0:2048, skip1:1024, skip2:512, skip3:256
            skip1s.append(skip1)
            skip2s.append(skip2)

            if i != len(self.channel_settings) - 1:
                up = self.upsamples[i](feature)
           
            if i == len(self.channel_settings) - 1:
                cross_conv = None
                if self.gen_cross_conv:
                    cross_conv = self.cross_conv(feature)

            res = self.predict[i](feature)
            global_res.append(res)

        return global_res, skip1s, skip2s, cross_conv




