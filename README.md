基于特权信息监督和注意力机制的多阶段姿态估计网络
(Learning using Privileged Information with Pose Refine Machine in Multi-Stage Pose Network,LPMSPN)
---
- 应用于极低光照环境下的人体姿态估计任务
- 多阶段网络模型
- 基于特权信息学习技术(LUPI)
- 姿态精炼机：轻量化注意力机制(PRM)
- 跨阶段特征聚合
- 在线困难点挖掘技术(OHKM)

模型训练所使用的数据集：[ExLPose](https://cg.postech.ac.kr/research/ExLPose/)

![LPMSPN网络结构示意图](https://github.com/Xuan-ll/LPMSPN/blob/main/images/t7.png)
***
模型在ExLPose测试集上测试效果：
![LPMSPN模型在ExLPose测试集上的测试结果](https://github.com/Xuan-ll/LPMSPN/blob/main/images/test.png)

其中，Baseline-first表示引入了基于特权信息学习策略(LUPI)的CPN网络，为ExLPose数据集的提出者所提供的基线方法。其中Baseline-first*表示本地复现的Baseline-first方法在ExLPose测试集上的最优结果。
Baseline-second表示本项目选取的第二种基线方法，即多阶段模型MSPN网络。在具体配置上，我们选取的阶段数为4，并且每个阶段的上采样过程也选取了预训练的ResNet-50网络。

