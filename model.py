import time
import json

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.parameter import Parameter
from torch.nn import init
import os  # 用于与操作系统交互，如文件和目录操作
# import lap  # 线性代数库，用于解决线性方程组和矩阵运算
import torch  # PyTorch深度学习框架，用于构建和训练神经网络
import numpy  # NumPy库，用于进行高效的数值计算
import pickle  # Python标准库中的模块，用于对象序列化和反序列化
import numpy as np  # NumPy的别名，方便使用
from torch import nn  # PyTorch的神经网络模块，包含构建神经网络所需的类和函数
from tqdm import tqdm  # 用于在Python长循环中添加进度条的库
import torch.nn.functional as F  # PyTorch的函数库，包含许多预定义的神经网络层和操作
from prettytable import PrettyTable  # 用于创建和打印格式化的表格
from torch.cuda.amp import autocast  # PyTorch自动混合精度训练工具，用于提高训练效率
from torch.utils.data import Dataset, DataLoader  # PyTorch的数据集和数据加载器，用于管理训练和验证数据
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix  # scikit-learn库中的混淆矩阵计算工具
from transformers import AdamW, get_linear_schedule_with_warmup  # transformers库中的优化器和学习率调度器

try:
    from thop import profile
except ImportError:
    profile = None


def sava_data(filename, data):
    print("Begin to save data：", filename)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()


def load_data(filename):
    print("Begin to load data：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data


import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


def get_accuracy(labels, prediction):
    cm = confusion_matrix(labels, prediction)

    def linear_assignment(cost_matrix):
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(row_ind, col_ind)))

    def _make_cost_m(cm):
        s = np.max(cm)
        return s - cm  # 将最大值减去混淆矩阵，以转换为成本矩阵

    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    accuracy = np.trace(cm2) / np.sum(cm2)
    return accuracy


# def get_accuracy(labels, prediction):
#     cm = confusion_matrix(labels, prediction)
#
#     # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     def linear_assignment(cost_matrix):
#         _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
#         return np.array([[y[i], i] for i in x if i >= 0])
#
#     def _make_cost_m(cm):
#         s = np.max(cm)
#         return (- cm + s)
#
#     indexes = linear_assignment(_make_cost_m(cm))
#     js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
#     cm2 = cm[:, js]
#     accuracy = np.trace(cm2) / np.sum(cm2)
#     return accuracy


"""
    average = 'macro'
    samplewise = average == "samples"
    MCM = multilabel_confusion_matrix(
        labels,
        predictions,
        sample_weight=None,
        labels=None,
        samplewise=samplewise,
    )
"""


def get_MCM_score(labels, predictions):
    accuracy = get_accuracy(labels, predictions)
    average = 'macro'
    samplewise = average == "samples"
    MCM = multilabel_confusion_matrix(
        labels,
        predictions,
        sample_weight=None,
        labels=None,
        samplewise=samplewise,
    )
    tn = MCM[:, 0, 0]
    fp = MCM[:, 0, 1]
    fn = MCM[:, 1, 0]
    tp = MCM[:, 1, 1]
    fpr_array = fp / (fp + tn)
    fnr_array = fn / (tp + fn)
    f1_array = 2 * tp / (2 * tp + fp + fn)
    sum_array = fn + tp
    M_fpr = fpr_array.mean()
    M_fnr = fnr_array.mean()
    M_f1 = f1_array.mean()
    W_fpr = (fpr_array * sum_array).sum() / sum(sum_array)
    W_fnr = (fnr_array * sum_array).sum() / sum(sum_array)
    W_f1 = (f1_array * sum_array).sum() / sum(sum_array)
    return {
        "M_fpr": format(M_fpr * 100, '.3f'),
        "M_fnr": format(M_fnr * 100, '.3f'),
        "M_f1": format(M_f1 * 100, '.3f'),
        "W_fpr": format(W_fpr * 100, '.3f'),
        "W_fnr": format(W_fnr * 100, '.3f'),
        "W_f1": format(W_f1 * 100, '.3f'),
        "ACC": format(accuracy * 100, '.3f'),
        "MCM": MCM
    }


class TraditionalDataset(Dataset):
    def __init__(self, texts, targets, max_len, hidden_size):
        self.texts = texts.tolist() if isinstance(texts, pd.Series) else texts
        self.targets = targets.tolist() if isinstance(targets, pd.Series) else targets
        self.max_len = max_len
        self.hidden_size = hidden_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        feature = self.texts[idx]
        target = self.targets[idx]
        vectors = numpy.zeros(shape=(3, self.max_len, self.hidden_size))
        for j in range(3):
            for i in range(min(len(feature[0]), self.max_len)):
                vectors[j][i] = feature[j][i]
        return {
            'vector': vectors,
            'targets': torch.tensor(target, dtype=torch.long)
        }


class TraditionalDataset_Aug(Dataset):
    def __init__(self, texts, targets, risk_levels, vul_types, max_len, hidden_size):
        """
        初始化方法，增加了对比学习所需的风险等级和漏洞类型标签。

        参数:
        texts (list): 文本特征。
        targets (list): 原始的0/1标签，表示是否存在漏洞。
        risk_levels (list): 漏洞风险等级标签（1-5）。
        vul_types (list): 漏洞类型标签（例如 'safe'、'buffer overflow' 等）。
        max_len (int): 文本的最大长度。
        hidden_size (int): 特征向量的隐藏层维度。
        """
        self.texts = texts.tolist() if isinstance(texts, pd.Series) else texts
        self.targets = targets.tolist() if isinstance(targets, pd.Series) else targets
        self.risk_levels = risk_levels.tolist() if isinstance(risk_levels, pd.Series) else risk_levels
        self.vul_types = vul_types.tolist() if isinstance(vul_types, pd.Series) else vul_types
        self.max_len = max_len
        self.hidden_size = hidden_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        feature = self.texts[idx]
        target = self.targets[idx]
        risk_level = self.risk_levels[idx]
        vul_type = self.vul_types[idx]

        # 初始化特征向量
        vectors = numpy.zeros(shape=(3, self.max_len, self.hidden_size))
        for j in range(3):
            for i in range(min(len(feature[0]), self.max_len)):
                vectors[j][i] = feature[j][i]

        return {
            'vector': vectors,
            'targets': torch.tensor(target, dtype=torch.long),
            'risk_level': torch.tensor(int(risk_level), dtype=torch.long),  # 增加风险等级标签
            'vul_type': torch.tensor(int(vul_type), dtype=torch.long)  # 增加漏洞类型标签
        }


class SequentialConvEncoder(nn.Module):
    def __init__(self, hidden_size):
        # 初始化TextCNN模型
        super(SequentialConvEncoder, self).__init__()
        # 定义卷积核的大小，这些卷积核将用于捕捉不同大小的文本特征
        self.filter_sizes = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        # 定义每个卷积核的数量，表示每种大小的卷积核有多少个
        self.num_filters = 32
        # 分类器的dropout率，用于防止过拟合
        classifier_dropout = 0.1
        # 创建卷积层的列表，每个卷积核大小对应一个卷积层
        self.convs = nn.ModuleList(
            [nn.Conv2d(3, self.num_filters, (k, hidden_size)) for k in self.filter_sizes])
        # 定义dropout层，用于全连接层之前的特征层的输出
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义全连接层，用于将卷积层输出的特征进行分类
        # num_classes 16 32 64 256
        num_classes = 64

        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        """
        对输入进行卷积和池化操作。

        参数:
            x: 输入张量，形状为(batch_size, in_channels, sequence_length, embedding_size)，
               其中,batch_size表示批量大小，in_channels表示输入通道数，sequence_length表示句子长度，
               embedding_size表示词嵌入维度。
            conv: 卷积层对象，用于对输入进行卷积操作。

        返回:
            x: 经过卷积和池化后的张量，形状为(batch_size, num_filters)，
               其中,num_filters表示卷积核的数量。
        """
        """
        print(x.shape)
        输出结果为：[32,3,100,128]
        批量大小 batch_size = 32
        输入通道数 in_channels = 3
        句子长度 sequence_length = 100
        词嵌入维度 embedding_size = 128
        """
        # 使用 ReLU激活函数进行卷积操作，得到特征图
        x = F.relu(conv(x)).squeeze(3)
        # 对卷积结果进行最大池化操作，池化窗口为整个句子长度，将每个卷积核的输出压缩成一个值
        # print(x.shape) #[32,32,n]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # print(x.shape)            输出结果为：[batch_size, num_filters]=[32, 32]
        return x

    def forward(self, x):
        # 前向传播函数
        """
        x: 输入的特征张量，大小为 [batch_size, num_channels, sequence_length, hidden_size]

        # - batch_size: 批量大小，表示一次传入模型的样本数量。
        # - num_channels: 输入特征的通道数，对于文本数据一般为词嵌入的维度。
        # - sequence_length: 序列长度，表示每个样本中包含的词语数量或者句子长度。
        # - hidden_size: 隐藏单元的大小，表示每个词语或句子的词嵌入维度。

        输出结果是：torch.Size([32, 3, 100, 128])
        """
        out = x.float()
        # print(out.shape)              输出结果是：torch.Size([32, 3, 100, 128])

        """
        对输入进行卷积和池化操作，并将所有不同大小的卷积核的结果拼接在一起。

        参数:
            out: 输入张量，形状为(batch_size, in_channels, sequence_length, embedding_size)，
                 其中batch_size表示批量大小，in_channels表示输入通道数，sequence_length表示句子长度，
                 embedding_size表示词嵌入维度。
            self.convs: 包含不同卷积核的列表，每个卷积核对应一个Conv2d对象。

        返回:
            hidden_state: 拼接在一起的特征张量，形状为(batch_size, num_filters * len(self.filter_sizes))，
                          其中num_filters表示每个卷积核的数量，len(self.filter_sizes)表示卷积核的个数。
        """

        hidden_state = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # print(hidden_state.shape)         输出结果为：torch.Size([32, 320])

        # 对拼接后的结果进行dropout操作，以减少过拟合
        out = self.dropout(hidden_state)
        # print(out.shape)            输出结果为：torch.Size([32, 320])

        # 将dropout后的特征输入全连接层进行分类，得到最终的预测结果
        out = self.fc(out)
        """
        print(out.shape)            输出结果为：[32,2]

        batch_size = 32
        num_classes = 2

        batch_size 表示每个批次中样本的数量，num_classes 表示分类任务中的类别数量。

        """

        # 返回预测结果和隐藏状态（拼接后的特征张量），用于可选的后续处理或分析
        return out, hidden_state


# ------------------------------------------------------------------------分割线------------------------------------------------------------------------------------------
class SequentialConvAttentionEncoder(nn.Module):
    def __init__(self, hidden_size, shuffle_attention=True):
        """
        TextCNN模型的初始化方法。

        参数:
            hidden_size (int): 隐藏层大小，也就是embedding的维度。
            shuffle_attention (bool): 是否使用ShuffleAttention模块，默认为True。
        """
        super(SequentialConvAttentionEncoder, self).__init__()
        self.filter_sizes = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.num_filters = 32
        classifier_dropout = 0.1
        self.parallel_attention_enabled = shuffle_attention
        self.convs = nn.ModuleList([
            nn.Conv2d(3, self.num_filters, (k, hidden_size)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = 2
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), num_classes)
        if self.parallel_attention_enabled:
            self.parallel_attention_block = ParallelAttentionBlock(channel=self.num_filters * len(self.filter_sizes), G=8)

    def conv_and_pool(self, x, conv):
        """
        执行卷积和池化操作。

        参数:
            x (torch.Tensor): 输入张量。
            conv (torch.nn.Conv2d): 卷积层。

        返回:
            torch.Tensor: 经过卷积和池化操作后的张量。
        """
        x = F.relu(conv(x)).squeeze(3)  # 应用ReLU激活函数并去除最后一个维度
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # 最大池化并去除维度
        return x

    def forward(self, x):
        """
        定义前向传播方法。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 模型的输出。
        """
        out = x.float()
        hidden_states = []
        for conv in self.convs:
            hidden_states.append(self.conv_and_pool(out, conv))
        hidden_state = torch.cat(hidden_states, 1)  # 拼接所有卷积层的输出
        out = self.dropout(hidden_state)
        if self.parallel_attention_enabled:
            out = self.parallel_attention_block(out)
        out = self.fc(out)
        return out, hidden_state


class ParallelAttentionBlock(nn.Module):

    def __init__(self, channel=320, reduction=16, G=32):
        """
        ShuffleAttention模块的初始化方法。

        参数:
            channel (int): 输入通道数，默认为320。
            reduction (int): 降维比例，默认为16。
            G (int): 分组数量，默认为8。
        """
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))  # 分组归一化
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        定义前向传播方法。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过ShuffleAttention模块处理后的张量。
        """
        b, c, h, w = x.size()
        x = x.view(b * self.G, -1, h, w)  # 将输入张量重塑成多个子组件
        x_0, x_1 = x.chunk(2, dim=1)  # 分割通道
        x_channel = self.avg_pool(x_0)  # 通道注意力
        x_channel = self.cweight * x_channel + self.cbias
        x_channel = x_0 * self.sigmoid(x_channel)
        x_spatial = self.gn(x_1)  # 空间注意力
        x_spatial = self.sweight * x_spatial + self.sbias
        x_spatial = x_1 * self.sigmoid(x_spatial)
        out = torch.cat([x_channel, x_spatial], dim=1)  # 拼接通道和空间注意力

        # x_spatial0 = self.gn(x_0)  # 空间注意力
        # x_spatial0 = self.sweight * x_spatial0 + self.sbias
        # x_spatial0 = x_0 * self.sigmoid(x_spatial0)
        # x_spatial1 = self.gn(x_1)  # 空间注意力
        # x_spatial1 = self.sweight * x_spatial1 + self.sbias
        # x_spatial1 = x_1 * self.sigmoid(x_spatial1)

        # x_channel0 = self.avg_pool(x_0)  # 通道注意力
        # x_channel0 = self.cweight * x_channel0 + self.cbias
        # x_channel0 = x_0 * self.sigmoid(x_channel0)
        # x_channel1 = self.avg_pool(x_1)  # 通道注意力
        # x_channel1 = self.cweight * x_channel1 + self.cbias
        # x_channel1 = x_0 * self.sigmoid(x_channel1)

        # out = torch.cat([x_channel0, x_channel1], dim=1)  # 拼接通道和空间注意力
        out = out.contiguous().view(b, -1, h, w)  # 恢复形状
        out = self.channel_shuffle(out, 2)  # 通道洗牌
        return out

    @staticmethod
    def channel_shuffle(x, groups):
        """
        对输入张量进行通道洗牌。

        参数:
            x (torch.Tensor): 输入张量。
            groups (int): 分组数量。

        返回:
            torch.Tensor: 通道洗牌后的张量。
        """
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)  # 重塑张量形状
        x = x.permute(0, 2, 1, 3, 4)  # 重新排列维度
        x = x.reshape(b, -1, h, w)  # 恢复形状
        return x


# ------------------------------------------------------------------------分割线------------------------------------------------------------------------------------------
class VulSCPAugmentedModel(nn.Module):
    def __init__(self, textcnn_hidden_size, shuffle_channel=512, reduction=16, G=8, num_classes=2, num_risk_levels=5,
                 num_vul_types=10):
        super(VulSCPAugmentedModel, self).__init__()
        # 保留与第一个阶段相同的部分
        self.sequential_encoder = SequentialConvEncoder(hidden_size=textcnn_hidden_size)
        self.parallel_attention = ParallelAttentionBlock(channel=shuffle_channel, reduction=reduction, G=G)
        self.adapt_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=shuffle_channel, kernel_size=(1, 1)),
            nn.ReLU()
        )

        # 分类层
        self.fc = nn.Linear(shuffle_channel * 2, num_classes)

        # 新增的多标签学习层（风险等级和漏洞类型预测）
        self.fc_risk_level = nn.Linear(shuffle_channel * 2, num_risk_levels)
        self.fc_vul_type = nn.Linear(shuffle_channel * 2, num_vul_types)

    def forward(self, x):
        # 通过 TextCNN 获取输出
        textcnn_output, _ = self.sequential_encoder(x)

        # 调整 TextCNN 输出的维度以适配 ShuffleAttention
        adapted_textcnn_output = textcnn_output.unsqueeze(-1).unsqueeze(-1)
        adapted_textcnn_output = self.adapt_layer(adapted_textcnn_output)

        # 通过 ShuffleAttention
        shuffle_output = self.parallel_attention(adapted_textcnn_output)

        # 合并 ShuffleAttention 和 TextCNN 的输出
        combined_output = torch.cat([adapted_textcnn_output.flatten(1), shuffle_output.flatten(1)], dim=1)

        # 通过全连接层进行分类
        output_class = self.fc(combined_output)

        # 通过风险等级和漏洞类型层进行多标签预测
        output_risk_level = self.fc_risk_level(combined_output)
        output_vul_type = self.fc_vul_type(combined_output)

        # 返回分类、风险等级、漏洞类型的预测和嵌入（用于对比学习）
        return output_class, output_risk_level, output_vul_type, combined_output


# ------------------------------------------------------------------------分割线------------------------------------------------------------------------------------------

# class TextCNN(nn.Module):
#     def __init__(self, textcnn_hidden_size, shuffle_channel=512, reduction=16, G=8, num_classes=2):
#         super(TextCNN, self).__init__()
#         self.textcnn = TextCNN_0(hidden_size=textcnn_hidden_size)
#         # 注意：ShuffleAttention的通道数需要根据TextCNN输出和适配策略进行调整
#         self.shuffleattention = ShuffleAttention(channel=shuffle_channel, reduction=reduction, G=G)
#         # 适配层：将TextCNN的输出转换为ShuffleAttention可处理的形式
#         self.adapt_layer = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=shuffle_channel, kernel_size=(1, 1)),
#             nn.ReLU()
#         )
#         self.fc = nn.Linear(shuffle_channel * 2, num_classes)  # 假设输出维度合并后的处理
#         self.fc.name = 'fc'
#     def forward(self, x):
#         # 通过TextCNN
#         textcnn_output, _ = self.textcnn(x)  # 取TextCNN的输出
#
#         # 对TextCNN的输出进行维度调整，以适配ShuffleAttention
#         # 假设我们将TextCNN的输出调整为与ShuffleAttention输入匹配的形式
#         adapted_textcnn_output = textcnn_output.unsqueeze(-1).unsqueeze(-1)  # 增加两个维度以匹配[H, W]
#         adapted_textcnn_output = self.adapt_layer(adapted_textcnn_output)
#
#         # 通过ShuffleAttention
#         shuffle_output = self.shuffleattention(adapted_textcnn_output)
#
#         # 假设输出直接合并，这里需要根据实际情况进行调整
#         combined_output = torch.cat([adapted_textcnn_output.flatten(1), shuffle_output.flatten(1)], dim=1)
#
#         # 通过全连接层
#         output = self.fc(combined_output)
#
#         return output, None

class VulSCPModel(nn.Module):
    def __init__(self, textcnn_hidden_size, shuffle_channel=512, reduction=16, G=8, num_classes=2):
        super(VulSCPModel, self).__init__()
        self.sequential_encoder = SequentialConvEncoder(hidden_size=textcnn_hidden_size)
        self.sequential_encoder.name = "sequential_encoder"
        self.parallel_attention = ParallelAttentionBlock(channel=shuffle_channel, reduction=reduction, G=G)
        self.parallel_attention.name = "parallel_attention"
        self.adapt_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=shuffle_channel, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.adapt_layer.name = "adapt_layer"
        self.fc = nn.Linear(shuffle_channel * 2, num_classes)  # 分类层保持不变
        self.fc.name = 'fc'

    def forward(self, x):
        # 通过 TextCNN 获取输出
        textcnn_output, _ = self.sequential_encoder(x)

        # 调整 TextCNN 输出的维度以适配 ShuffleAttention
        adapted_textcnn_output = textcnn_output.unsqueeze(-1).unsqueeze(-1)
        adapted_textcnn_output = self.adapt_layer(adapted_textcnn_output)

        # 通过 ShuffleAttention
        shuffle_output = self.parallel_attention(adapted_textcnn_output)

        # 合并 ShuffleAttention 和 TextCNN 的输出
        combined_output = torch.cat([adapted_textcnn_output.flatten(1), shuffle_output.flatten(1)], dim=1)

        # 通过全连接层进行分类
        output = self.fc(combined_output)

        return output, combined_output  # 返回分类输出和特征向量


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_feature = features
        anchor_feature = features
        anchor_count = 1

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, 1)
        logits_mask = torch.ones_like(mask).scatter_(
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss


class VulSCPTrainer():
    def __init__(self, max_len=100, n_classes=2, epochs=100, batch_size=32, learning_rate=0.001, \
                 result_save_path=r"D:\PaperAndCode\code\VulSCP\data\results", item_num=0,
                 hidden_size=128, best_sc=0):
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.best_sc = best_sc
        result_save_path = result_save_path + "/" if result_save_path[-1] != "/" else result_save_path
        if not os.path.exists(result_save_path): os.makedirs(result_save_path)
        self.result_save_path = result_save_path + str(item_num) + "_epo" + str(epochs) + "_bat" + str(
            batch_size) + ".result"

    def preparation_Aug(self, X_train, y_train, t_vul, t_risk, X_valid, y_valid, X_test=None, y_test=None):
        # create datasets
        self.model = VulSCPModel(self.hidden_size)
        self.model.to(self.device)
        self.train_set = TraditionalDataset_Aug(X_train, y_train, t_risk, t_vul, self.max_len, self.hidden_size)
        self.valid_set = TraditionalDataset(X_valid, y_valid, self.max_len, self.hidden_size)
        self.test_set = None
        if X_test is not None and y_test is not None:
            self.test_set = TraditionalDataset(X_test, y_test, self.max_len, self.hidden_size)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False) if self.test_set is not None else None

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        self.prediction_times = []
        
        # 对于 Aug 阶段也计算一次
        try:
            print("Running Complexity Analysis (Augmentation Stage)...")
            sample = torch.randn(1, 3, self.max_len, self.hidden_size).to(self.device)
            self.analyze_complexity(sample)
        except Exception as e:
            print(f"Complexity Analysis Failed: {e}")


    def analyze_complexity(self, sample_input):
        """
        Analyze model complexity (Params, FLOPs, peak GPU memory, model file size).
        sample_input: A tensor with shape (1, 3, max_len, hidden_size)
        """
        # 为了不刷屏，只在 fold=0 的时候打印。因为网络结构每一折是一模一样的。
        if getattr(self, 'complexity_analyzed', False):
            return {}
        self.complexity_analyzed = True
        
        print("\n" + "=" * 50)
        print("[Computational Complexity Analysis (Single Model Instance)]")
        
        # 1. Total & Trainable Parameters
        # 这是绝对纯粹的只算 `self.model` (也就是您当前实例化的 PyTorch 网络) 的权重个数
        # 与您电脑上跑的 QQ、微信、甚至别的 Python 进程毫无关系
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total Parameters:      {total_params:,}")
        print(f"Trainable Parameters:  {trainable_params:,}")

        # 2. FLOPs / MACs (只针对给定的一条数据和当前模型进行前向推理计算)
        flops = 0
        if profile is not None:
            try:
                sample_input = sample_input.to(self.device)
                macs, params = profile(self.model, inputs=(sample_input, ), verbose=False)
                flops = macs * 2  # MACs 乘 2 约等于 FLOPs
                print(f"FLOPs (per sample):    {flops / 1e9:.4f} G")
                print(f"MACs (per sample):     {macs / 1e9:.4f} G")
            except Exception as e:
                print(f"FLOPs Calculation Failed: {e}")
        else:
            print("FLOPs: 'thop' library not installed.")

        # 3. Model Size (理论静态大小)
        model_size_mb = total_params * 4 / (1024 ** 2)
        print(f"Model Size (Weights):  {model_size_mb:.2f} MB")

        # 4. Peak GPU Memory (Inference / Training)
        inference_peak_mb = 0.0
        training_peak_mb = 0.0
        peak_mem_batch_size = 0
        if torch.cuda.is_available():
            candidate_batch_sizes = []
            for bs in [self.batch_size, 8, 1]:
                if bs > 0 and bs not in candidate_batch_sizes:
                    candidate_batch_sizes.append(bs)

            for bs in candidate_batch_sizes:
                try:
                    mem_sample = torch.randn(bs, 3, self.max_len, self.hidden_size, device=self.device)

                    self.model.eval()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(self.device)
                    with torch.no_grad():
                        _ = self.model(mem_sample)
                    torch.cuda.synchronize(self.device)
                    inference_peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

                    self.model.train()
                    self.model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(self.device)
                    outputs = self.model(mem_sample)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = outputs.sum()
                    loss.backward()
                    torch.cuda.synchronize(self.device)
                    training_peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

                    peak_mem_batch_size = bs
                    self.model.zero_grad(set_to_none=True)
                    del mem_sample, outputs, loss
                    torch.cuda.empty_cache()
                    break
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        torch.cuda.empty_cache()
                        continue
                    raise

            if peak_mem_batch_size > 0:
                print(f"Inference Peak GPU Mem: {inference_peak_mb:.2f} MB (batch={peak_mem_batch_size})")
                print(f"Training Peak GPU Mem:  {training_peak_mb:.2f} MB (batch={peak_mem_batch_size})")
            else:
                print("Inference Peak GPU Mem: N/A")
                print("Training Peak GPU Mem:  N/A")
        else:
            print("Inference Peak GPU Mem: CUDA not available")
            print("Training Peak GPU Mem:  CUDA not available")

        # 5. Real Model File Size (state_dict 落盘后真实文件大小)
        model_file_size_mb = 0.0
        model_file_path = None
        try:
            complexity_dir = os.path.dirname(self.result_save_path) if self.result_save_path else '.'
            os.makedirs(complexity_dir, exist_ok=True)
            model_file_path = os.path.join(complexity_dir, '_complexity_tmp_model_state_dict.pth')
            torch.save(self.model.state_dict(), model_file_path)
            model_file_size_mb = os.path.getsize(model_file_path) / (1024 ** 2)
            print(f"Model File Size:       {model_file_size_mb:.2f} MB")
        finally:
            if model_file_path and os.path.exists(model_file_path):
                os.remove(model_file_path)

        complexity_report = {
            'Total Params': int(total_params),
            'Trainable Params': int(trainable_params),
            'FLOPs (G, per sample)': float(flops / 1e9 if flops else 0.0),
            'Model Size (Weights, MB)': float(model_size_mb),
            'Inference Peak GPU Memory (MB)': float(inference_peak_mb),
            'Training Peak GPU Memory (MB)': float(training_peak_mb),
            'Peak Memory Batch Size': int(peak_mem_batch_size),
            'Model File Size (MB)': float(model_file_size_mb),
        }

        complexity_json_path = self.result_save_path.replace('.result', '_complexity.json')
        with open(complexity_json_path, 'w', encoding='utf-8') as f:
            json.dump(complexity_report, f, ensure_ascii=False, indent=2)
        print(f"Complexity JSON Saved:  {complexity_json_path}")
        print("=" * 50 + "\n")
        return complexity_report


    def fit_Aug(self):
        self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

        supcon_loss_fn = SupConLoss().to(self.device)
          # 超参数

        for i, data in progress_bar:
            self.optimizer.zero_grad()
            vectors = data["vector"].to(self.device)
            targets = data["targets"].to(self.device)
            risk_levels = data["risk_level"].to(self.device)  # 假设risk_level作为对比学习的标签
            vul_types = data["vul_type"].to(self.device)
            with autocast():
                output_class, embeddings = self.model(vectors)  # 模型输出分类和嵌入
                loss_class = self.loss_fn(output_class, targets)

                # 使用风险等级计算对比学习损失
                risk_level_contrastive_loss = supcon_loss_fn(embeddings, risk_levels)
                vul_type_contrastive_loss = supcon_loss_fn(embeddings, vul_types)
                # 总损失
                loss = loss_class # + 0.001 * risk_level_contrastive_loss # + 0.1 * vul_type_contrastive_loss

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            preds = torch.argmax(output_class, dim=1).flatten()
            losses.append(loss.item())
            predictions += list(np.array(preds.cpu()))
            labels += list(np.array(targets.cpu()))

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scheduler.step()
            progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets) / len(targets)):.3f}'
            )

        train_loss = np.mean(losses)
        score_dict = get_MCM_score(labels, predictions)
        return train_loss, score_dict

    def train_Aug(self):
        learning_record_dict = {}
        best_sc = self.best_sc
        train_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])
        test_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_loss, train_score = self.fit_Aug()
            train_table.add_row(
                ["tra", str(epoch + 1), format(train_loss, '.4f')] + [train_score[j] for j in train_score if
                                                                      j != "MCM"])
            print(train_table)

            val_loss, val_score, best_sc = self.eval(best_sc)
            print("=" * 100)
            print(best_sc)
            print("=" * 100)

            test_table.add_row(
                ["val", str(epoch + 1), format(val_loss, '.4f')] + [val_score[j] for j in val_score if j != "MCM"])
            print(test_table)
            print("\n")

            learning_record_dict[epoch] = {'train_loss': train_loss, 'val_loss': val_loss, \
                                           "train_score": train_score, "val_score": val_score}
            sava_data(self.result_save_path, learning_record_dict)
            print("\n")
        return best_sc
    def preparation(self, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None):
        # create datasets
        self.model = VulSCPModel(self.hidden_size)
        self.model.to(self.device)
        self.train_set = TraditionalDataset(X_train, y_train, self.max_len, self.hidden_size)
        self.valid_set = TraditionalDataset(X_valid, y_valid, self.max_len, self.hidden_size)
        self.test_set = None
        if X_test is not None and y_test is not None:
            self.test_set = TraditionalDataset(X_test, y_test, self.max_len, self.hidden_size)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False) if self.test_set is not None else None

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

        # 计算一次模型复杂度
        try:
            if self.result_save_path and "0_epo" in self.result_save_path:
                print("Running Complexity Analysis...")
                sample = torch.randn(1, 3, self.max_len, self.hidden_size).to(self.device)
                self.analyze_complexity(sample)
        except Exception as e:
            print(f"Complexity Analysis Failed: {e}")


    def fit(self):
        self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

        for i, data in progress_bar:
            self.optimizer.zero_grad()
            vectors = data["vector"].to(self.device)
            targets = data["targets"].to(self.device)

            with autocast():
                outputs, _ = self.model(vectors)
                loss = self.loss_fn(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1).flatten()
            losses.append(loss.item())
            predictions += list(np.array(preds.cpu()))
            labels += list(np.array(targets.cpu()))

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scheduler.step()
            progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets) / len(targets)):.3f}')

        train_loss = np.mean(losses)
        score_dict = get_MCM_score(labels, predictions)
        return train_loss, score_dict

    def eval(self, best_sc=0):
        print("start evaluating...")
        self.model = self.model.eval()
        losses = []
        pre = []
        label = []
        prob_list = []  # 新增：用于记录正类预测概率 (ROC/PR 曲线所需)
        correct_predictions = 0
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        with torch.no_grad():
            for _, data in progress_bar:
                vectors = data["vector"].to(self.device)
                targets = data["targets"].to(self.device)
                outputs, _ = self.model(vectors)
                loss = self.loss_fn(outputs, targets)
                
                # 新增：计算Softmax后的类概率，并取类别 1 (漏洞) 的概率
                probs = torch.softmax(outputs, dim=1)[:, 1]
                prob_list += list(np.array(probs.cpu()))
                
                preds = torch.argmax(outputs, dim=1).flatten()
                correct_predictions += torch.sum(preds == targets)

                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))

                losses.append(loss.item())
                progress_bar.set_description(
                    f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets) / len(targets)):.3f}')
        val_acc = correct_predictions.double() / len(self.valid_set)
        print("val_acc : ", val_acc)

        if val_acc > best_sc:
            self.save_model("model.pkl")
            best_sc = val_acc
            # 新增：在最佳 Epoch 保存用于绘制 ROC 和 PR 的概率数据
            prob_save_path = self.result_save_path.replace(".result", "_probs.pkl")
            try:
                import pickle
                with open(prob_save_path, 'wb') as f:
                    pickle.dump({'labels': label, 'probs': prob_list}, f)
                print(f"[*] Successfully saved best probabilities for ROC/PR to: {prob_save_path}")
            except Exception as e:
                print(f"[*] Error saving probabilities: {e}")

        score_dict = get_MCM_score(label, pre)
        val_loss = np.mean(losses)
        return val_loss, score_dict, best_sc

    def test_best_model(self, model_path="model.pkl"):
        if getattr(self, 'test_loader', None) is None or self.test_set is None:
            return None
        if os.path.exists(model_path):
            self.load_model(model_path)

        original_loader = self.valid_loader
        original_set = self.valid_set
        try:
            self.valid_loader = self.test_loader
            self.valid_set = self.test_set
            test_loss, test_score, _ = self.eval(best_sc=1)
        finally:
            self.valid_loader = original_loader
            self.valid_set = original_set

        mcm = np.array(test_score.get("MCM", []))
        if mcm.ndim == 3 and mcm.shape[0] >= 2:
            positive_cm = mcm[1]
            tn = int(positive_cm[0, 0])
            fp = int(positive_cm[0, 1])
            fn = int(positive_cm[1, 0])
            tp = int(positive_cm[1, 1])
        else:
            tp = fp = tn = fn = 0

        precision = (tp / (tp + fp) * 100.0) if (tp + fp) else 0.0
        recall = (tp / (tp + fn) * 100.0) if (tp + fn) else 0.0

        return {
            "loss": float(test_loss),
            "metrics": {
                "ACC": float(test_score.get("ACC", 0.0)),
                "P": round(precision, 4),
                "R": round(recall, 4),
                "F1": float(test_score.get("W_f1", 0.0)),
                "FPR": float(test_score.get("W_fpr", 0.0)),
                "FNR": float(test_score.get("W_fnr", 0.0)),
            },
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            "raw_score": test_score,
        }

    def train(self):
        learning_record_dict = {}
        train_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])
        test_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])
        best_sc = self.best_sc
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_loss, train_score = self.fit()
            train_table.add_row(
                ["tra", str(epoch + 1), format(train_loss, '.4f')] + [train_score[j] for j in train_score if
                                                                      j != "MCM"])
            print(train_table)

            val_loss, val_score, best_sc = self.eval(best_sc)
            print("=" * 100)
            print(best_sc)
            print("=" * 100)
            test_table.add_row(
                ["val", str(epoch + 1), format(val_loss, '.4f')] + [val_score[j] for j in val_score if j != "MCM"])
            print(test_table)
            print("\n")
            learning_record_dict[epoch] = {'train_loss': train_loss, 'val_loss': val_loss, "train_score": train_score,
                                           "val_score": val_score}
            sava_data(self.result_save_path, learning_record_dict)
            print("\n")

        return best_sc

    def freeze_parameters(self):
        """冻结除最后一层外的所有层参数。"""
        for name, param in self.model.named_parameters():
            if (self.model.fc.name in name) or (self.model.sequential_encoder.name in name) or (
                    self.model.adapt_layer.name in name):  # 替换为实际最后一层的名称
                param.requires_grad = False
                print(f"冻结参数: {name}")

    def save_model(self, model_path):
        """保存模型及其状态。"""
        torch.save(self.model.state_dict(), model_path)
        print(f"模型已保存到 {model_path}")

    def load_model(self, model_path):
        """加载模型及其状态。"""
        self.model.load_state_dict(torch.load(model_path))
        # if os.path.exists(model_path):
        #     checkpoint = torch.load(model_path)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     # self.load_first_stage_weights(self.model, model_path)
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #     self.epochs = checkpoint['epochs']
        #     print(f"模型已从 {model_path} 加载")
        # else:
        #     print(f"未找到模型文件: {model_path}")

    # def load_first_stage_weights(self, textcnn_aug_model, first_stage_checkpoint):
    #     # 加载第一个阶段的checkpoint
    #     checkpoint = torch.load(first_stage_checkpoint)
    #     first_stage_state_dict = checkpoint['model_state_dict']
    #
    #     # 获取当前模型的 state_dict
    #     model_dict = textcnn_aug_model.state_dict()
    #
    #     # 过滤掉与第一个阶段不匹配的层，仅保留第一个阶段的部分（即不要加载 fc_risk_level 和 fc_vul_type）
    #     pretrained_dict = {k: v for k, v in first_stage_state_dict.items() if
    #                        k in model_dict and v.size() == model_dict[k].size()}
    #
    #     # 更新当前模型的 state_dict
    #     model_dict.update(pretrained_dict)
    #     textcnn_aug_model.load_state_dict(model_dict)
    #
    #     print("成功加载第一个阶段的参数到TextCNN_Aug模型中")
# Backward-compatible aliases.
TextCNN_0 = SequentialConvEncoder
TextCNN_1 = SequentialConvAttentionEncoder
ShuffleAttention = ParallelAttentionBlock
TextCNN_Aug = VulSCPAugmentedModel
TextCNN = VulSCPModel
CNN_Classifier = VulSCPTrainer
