# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import time
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
import os
import logging
from transformers.utils.notebook import format_time

# 超参数设置
data_path = './data/merge_all_4.csv'        # 数据集路径
save_path = './saved_dict/lstm.pt'          # 模型训练结果
dropout = 0.5                               # 随机丢弃
num_classes = 2                             # 类别数
num_epochs = 30                             # epoch数
batch_size = 128                            # mini-batch大小
learning_rate = 1e-3                        # 学习率
hidden_size = 128                           # lstm隐藏层
num_layers = 2                              # lstm层数

# 指定BERT模型的本地路径
local_bert_path = './bert-base-chinese/'

# 初始化BERT的tokenizer
tokenizer = BertTokenizer.from_pretrained(local_bert_path)


#训练日志，复用性很高的代码
def log_creater(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir, log_name)
    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log


def get_data():
    train, dev, test = load_dataset(data_path)
    return train, dev, test

def load_dataset(path):
    # 读取CSV文件
    df = pd.read_csv(path)
    df['label'] = df['label'].astype(int)

    # 批量处理文本
    texts = list(df['comment'])
    labels = list(df['label'])
    encodings = tokenizer(texts, add_special_tokens=True, max_length=512,
                          padding='max_length', truncation=True, return_attention_mask=True)

    # 准备输入数据
    contents = [(torch.tensor(encodings['input_ids'][i]),
                 torch.tensor(encodings['attention_mask'][i]),
                 labels[i]) for i in range(len(texts))]

    # 划分数据集
    train, X_t = train_test_split(contents, test_size=0.4, random_state=42)
    dev, test = train_test_split(X_t, test_size=0.5, random_state=42)
    return train, dev, test


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        input_ids, attention_mask, label = self.data[index]
        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.data)


# 以上是数据预处理的部分

# def get_time_dif(start_time):
#     """获取已使用时间"""
#     end_time = time.time()
#     time_dif = end_time - start_time
#     return timedelta(seconds=int(round(time_dif)))


# 定义BERT-LSTM模型
class Model(nn.Module):
    def __init__(self, hidden_size, num_classes, num_layers, dropout):
        super(Model, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(local_bert_path)
        bert_output_dim = self.bert.config.hidden_size  # BERT输出维度

        # 定义LSTM层
        self.lstm = nn.LSTM(bert_output_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)

        # 定义全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向LSTM，因此维度是hidden_size的两倍

    def forward(self, x, mask):
        # BERT模型的输出
        with torch.no_grad():  # 不计算BERT的梯度，以加快训练速度
            bert_output = self.bert(x, attention_mask=mask)

        # LSTM层的输入
        lstm_output, _ = self.lstm(bert_output[0])  # 使用BERT模型的最后一层隐藏状态

        # 使用LSTM最后一个时间步的输出
        out = self.fc(lstm_output[:, -1, :])
        return out


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# 权重初始化，默认xavier
# xavier和kaiming是两种初始化参数的方法
def init_network(model, method='xavier', exclude='bert'):
    for name, w in model.named_parameters():
        if not any(nd in name for nd in [exclude, 'embedding']):  # 排除BERT层和embedding层
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def plot_acc(train_acc):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_acc)))
    plt.plot(x, train_acc, alpha=0.9, linewidth=2, label='train acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend(loc='best')
    plt.savefig('results/acc.png', dpi=400)

def plot_loss(train_loss):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_loss)))
    plt.plot(x, train_loss, alpha=0.9, linewidth=2, label='train acc')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('results/loss.png', dpi=400)

# 定义训练的过程
def train(model, dataloaders):
    '''
    训练模型
    :param model: 模型
    :param dataloaders: 处理后的数据，包含trian,dev,test
    :param log 训练日志
    '''
    log = log_creater(output_dir='./results/logs/')
    log.info("   Train batch size = {}".format(batch_size))
    log.info("   Num epochs = {}".format(num_epochs))
    log.info("   Training Start!")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    total_t0 = time.time()

    dev_best_loss = float('inf')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Start Training...\n")
    plot_train_acc = []
    plot_train_loss = []

    for i in range(num_epochs):
        # 1，训练循环----------------------------------------------------------------
        # 将数据全部取完
        # 记录每一个batch
        step = 0
        train_lossi=0
        train_acci = 0
        t0 = time.time()
        for inputs, masks, labels in dataloaders['train']:
            # 训练模式，可以更新参数
            model.train()

            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            # 梯度清零，防止累加
            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            step += 1
            true = labels.data.cpu()
            predic = torch.max(outputs.data, 1)[1].cpu()
            train_lossi += loss.item()
            train_acci += metrics.accuracy_score(true, predic)
        train_time = format_time(time.time() - t0)
        # 2，验证集验证----------------------------------------------------------------
        t1 = time.time()
        dev_acc, dev_loss = dev_eval(model, dataloaders['dev'], loss_function,Result_test=False)
        val_time = format_time(time.time() - t1)
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), save_path)
        train_acc = train_acci/step
        train_loss = train_lossi/step
        plot_train_acc.append(train_acc)
        plot_train_loss.append(train_loss)

        print("epoch = {} :  train_loss = {:.3f}, train_acc = {:.2%}, dev_loss = {:.3f}, dev_acc = {:.2%}".
                  format(i+1, train_loss, train_acc, dev_loss, dev_acc))

        # 训练log
        log.info("Training: epoch = {}, train_loss = {:.3f}, train_acc = {:.2%}".format(i + 1, train_loss, train_acc))
        log.info('====Training epoch took: {:}===='.format(train_time))
        log.info('Running Validation...')
        # 验证log
        log.info("Validation: epoch = {}, dev_loss = {:.3f}, dev_acc = {:.2%}".format(i + 1, dev_loss, dev_acc))
        log.info('====Validation epoch took: {:}===='.format(val_time))
        log.info('Running Testing...')
    plot_acc(plot_train_acc)
    plot_loss(plot_train_loss)
    # 3，验证循环----------------------------------------------------------------
    # 读取最优的模型
    model.load_state_dict(torch.load(save_path))
    model.eval()
    t2 = time.time()
    test_acc, test_loss = dev_eval(model, dataloaders['test'], loss_function,Result_test=True)
    test_time = format_time(time.time() - t2)
    print('================'*8)
    print('test_loss: {:.3f}      test_acc: {:.2%}'.format(test_loss, test_acc))
    log.info("Test: test_loss = {:.3f}, test_acc = {:.2%}".format(test_loss, test_acc))
    log.info('====Test epoch took: {:}===='.format(test_time))
    log.info('')
    log.info('   Training Completed!')
    log.info('Total training took {:} (h:mm:ss)'.format(format_time(time.time() - total_t0)))

def result_test(real, pred):
    cv_conf = confusion_matrix(real, pred)
    acc = accuracy_score(real, pred)
    precision = precision_score(real, pred, average='micro')
    recall = recall_score(real, pred, average='micro')
    f1 = f1_score(real, pred, average='micro')

    # 打印指标
    patten = 'test:  acc: %.4f   precision: %.4f   recall: %.4f   f1: %.4f'
    print(patten % (acc, precision, recall, f1))

    # 将指标保存到DataFrame
    df = pd.DataFrame({
        'Accuracy': [acc],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    })

    # 将DataFrame保存到Excel
    df.to_excel("./results/score_data.xlsx", index=False)

    # 绘制并保存混淆矩阵
    labels = ['negative', 'active']
    disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels)
    disp.plot(cmap="Blues", values_format='')
    plt.savefig("results/reConfusionMatrix.tif", dpi=400)

# 模型评估
def dev_eval(model, data, loss_function,Result_test=False):
    '''
    :param model: 模型
    :param data: 验证集集或者测试集的数据
    :param loss_function: 损失函数
    :return: 损失和准确率
    '''
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, masks, labels in data:
            texts = texts.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            outputs = model(texts, masks)
            loss = loss_function(outputs, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if Result_test:
        result_test(labels_all, predict_all)
    else:
        pass
    return acc, loss_total / len(data)

if __name__ == '__main__':
    # 设置随机数种子，保证每次运行结果一致
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")

    # 加载数据
    train_data, dev_data, test_data = get_data()

    # 初始化BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(local_bert_path)

    # 创建DataLoader
    # dataloaders = {
    #     'train': DataLoader(TextDataset(train_data), batch_size, shuffle=True),
    #     'dev': DataLoader(TextDataset(dev_data), batch_size, shuffle=False),
    #     'test': DataLoader(TextDataset(test_data), batch_size, shuffle=False)
    # }
    dataloaders = {
        'train': DataLoader(TextDataset(train_data), batch_size=batch_size, shuffle=True, num_workers=4),
        'dev': DataLoader(TextDataset(dev_data), batch_size=batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(TextDataset(test_data), batch_size=batch_size, shuffle=False, num_workers=4)
    }

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(device)

    # 初始化模型
    model = Model(hidden_size, num_classes, num_layers, dropout).to(device)

    # 初始化网络权重
    init_network(model, exclude='bert')

    # 开始训练
    train(model, dataloaders)
