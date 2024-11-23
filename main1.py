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
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 超参数设置
# 数据集路径和模型保存路径的定义
pretrain_path = './data/pretrain_dataset.csv'  # 通用训练集路径
finetune_path = './data/finetune_dataset.csv'  # 微调数据集路径
pretrain_save_path = './saved_dict/pretrained_model.pt'  # 预训练模型保存路径
finetune_save_path = './saved_dict/finetuned_model.pt'  # 微调模型保存路径
local_bert_path = './bert-base-chinese/'  # BERT模型路径

# 初始化 BERT 分词器
tokenizer = BertTokenizer.from_pretrained(local_bert_path)


# 预训练阶段（通用数据集）的超参数配置
# 包括模型结构、优化参数和训练周期
pretrain_config = {
    "dropout": 0.4,
    "num_classes": 2,  # 分类任务的类别数
    "num_epochs": 10,  # 预训练的 epoch 数量
    "batch_size": 128,  # 批量大小
    "learning_rate": 1e-4,  # 学习率
    "hidden_size": 128,  # LSTM 隐藏层维度
    "num_layers": 2,  # LSTM 的层数
    "transformer_layers": 2,  # Transformer 编码层的数量
    "transformer_heads": 8,  # Transformer 的多头注意力头数
    "dim_feedforward": 2048,  # 适合大数据集的高容量设置
}

# 微调阶段（目标数据集）的超参数配置
# 适用于小规模数据集，需更小的学习率和批量大小
finetune_config = {
    "dropout": 0.5,
    "num_classes": 2,
    "num_epochs": 20,  # 微调的 epoch 数量
    "batch_size": 32,  # 批量大小
    "learning_rate": 2e-5,  # 学习率
    "hidden_size": 64,  # LSTM 隐藏层维度
    "num_layers": 1,  # LSTM 的层数
    "transformer_layers": 1,  # Transformer 编码层的数量
    "transformer_heads": 4,  # Transformer 的多头注意力头数
    "dim_feedforward": 512,  # 适合小数据集的低容量设置
}


# 日志创建函数
def log_creater(output_dir='./logs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir, log_name)

    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter('[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    log.addHandler(file)
    log.addHandler(stream)

    log.info('Creating log file: {}'.format(final_log_file))
    return log


# 数据集加载函数
def load_dataset(path, phase="finetune", test_size=0.4, dev_size=0.5, random_seed=42):
    """
    加载数据集并划分
    :param path: 数据集路径
    :param phase: 训练阶段 ("pretrain" 或 "finetune")
    :param test_size: 测试集比例
    :param dev_size: 验证集比例（相对测试集）
    :param random_seed: 随机种子
    :return: 划分后的训练、验证、测试数据
    """
    try:
        df = pd.read_csv(path)
        df['label'] = df['label'].astype(int)

        # 文本清洗和编码
        texts = list(df['comment'].fillna("").str.strip())
        labels = list(df['label'])
        encodings = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )

        contents = [(torch.tensor(encodings['input_ids'][i]),
                     torch.tensor(encodings['attention_mask'][i]),
                     labels[i]) for i in range(len(texts))]

        if phase == "pretrain":
            # 预训练阶段只需要返回完整训练集
            return contents, None, None
        elif phase == "finetune":
            # 微调阶段划分训练、验证和测试集
            train, X_t = train_test_split(contents, test_size=test_size, random_state=random_seed)
            dev, test = train_test_split(X_t, test_size=dev_size, random_seed=random_seed)
            return train, dev, test
        else:
            raise ValueError("Invalid phase. Use 'pretrain' or 'finetune'.")
    except Exception as e:
        raise ValueError(f"Error loading dataset from {path}: {str(e)}")


# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        input_ids, attention_mask, label = self.data[index]
        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.data)


# 数据加载器创建函数
def create_dataloader(data, batch_size, shuffle=False, num_workers=0):
    """
    根据数据生成 DataLoader
    :param data: 数据集
    :param batch_size: 每批次数据大小
    :param shuffle: 是否随机打乱数据
    :param num_workers: 数据加载时使用的子进程数
    :return: DataLoader 对象
    """
    if data is None:
        return None
    dataset = TextDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)


# 数据加载接口
def get_data(data_path, phase="finetune", batch_sizes=None):
    """
    数据加载入口
    :param data_path: 数据集路径
    :param phase: 训练阶段 ("pretrain" 或 "finetune")
    :param batch_sizes: dict，包含训练、验证、测试的 batch size
    :return: 包含 DataLoader 的字典
    """
    if batch_sizes is None:
        batch_sizes = {"train": 32, "dev": 32, "test": 32}

    # 动态设置 num_workers 参数
    num_workers = 4 if phase == "pretrain" else 0

    train_data, dev_data, test_data = load_dataset(data_path, phase=phase)

    dataloaders = {
        "train": create_dataloader(train_data, batch_size=batch_sizes["train"], shuffle=True, num_workers=num_workers),
    }

    # 仅在微调阶段创建 dev 和 test 的 DataLoader
    if phase == "finetune":
        dataloaders["dev"] = create_dataloader(dev_data, batch_size=batch_sizes["dev"], shuffle=False, num_workers=num_workers)
        dataloaders["test"] = create_dataloader(test_data, batch_size=batch_sizes["test"], shuffle=False, num_workers=num_workers)

    return dataloaders


# 模型定义(BERT-LSTM-Transformer模型)
class Model(nn.Module):
    def __init__(self, hidden_size, num_classes, num_layers, dropout, transformer_layers, transformer_heads, dim_feedforward):
        super(Model, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(local_bert_path)
        bert_output_dim = self.bert.config.hidden_size  # BERT输出维度

        # 定义LSTM层
        self.lstm = nn.LSTM(
            input_size=bert_output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        # Transformer编码器层
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size * 2,
            nhead=transformer_heads,
            dim_feedforward=dim_feedforward,  # 动态调整
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=transformer_layers
        )

        # 定义全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向LSTM，因此维度是hidden_size的两倍

    def forward(self, x, mask):
        # BERT模型的输出
        with torch.no_grad():
            bert_output = self.bert(x, attention_mask=mask)

        # LSTM层的输入
        lstm_output, _ = self.lstm(bert_output.last_hidden_state)

        # Transformer层
        transformer_output = self.transformer_encoder(lstm_output)

        # 全连接层  使用Transformer编码器最后一个时间步的输出
        out = self.fc(transformer_output[:, -1, :])
        return out


# 获取已使用时间
def get_time_dif(start_time):
    """
    获取运行时间并格式化为 h:mm:ss
    :param start_time: 起始时间
    :return: 格式化的时间字符串
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return str(timedelta(seconds=int(round(time_dif))))


# 权重初始化函数
def init_network(model, method='xavier', exclude=['bert', 'embedding']):
    """
    初始化模型权重
    :param model: 待初始化的模型
    :param method: 初始化方法，支持 'xavier', 'kaiming' 或 'normal'
    :param exclude: 不进行初始化的模块名列表
    """
    for name, w in model.named_parameters():
        # 跳过需要排除的模块
        if any(nd in name for nd in exclude):
            continue

        # 初始化权重
        if 'weight' in name and w.dim() > 1:  # 仅对多于1维的权重应用初始化
            if method == 'xavier':
                nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.normal_(w, mean=0, std=0.01)

        # 初始化偏置
        elif 'bias' in name:
            nn.init.constant_(w, 0)


def plot_acc(train_acc, save_path='results/acc.png', title='Training Accuracy', show=False):
    """
    绘制训练准确率曲线
    :param train_acc: 训练准确率的历史记录
    :param save_path: 保存图片的路径
    :param title: 图表标题
    :param show: 是否显示图表
    """
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_acc)))
    plt.plot(x, train_acc, alpha=0.9, linewidth=2, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig(save_path, dpi=400)
    if show:
        plt.show()
    plt.close()


def plot_loss(train_loss, save_path='results/loss.png', title='Training Loss', show=False):
    """
    绘制训练损失曲线
    :param train_loss: 训练损失的历史记录
    :param save_path: 保存图片的路径
    :param title: 图表标题
    :param show: 是否显示图表
    """
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_loss)))
    plt.plot(x, train_loss, alpha=0.9, linewidth=2, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig(save_path, dpi=400)
    if show:
        plt.show()
    plt.close()


def train(model, dataloaders, config, save_path, log, phase="pretrain"):
    """
    训练模型
    :param model: 待训练的模型
    :param dataloaders: 数据加载器，包含 train/dev/test
    :param config: 配置字典，包含超参数
    :param save_path: 模型保存路径
    :param log: 日志记录器
    :param phase: 当前阶段 ("pretrain" 或 "finetune")
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    log.info("Training Configuration:")
    log.info("   Train batch size = {}".format(config['batch_size']))
    log.info("   Learning rate = {:.1e}".format(config['learning_rate']))
    log.info("   Number of epochs = {}".format(config['num_epochs']))
    log.info("   Model save path = {}".format(save_path))
    log.info("   Current phase = {}".format(phase))
    log.info("   Training Start!")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    loss_function = torch.nn.CrossEntropyLoss()

    best_dev_loss = float('inf')
    early_stop_count = 0
    patience = 10
    num_epochs = config['num_epochs']

    # 动态图像路径
    acc_path = f'results/{phase}_accuracy_curve.png'
    loss_path = f'results/{phase}_loss_curve.png'

    # 存储训练结果
    train_acc_history, train_loss_history = [], []

    log.info(f"Starting {phase} training with {num_epochs} epochs, batch size {config['batch_size']}")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for inputs, masks, labels in dataloaders['train']:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = loss_function(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss / len(dataloaders['train']))

        # 动态验证集评估
        if dataloaders.get('dev') is not None:
            dev_acc, dev_loss = dev_eval(model, dataloaders['dev'], loss_function, device)
            scheduler.step(dev_loss)

            # 保存最优模型
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                torch.save(model.state_dict(), save_path)
                early_stop_count = 0
            else:
                early_stop_count += 1
        else:
            dev_acc, dev_loss = None, None
            log.info(f"No validation data available for {phase} phase.")

        # 动态日志记录
        if dev_loss is not None and dev_acc is not None:
            log_line = f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}"
        else:
            log_line = f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | No validation data"
        log.info(log_line)
        print(log_line)

        # 早停机制
        if early_stop_count >= patience:
            log.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # 绘制训练曲线
    plot_acc(train_acc_history, save_path=acc_path)
    plot_loss(train_loss_history, save_path=loss_path)

    log.info(f"{phase.capitalize()} training completed in {get_time_dif(start_time)}.")


def dev_eval(model, dataloader, loss_function, device):
    """
    在验证集或测试集上评估模型
    :param model: 模型
    :param dataloader: 数据加载器
    :param loss_function: 损失函数
    :param device: 设备
    :return: 准确率和平均损失
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, masks, labels in dataloader:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

            outputs = model(inputs, masks)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    acc = correct / total
    return acc, total_loss / len(dataloader)


def result_test(model, dataloader, loss_function, device, save_path="results/test_confusion_matrix.png"):
    """
    在测试集上评估模型性能并输出详细结果
    :param model: 模型
    :param dataloader: 测试数据加载器
    :param loss_function: 损失函数
    :param device: 设备
    :param save_path: 混淆矩阵保存路径
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, masks, labels in dataloader:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

            outputs = model(inputs, masks)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    acc = correct / total
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    # 打印指标
    print(f"Test Results - Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # 保存指标到 Excel
    score_data_path = "./results/score_data.xlsx"
    df = pd.DataFrame({
        'Accuracy': [acc],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    })
    df.to_excel(score_data_path, index=False)
    print(f"Test metrics saved to {score_data_path}")

    # 绘制并保存混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(conf_matrix, display_labels=['Negative', 'Positive']).plot(cmap='Blues')
    plt.savefig(save_path, dpi=400)
    print(f"Confusion matrix saved to {save_path}")



# 主函数
if __name__ == '__main__':
    # 设置随机数种子，保证结果一致
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    # 记录开始时间
    start_time = time.time()

    # 日志记录器
    log = log_creater(output_dir='./results/logs/')

    # --------------------- 预训练阶段 ---------------------
    log.info("=== Starting Pretrain Phase ===")
    pretrain_path = './data/pretrain_dataset.csv'  # 替换为通用训练集路径
    log.info(f"Pretrain data path: {pretrain_path}")

    # 加载预训练数据
    pretrain_dataloaders = get_data(
        data_path=pretrain_path,
        phase="pretrain",
        batch_sizes={"train": pretrain_config['batch_size']}
    )

    log.info("Initializing pretrain model...")
    # 预训练阶段
    pretrain_model = Model(
        hidden_size=pretrain_config['hidden_size'],
        num_classes=pretrain_config['num_classes'],
        num_layers=pretrain_config['num_layers'],
        dropout=pretrain_config['dropout'],
        transformer_layers=pretrain_config['transformer_layers'],
        transformer_heads=pretrain_config['transformer_heads'],
        dim_feedforward=pretrain_config['dim_feedforward']  # 动态传递
    )
    init_network(pretrain_model, method='xavier', exclude=['bert', 'embedding'])

    # 开始预训练
    train(pretrain_model, pretrain_dataloaders, pretrain_config, pretrain_save_path, log, phase="pretrain")
    log.info("Pretrain phase completed.")

    # --------------------- 微调阶段 ---------------------
    log.info("=== Starting Finetune Phase ===")
    finetune_path = './data/finetune_dataset.csv'  # 替换为微调数据集路径
    log.info(f"Finetune data path: {finetune_path}")

    # 加载微调数据
    finetune_dataloaders = get_data(
        data_path=finetune_path,
        phase="finetune",
        batch_sizes={
            "train": finetune_config['batch_size'],
            "dev": finetune_config['batch_size'],
            "test": finetune_config['batch_size']
        }
    )

    log.info("Initializing finetune model...")
    # 微调阶段
    finetune_model = Model(
        hidden_size=finetune_config['hidden_size'],
        num_classes=finetune_config['num_classes'],
        num_layers=finetune_config['num_layers'],
        dropout=finetune_config['dropout'],
        transformer_layers=finetune_config['transformer_layers'],
        transformer_heads=finetune_config['transformer_heads'],
        dim_feedforward=finetune_config['dim_feedforward']  # 动态传递
    )
    finetune_model.load_state_dict(torch.load(pretrain_save_path))  # 加载预训练权重

    # 开始微调
    train(finetune_model, finetune_dataloaders, finetune_config, finetune_save_path, log, phase="finetune")
    log.info("Finetune phase completed.")

    # 测试阶段
    log.info("Evaluating finetune model on test set...")
    # 加载最优微调模型
    finetune_model.load_state_dict(torch.load(finetune_save_path))
    result_test(
        model=finetune_model,
        dataloader=finetune_dataloaders['test'],
        loss_function=torch.nn.CrossEntropyLoss(),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        save_path="results/test_confusion_matrix.png"
    )

    # 总时间记录
    total_time = get_time_dif(start_time)
    log.info(f"All training completed. Total time usage: {total_time}.")
