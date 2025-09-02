
import argparse
import random
import time
import datetime
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from data.prepare_data import SingleLabelImageFolder, build_transform_ffa
from model.flair_single_ffa import FLAIRConceptClassifier
from process.finetune_single_ffa import train_one_epoch, evaluate
from utils.eval import save_model

# ======================
# 固定随机种子
# ======================
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ======================
# 参数解析
# ======================
def get_args_parser():
    parser = argparse.ArgumentParser('F2C_Distill', add_help=False)
    parser.add_argument('--modality', default='ffa', type=str, help='Modality for training backbone model')
    parser.add_argument('--device_id', default='0', type=str, help='Select GPU device ID')
    parser.add_argument('--device', default='cuda', type=str, help='Device: cuda or cpu')
    parser.add_argument('--seed', default=1, type=int)

    # 学习率
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Lower LR bound for cyclic schedulers')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    # 数据和概念路径
    parser.add_argument('--concept_path', default='concepts', type=str)

    # 数据增强
    parser.add_argument('--input_size', default=512, type=int)
    parser.add_argument('--color_jitter', type=float, default=None)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)

    # 训练参数
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--num_samples', default=8000, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--n_classes', default=5, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--print_freq', default=50, type=int)

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--output_dir', default='checkpoint/last_checkpoint', help='Path to save models')

    return parser


# ======================
# 数据加载与处理
# ======================
def load_and_preprocess_data(file_path, device):
    df = pd.read_excel(file_path)
    df = df[['病人名称', '数据路径', 'label']]
    df_filtered = df.dropna(subset=["数据路径", "label"]).copy()
    df_filtered["数据路径"] = df_filtered["数据路径"].str.replace("../", "./")
    df_filtered = df_filtered[df_filtered["数据路径"].apply(os.path.exists)].copy()

    # 标签映射为整数
    unique_labels = sorted(df_filtered["label"].unique())
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    df_filtered.loc[:, "label"] = df_filtered["label"].map(label_to_index)
    print("Label to index mapping:", label_to_index)

    # 统计每类样本数量
    class_counts = torch.tensor(SingleLabelImageFolder.label_statistics(df_filtered), dtype=torch.float, device=device)

    # 只划分 train/val，比例 8:2
    train_df, val_df = train_test_split(
        df_filtered, test_size=0.2, stratify=df_filtered["label"], random_state=42
    )

    print(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")

    # 构建 Dataset
    train_dataset = SingleLabelImageFolder(train_df, transform=build_transform_ffa("train"))
    val_dataset = SingleLabelImageFolder(val_df, transform=build_transform_ffa("val"))

    # 权重采样器
    weights = train_dataset.label_weights_for_balance(train_df)
    sampler_train = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # 构建 DataLoader
    data_loader_train = DataLoader(train_dataset, sampler=sampler_train, batch_size=args.batch_size,
                                   num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    data_loader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    print(f"DataLoader prepared: Train={len(data_loader_train)}, Val={len(data_loader_val)}")
    return data_loader_train, data_loader_val, class_counts


# ======================
# 模型初始化
# ======================
def initialize_model(args, device):
    model = FLAIRConceptClassifier(args, device)

    # 冻结 backbone
    for p in model.flair_model.parameters():
        p.requires_grad = False
    for p in model.flair_model.vision_model.parameters():
        p.requires_grad = True
    model.concept_classifier.requires_grad = True

    model = model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_parameters}')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer: {optimizer}")
    return model, criterion, optimizer


# ======================
# 主训练函数
# ======================
def main(args):
    # 设置设备
    device = torch.device(f"cuda:{args.device_id}" if args.device == 'cuda' else args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    print(f"Using device: {device}")

    # your data path
    file_path = "./data/data.xlsx"
    data_loader_train, data_loader_val, data_loader_test, class_counts = load_and_preprocess_data(file_path, device)

    # 初始化模型
    model, criterion, optimizer = initialize_model(args, device)

    if args.eval:
        evaluate(args, data_loader_test, model, device, num_class=args.n_classes)
        return

    # 训练循环
    max_metric = 0.0
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args=args)
        val_stats, val_metric = evaluate(args, data_loader_val, model, device, num_class=args.n_classes)

        # 保存最佳模型
        if val_metric > max_metric:
            max_metric = val_metric
            if args.output_dir:
                save_model(args, model, optimizer, epoch, if_best=True)
            print("------ Best model updated ------")

        # 每 20 轮保存快照
        if (epoch + 1) % 20 == 0 and args.output_dir:
            snapshot_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}_snapshot.pth")
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1},
                       snapshot_path)
            print(f"Saved snapshot: {snapshot_path}")

        # 最后一轮保存
        if epoch == args.epochs - 1 and args.output_dir:
            save_model(args, model, optimizer, epoch, if_best=False)
            print("------ Last model saved ------")

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training finished in {total_time}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)
