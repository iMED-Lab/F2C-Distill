import os
import argparse
import random
import time
import datetime
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from data.prepare_data import MultiModalImageFolder, build_transform_ct, build_transform_ffa
from model.flair_distill import FLAIRMultiLayer
from process.finetune import train_one_epoch, evaluate
from utils.eval import save_model, load_model

# ======================
# 固定随机种子
# ======================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ======================
# 参数解析
# ======================
def get_args_parser():
    parser = argparse.ArgumentParser('Multi Eye CLIP', add_help=False)
    parser.add_argument('--modality', default='fundus', type=str, help='Modality for training backbone model')
    parser.add_argument('--device_id', default='3', type=str, help='Select device id')
    parser.add_argument('--device', default='cuda', type=str, help='Device: cuda or cpu')

    # 学习率和蒸馏参数
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--alpha_distill', type=float, default=0.6)
    parser.add_argument('--beta_distill', type=float, default=0.05)
    parser.add_argument('--temperature', type=float, default=10)

    # 数据路径
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--concept_path', default='concepts', type=str)
    parser.add_argument('--checkpoint_path', default='checkpoint/ffa9371_checkpoint')
    parser.add_argument('--fc_checkpoint_path', default='./checkpoint/fc9371_checkpoint')
    parser.add_argument('--output_dir', default='./checkpoints/fct9378guiyi_checkpoint')

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
    parser.add_argument('--num_samples', default=15000, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--n_classes', default=5, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--eval', action='store_true', default=False)

    return parser


# ======================
# 数据处理与 DataLoader
# ======================
def load_data(args):
    # your data path
    file_path = "./data/data.xlsx"
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

    # 标签映射
    unique_labels = sorted(df_filtered["label"].unique())
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    df_filtered["label"] = df_filtered["label"].map(label_to_index)
    print("Label to index mapping:", label_to_index)

    # 标签统计
    MultiModalImageFolder.label_statistics(df_filtered, label_column='label', cls_num=args.n_classes)

    # 划分 train/val/test（70/15/15）
    train_val_df, test_df = train_test_split(df_filtered, test_size=0.15, stratify=df_filtered["label"],
                                             random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1765, stratify=train_val_df["label"], random_state=42)
    print(f"Train set: {len(train_df)}, Val set: {len(val_df)}, Test set: {len(test_df)}")

    # 数据增强
    transform_ffa_train = build_transform_ffa("train")
    transform_ct_train = build_transform_ct("train", args)
    transform_ffa_val = build_transform_ffa("val")

    # Dataset
    train_dataset = MultiModalImageFolder(train_df, args.n_classes, transform_ffa=transform_ffa_train,
                                          transform_fc=transform_ct_train)
    val_dataset = MultiModalImageFolder(val_df, args.n_classes, transform_ffa=transform_ffa_val,
                                        transform_fc=transform_ffa_val)
    test_dataset = MultiModalImageFolder(test_df, args.n_classes, transform_ffa=transform_ffa_val,
                                         transform_fc=transform_ffa_val)

    # 权重采样器
    weights = train_dataset.label_weights_for_balance(train_df, label_column='label', cls_num=args.n_classes)
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=args.num_samples,
                                                                   replacement=True)

    # DataLoader
    data_loader_train = torch.utils.data.DataLoader(train_dataset, sampler=sampler_train, batch_size=args.batch_size,
                                                    num_workers=args.num_workers, pin_memory=args.pin_mem,
                                                    drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=args.pin_mem,
                                                  drop_last=False)
    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.num_workers, pin_memory=args.pin_mem,
                                                   drop_last=False)

    return data_loader_train, data_loader_val, data_loader_test


# ======================
# 模型初始化
# ======================
def initialize_models(args, device):
    # 主模型
    model = FLAIRMultiLayer(args, device)
    for p in model.flair_model.parameters():
        p.requires_grad = False
    for p in model.flair_model.vision_model.parameters():
        p.requires_grad = True
    model.concept_classifier.requires_grad = True
    model.project_4.requires_grad = True
    model = model.to(device)

    # FFA 教师模型
    ffa_model = FLAIRMultiLayer(args, device, modality='ffa')
    ffa_state_dict = load_model(args=args, if_best=True, device=device, checkpoint=args.checkpoint_path,
                                weights_only=False)
    ffa_model.load_state_dict(ffa_state_dict, strict=False)
    ffa_model.to(device).eval()

    return model, ffa_model


# ======================
# 主训练循环
# ======================
def main(args):
    print('Job dir:', os.path.dirname(os.path.realpath(__file__)))
    print(args)

    device = torch.device(args.device, int(args.device_id))
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    # 数据加载
    data_loader_train, data_loader_val, data_loader_test = load_data(args)

    # 模型
    model, ffa_model = initialize_models(args, device)

    # 损失和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    params_to_update = [p for n, p in ffa_model.named_parameters() if 'project' in n]
    for p in params_to_update:
        p.requires_grad = True
    ffa_optimizer = torch.optim.AdamW(params_to_update, lr=args.lr, weight_decay=args.weight_decay)

    if args.eval:
        evaluate(args, data_loader_test, model, device, num_class=args.n_classes)
        return

    # 训练循环
    max_metric = 0.0
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args, ffa_model, ffa_optimizer)
        val_stats, val_metric = evaluate(args, data_loader_val, model, device, num_class=args.n_classes)

        # 保存最佳模型
        if val_metric > max_metric:
            max_metric = val_metric
            if args.output_dir:
                save_model(args=args, model=model, optimizer=optimizer, epoch=epoch, if_best=True)
            print('------ Best model updated ------')

        # 最后一轮保存
        if epoch == args.epochs - 1 and args.output_dir:
            save_model(args=args, model=model, optimizer=optimizer, epoch=epoch, if_best=False)
            print('------ Last model saved ------')

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training finished in {total_time}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)
