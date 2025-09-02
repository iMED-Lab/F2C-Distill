
"""
Main function to pretrain FLAIR model using
an assembly dataset and vision-text modalities.
"""
import pandas as pd
import argparse
import torch
from flair.modeling.model import FLAIRModel
import os
from sklearn.model_selection import train_test_split

def process(args):

    if args.device == 'cuda':
        torch.cuda.set_device(int(args.device_id))  # 通过指定的 GPU ID 来设置设备
        device = torch.device(f"cuda:{args.device_id}")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    file_path = "../data/processed_label_415_fc_single.xlsx"
    df = pd.read_excel(file_path)
    # df = df[['病人名称', '图片路径', 'label']]
    # df_filtered = df.dropna(subset=["图片路径", "label"]).copy()
    # df_filtered["图片路径"] = df_filtered["图片路径"].str.replace("../", "./")
    # df_filtered = df_filtered[df_filtered["图片路径"].apply(os.path.exists)].copy()
    df = df[['病人名称', '数据路径', 'label']]
    df_filtered = df.dropna(subset=["数据路径", "label"]).copy()
    df_filtered["数据路径"] = df_filtered["数据路径"].str.replace("../", "../")
    # 检查数据路径是否真的存在
    df_filtered = df_filtered[df_filtered["数据路径"].apply(os.path.exists)].copy()

    # 映射标签为整数
    unique_labels = sorted(df_filtered["label"].unique())
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    df_filtered.loc[:, "label"] = df_filtered["label"].map(label_to_index)
    # 打印标签映射关系
    print("Label to index mapping:", label_to_index)
    # 标签统计信息
    from data.prepare_data import SingleLabelImageFolder
    class_counts = torch.tensor(SingleLabelImageFolder.label_statistics(df_filtered), dtype=torch.float, device=device)

    # 划分训练/验证/测试集（70/15/15）
    train_val_df, test_df = train_test_split(
        df_filtered, test_size=0.15, stratify=df_filtered["label"], random_state=42
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.1765, stratify=train_val_df["label"], random_state=42
    )
    # 说明：0.1765 ≈ 15% / 85%，确保最终 train/val/test 比例为约 70/15/15

    # 确保 train_df 占 70%，val_df 占 15%，test_df 占 15%
    print(f"Train set size: {len(train_val_df)}")
    print(f"Validation set size: {len(test_df)}")
    # print(f"Test set size: {len(test_df)}")
    # ======================
    # 构建 Dataset 和 DataLoader
    # ======================

    from data.prepare_data import build_transform_ffa
    train_dataset = SingleLabelImageFolder(train_val_df, transform=build_transform_ffa("train"))
    dev_dataset = SingleLabelImageFolder(test_df, transform=build_transform_ffa("val"))
    test_dataset = SingleLabelImageFolder(test_df, transform=build_transform_ffa("val"))  # 测试集使用验证集同样的增强

    print("✅ 数据加载完成，train/val/test 均已准备就绪。")

    weights = train_dataset.label_weights_for_balance(train_val_df)
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=8000,
                                                                   replacement=True)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        shuffle=False, num_workers=args.num_workers,
        drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        shuffle=False, num_workers=args.num_workers,
        drop_last=True)

    print(f"Train batches: {len(data_loader_train)}")
    print(f"Validation batches: {len(data_loader_val)}")
    print(f"Test batches: {len(data_loader_test)}")

    # Init FLAIR model
    model = FLAIRModel(device=device, vision_type=args.architecture, out_path=args.out_path, from_checkpoint=False, vision_pretrained=True,
                  )

    # Training
    model.fit(data_loader_train, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler,
              warmup_epoch=args.warmup_epoch, store_num=args.store_num)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default='2', type=str, help='select device id')
    parser.add_argument('--device', default='cuda', type=str, help='device: cuda or cpu')
    # Folders, data, etc.

    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--banned_categories', default=['myopia', 'cataract', 'macular hole', 'retinitis pigmentosa',
                                                        "myopic", "myope", "myop", "retinitis"])
    parser.add_argument('--out_path', default='checkpoint/flair_checkpoint', help='output path')

    # Prompts setting and augmentation hyperparams
    parser.add_argument('--caption', default="A [ATR] fundus photograph of [CLS]")
    parser.add_argument('--augment_description', default=True, type=lambda x: (str(x).lower() == 'true'))

    # Dataloader setting
    parser.add_argument('--balance', default=True, type=lambda x: (str(x).lower() == 'true'))

    # Training options
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    parser.add_argument('--scheduler', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--warmup_epoch', default=1, type=int, help='number of warmup epochs')
    parser.add_argument('--store_num', default=5, type=int)

    # Architecture and pretrained weights options
    parser.add_argument('--architecture', default='resnet_v2', help='resnet_v1 -- efficientnet')

    # Resources
    parser.add_argument('--num_workers', default=0, type=int, help='workers number for DataLoader')

    args, unknown = parser.parse_known_args()
    process(args=args)


if __name__ == "__main__":
    main()