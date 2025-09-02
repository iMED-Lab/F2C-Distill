import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import os
from PIL import Image
from torchvision import transforms
from timm.data import create_transform
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def build_transform_ffa(mode="train"):

    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])


def build_transform_ct(mode, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if mode == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class SingleLabelImageFolder(Dataset):
    def __init__(self, data, transform=None):
        data = data.copy()
        data["label"] = data["label"].astype(int)  # 💡 保证是 int 类型
        self.data = data.values
        self.transform = transform

    def __getitem__(self, index):
        while True:
            if index >= len(self.data):
                index = 0  # 保底防止越界
            # patient_name, image_path, label, report = self.data[index]
            # image_path, label = self.data[index]
            patient_name, image_path, label = self.data[index]
            # 检查文件存在性
            if not os.path.exists(image_path):
                print(f"⚠️ 文件不存在，跳过：{image_path}")
                index += 1
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except (IOError, SyntaxError) as e:
                print(f"⚠️ 读取图片失败，跳过：{image_path} - {e}")
                index += 1
                continue

            if self.transform:
                image = self.transform(image)

            label = torch.tensor(label, dtype=torch.long)
            # skg1 = skg
            return image, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def label_statistics(df):
        print("\n📊 标签分布:")
        stats = df['label'].value_counts().sort_index()
        class_counts = []
        for label, count in stats.items():
            print(f"类别 {label}: {count} 个样本")
            class_counts.append(count)
        return class_counts

    @staticmethod
    def label_weights_for_balance(df):
        # 强制转换为整数类型，并移除非法值（比如 None 或 NaN）
        labels = pd.to_numeric(df["label"], errors="coerce")  # 转换失败会变成 NaN
        labels = labels.dropna().astype(int).values  # 去掉 NaN 后转为 int，再转为 numpy array

        class_sample_count = np.bincount(labels)
        class_weights = 1.0 / class_sample_count
        sample_weights = [class_weights[label] for label in labels]
        return sample_weights



class MultiModalImageFolder(Dataset):
    def __init__(self, data, cls_num, mode='train', transform_ffa=None, target_transform=None, transform_fc=None, if_semi=False):
        super(MultiModalImageFolder, self).__init__()

        data = data.copy()
        data["label"] = data["label"].astype(int)
        self.data = data.values  # shape: (N, 5) -> patient_name, ffa_path, label, report, fc_path

        self.cls_num = cls_num
        self.mode = mode
        self.transform_ffa = transform_ffa
        self.transform_fc = transform_fc
        self.target_transform = target_transform

        # 统计标签用
        self.labels = data["label"].tolist()

    def __getitem__(self, index):
        while True:
            if index >= len(self.data):
                index = 0

            patient_name, ffa_image_path, label, _, fc_image_path = self.data[index]

            # 检查图像路径是否存在
            if not (os.path.exists(ffa_image_path) and os.path.exists(fc_image_path)):
                print(f"⚠️ 文件不存在，跳过：{ffa_image_path}, {fc_image_path}")
                index += 1
                continue

            try:
                ffa_image = Image.open(ffa_image_path).convert("RGB")
                fc_image = Image.open(fc_image_path).convert("RGB")
            except (IOError, SyntaxError) as e:
                print(f"⚠️ 读取图片失败，跳过：{ffa_image_path}, {fc_image_path} - {e}")
                index += 1
                continue

            if self.transform_ffa:
                ffa_image = self.transform_ffa(ffa_image)
            if self.transform_fc:
                fc_image = self.transform_fc(fc_image)

            if self.target_transform:
                label = self.target_transform(label)

            ffa_label = torch.tensor(label, dtype=torch.long)
            fc_label = torch.tensor(label, dtype=torch.long)
            # skg1 = skg
            return (fc_image, ffa_image), (fc_label, ffa_label), patient_name

    def __len__(self):
        return len(self.data)

    @staticmethod
    def label_statistics(df, label_column='label', cls_num=5):
        print("\n📊 标签分布:")
        cls_count = np.zeros(cls_num).astype(np.int64)
        for label in df[label_column]:
            cls_count[label] += 1
        for i in range(cls_num):
            print(f"类别 {i}: {cls_count[i]} 个样本")
        return cls_count

    @staticmethod
    def label_weights_for_balance(df, label_column='label', cls_num=5):
        cls_count = MultiModalImageFolder.label_statistics(df, label_column, cls_num)
        labels_weight_list = []
        for label in df[label_column]:
            weight = 1 / cls_count[label]
            labels_weight_list.append(weight)
        return labels_weight_list

