import math
import sys
import os

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from utils.lr_sched import adjust_learning_rate
from utils.logger import MetricLogger, SmoothedValue
from utils.eval_single_ffa import compute_metrics, compute_classwise_metrics, print_result
from utils.losses import KDloss

# fundus

alldiseases_f = ['AMD', 'CSC', 'DR', 'Normal', 'RVO']

alldiseases_o = ['AMD', 'CSC', 'DR', 'Normal', 'RVO']


import torch.nn.functional as F

def get_effective_number(class_counts, beta):
    effective_num = 1.0 - torch.pow(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / weights.sum()  # normalize
    return weights

def class_balanced_focal_loss(outputs, targets, class_counts, beta=0.9, gamma=2.0):
    """
    outputs: [B, C] logits
    targets: [B] long tensor of class ids
    class_counts: [C] number of samples per class
    """
    device = outputs.device
    num_classes = outputs.shape[1]

    weights = get_effective_number(class_counts.to(device), beta).to(device)  # [C]

    # convert targets to one-hot
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()  # [B, C]

    probs = torch.softmax(outputs, dim=1)  # [B, C]
    probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)

    # focal loss formula
    focal_term = torch.pow(1.0 - probs, gamma)
    ce_loss = -targets_one_hot * torch.log(probs)
    loss = focal_term * ce_loss  # [B, C]

    # apply class-balanced weights
    loss = loss * weights.unsqueeze(0)  # [B, C]
    loss = loss.sum(dim=1).mean()  # scalar

    return loss


def train_one_epoch(model, criterion, data_loader, optimizer,
                    device, epoch, args):


    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = args.accum_iter

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.to(device)
        targets = targets.to(device).long()

        with torch.amp.autocast(device_type='cuda'):
            outputs = model(samples)

            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, data_loader, model, device, num_class, write_pred=False):
    if args.modality == 'fundus':
        alldiseases = alldiseases_f
    elif args.modality == 'ffa':
        alldiseases = alldiseases_o
    # 修改了
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    prediction_decode_list = []
    prediction_prob_list = []
    true_label_decode_list = []
    img_name_list = []
    
    # switch to evaluation mode
    model.eval()

    for _, (images, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()

        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, target)
            
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

        prediction_prob = torch.softmax(outputs, dim=1)
        # prediction_prob = torch.sigmoid(outputs)
        prediction_decode = torch.argmax(prediction_prob, dim=1)
        # prediction_decode = (prediction_prob > 0.5).float()  # 大于0.5的视为正类
        true_label_decode = target
        
        prediction_decode_list.extend(prediction_decode.cpu().detach().numpy().tolist())
        true_label_decode_list.extend(true_label_decode.cpu().detach().numpy().tolist())
        prediction_prob_list.extend(prediction_prob.cpu().detach().numpy().tolist())
        
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    prediction_prob_list = np.array(prediction_prob_list)
    if write_pred:
        init_dict = {'ImgName': [], 'Label': []}
        for d in alldiseases:
            init_dict[d] = []
        for img_name, pred, targets in zip(img_name_list, prediction_decode_list, true_label_decode_list):
            init_dict['ImgName'].extend([img_name, '', ])
            init_dict['Label'].extend(['pred', 'true'])
            for i, d in enumerate(alldiseases):
                init_dict[d].extend([pred[i], targets[i]])
        df = pd.DataFrame.from_dict(init_dict)
        df.to_excel(os.path.join(args.output_dir, 'test_result.xlsx'), float_format='%.4f')

    # 计算多标签的评估指标
    results = compute_metrics(true_label_decode_list, prediction_decode_list, prediction_prob_list)
    class_wise_results = compute_classwise_metrics(true_label_decode_list, prediction_decode_list, prediction_prob_list)

    print_result(class_wise_results, results, alldiseases)
    metric_logger.meters['kappa'].update(
        results['kappa'].item() if isinstance(results['kappa'], torch.Tensor) else results['kappa'])
    metric_logger.meters['f1_pr'].update(
        results['f1'].item() if isinstance(results['f1'], torch.Tensor) else results['f1'])
    metric_logger.meters['acc'].update(
        results['accuracy'].item() if isinstance(results['accuracy'], torch.Tensor) else results['accuracy'])
    metric_logger.meters['precision'].update(
        results['precision'].item() if isinstance(results['precision'], torch.Tensor) else results['precision'])
    metric_logger.meters['recall'].update(
        results['recall'].item() if isinstance(results['recall'], torch.Tensor) else results['recall'])

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, results['f1']

