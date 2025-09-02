import math
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import pandas as pd

from utils.lr_sched import adjust_learning_rate
from utils.logger import MetricLogger, SmoothedValue
from utils.eval_single import compute_metrics, compute_classwise_metrics, print_result
from utils.losses import KDloss

# fundus
alldiseases_f = ['AMD', 'CSC', 'DR', 'Normal', 'RVO']

# OCT
alldiseases_o = ['AMD', 'CSC', 'DR', 'Normal', 'RVO']

def dist_s_t(p_logit, q_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    q = F.softmax(q_logit / T, dim=-1)
    dist = torch.sum(torch.abs(q - p), 1)

    return torch.mean(dist)


def dist_s_label(y, q):

    q = F.softmax(q, dim=-1)
    dist = torch.sum(torch.abs(q - y), 1)

    return torch.mean(dist)

def train_one_epoch(model, criterion, data_loader, optimizer,
                    device, epoch, args, ffa_model, oct_optimizer):
    criterion = nn.CrossEntropyLoss()
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = args.accum_iter

    for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        fc_samples = samples[0].to(device)
        fc_targets = targets[0].to(device)
        ffa_samples = samples[1].to(device)
        ffa_targets = targets[1].to(device)

        with torch.amp.autocast('cuda'):
            # 正向传播
            outputs, concept_sims = model.forward_distill(fc_samples)
            loss_cls_f = criterion(outputs, fc_targets)
            preds = torch.softmax(outputs, dim=1)

            ffa_outputs, ffa_concept_sims = ffa_model.forward_distill(ffa_samples)
            loss_cls_o = criterion(ffa_outputs, ffa_targets)

            # 更新 ffa_model（教师）
            oct_optimizer.zero_grad()
            loss_cls_o.backward()
            oct_optimizer.step()

            ffa_concept_sims = ffa_concept_sims.detach()  # 停梯度传播

            # 计算学生与教师之间的距离关系（用于判断是否蒸馏）
            target_onehot = torch.zeros(fc_samples.size(0), args.n_classes).to(device)
            target_onehot.scatter_(1, fc_targets.view(-1, 1), 1)
            s_label = dist_s_label(target_onehot, outputs.detach())
            t_label = dist_s_label(target_onehot, ffa_outputs.detach())

            ps_pt = dist_s_t(ffa_outputs.detach(), outputs.detach(), 1)
            epsilon = torch.exp(-1 * t_label / (s_label + t_label))
            delta = 0.8 * (s_label - epsilon * t_label)

            # ========== 蒸馏策略 ==========
            if ps_pt > delta and t_label < s_label:
                # 直接使用 concept_sims 和 ffa_concept_sims 进行 MSE 蒸馏
                loss_distill_feat = nn.MSELoss()(
                    concept_sims.view(fc_samples.size(0), -1),
                    ffa_concept_sims.view(fc_samples.size(0), -1)
                ) * args.beta_distill
                loss_distill = loss_distill_feat
                loss_dis_value = loss_distill.item()
            else:
                loss_distill = 0
                loss_dis_value = 0

            # ======== 总 loss 组合 ========
            cls_weight, distill_weight = get_loss_weights(epoch)
            if not math.isnan(loss_dis_value):
                loss = cls_weight * loss_cls_f + distill_weight * loss_distill
            else:
                loss = loss_cls_f

            loss_cls_o_value = loss_cls_o.item()
            loss_cls_value = loss_cls_f.item()
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dis_value)
                print(loss_cls_f)
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 日志记录
        metric_logger.update(loss_cls_o=loss_cls_o_value)
        metric_logger.update(loss_cls=loss_cls_value)
        metric_logger.update(loss_dis=loss_dis_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def get_loss_weights(epoch, total_epochs=100, distill_final_ratio=0.4, warmup_ratio=0.3):
    """
    根据 epoch 动态调整分类损失和蒸馏损失的权重。
    - warmup_ratio: 前多少比例 epoch 进行权重调整
    - distill_final_ratio: 最终蒸馏损失占比，例如 0.4
    """
    if epoch < total_epochs * warmup_ratio:
        # 在 warmup 区间内线性插值
        alpha = epoch / (total_epochs * warmup_ratio)  # 从 0 到 1
        distill_weight = alpha * distill_final_ratio
    else:
        distill_weight = distill_final_ratio

    cls_weight = 1.0 - distill_weight
    return cls_weight, distill_weight
@torch.no_grad()
def evaluate(args, data_loader, model, device, num_class, write_pred=False):
    if args.modality == 'fundus':
        alldiseases = alldiseases_f

    elif args.modality == 'ffa':
        alldiseases = alldiseases_o

    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    prediction_decode_list = []
    prediction_prob_list = []
    true_label_decode_list = []
    img_name_list = []
    
    # switch to evaluation mode
    model.eval()

    for _, (samples, targets, img_names) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        fc_samples = samples[0].to(device)
        fc_targets = targets[0].to(device)
        images = fc_samples.to(device, non_blocking=True)
        target = fc_targets.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, target)
            
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

        prediction_prob = torch.softmax(outputs, dim=1)
        prediction_decode = torch.argmax(prediction_prob, dim=1)
        true_label_decode = target
        
        prediction_decode_list.extend(prediction_decode.cpu().detach().numpy().tolist())
        true_label_decode_list.extend(true_label_decode.cpu().detach().numpy().tolist())
        prediction_prob_list.extend(prediction_prob.cpu().detach().numpy().tolist())
        img_name_list.extend(list(img_names))
        
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
    
    results = compute_metrics(true_label_decode_list, prediction_decode_list,prediction_prob_list)
    class_wise_results = compute_classwise_metrics(true_label_decode_list, prediction_decode_list,prediction_prob_list)

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
