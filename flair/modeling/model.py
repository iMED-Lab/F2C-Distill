"""
Main FLAIR modeling function.
"""

import torch
import torchvision
import numpy as np
import os

from .dictionary import definitions
from . import constants
from .misc import wget_gdrive_secure

from tqdm import tqdm
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, logging, BertConfig
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


definitions = {
    "2": ["no diabetic retinopathy", "no microaneurysms", "many exudates near the macula",
          "many haemorrhages near the macula", "retinal thickening near the macula",
          "hard exudates", "cotton wool spots", "few severe haemorrhages",
          "only few microaneurysms", "venous beading", "many severe haemorrhages",
          "intraretinal microvascular abnormality", "preretinal or vitreous haemorrhage",
          "neovascularization"],

    "0": ["many small drusen", "few medium-sized drusen", "large drusen",
          "macular degeneration"],

    "4": ["localized serous retinal detachment",
          "detachment of the neurosensory retina",
          "leakage point on fluorescein angiography",
          "focal or diffuse retinal pigment epithelium (RPE) decompensation",
          "thickened choroid on OCT",
          "smokestack or inkblot leakage pattern",
          "round or oval area of subretinal fluid",
          "absence of hemorrhage or hard exudates",
          "dome-shaped elevation of retina"],

    "3": ["normal"],

    "1": ["localized serous retinal detachment",
          "detachment of the neurosensory retina",
          "leakage point on fluorescein angiography",
          "focal or diffuse retinal pigment epithelium (RPE) decompensation",
          "thickened choroid on OCT",
          "smokestack or inkblot leakage pattern",
          "round or oval area of subretinal fluid",
          "absence of hemorrhage or hard exudates",
          "hyperreflective subretinal fluid on OCT",
          "dome-shaped elevation of retina"]
}

class FLAIRModel(torch.nn.Module):
    def __init__(self, device, vision_type='resnet_v1', bert_type='emilyalsentzer/Bio_ClinicalBERT', vision_pretrained=True,
                 proj_dim=512, proj_bias=False, logit_scale_init_value=0.07, from_checkpoint=True, weights_path=None,
                 out_path=None, image_size=512, caption="A fundus photograph of [CLS]", projection=True,
                 norm_features=True):
        super().__init__()

        self.device = device
        # Set attributes
        self.vision_type = vision_type
        self.bert_type = bert_type
        self.vision_pretrained = vision_pretrained
        self.proj_dim = proj_dim
        self.proj_bias = proj_bias
        self.logit_scale_init_value = logit_scale_init_value
        self.from_checkpoint = from_checkpoint
        self.weights_path = weights_path
        self.out_path = out_path
        self.image_size = image_size
        self.caption = caption
        # Use of projection head and feature normalization on visione encoder
        # (only relevant during transferability stage)
        self.projection = projection
        self.norm_features = norm_features

        # Set vision and text encoder
        self.vision_model = VisionModel(vision_type=self.vision_type, pretrained=self.vision_pretrained,
                                        proj_dim=self.proj_dim, proj_bias=self.proj_bias, projection=self.projection,
                                        norm=self.norm_features)
        self.text_model = TextModel(bert_type=self.bert_type, proj_dim=self.proj_dim, proj_bias=self.proj_bias,
                                    projection=self.projection, norm=self.norm_features)

        # learnable temperature for contrastive loss
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1/self.logit_scale_init_value)))

        # Load pretrained weights
        if from_checkpoint:
            self.load_from_pretrained(self.weights_path)

        # Set model to device
        self.to(self.device)

    def load_from_pretrained(self, weights_path=None):

        if weights_path is None:
            import zipfile

            input_dir = constants.PATH_PRETRAINED_WEIGHTS
            pretrained_id = constants.ID_FLAIR_RESNET_V1
            pretrained_url_id = constants.URL_ID_FLAIR_RESNET_V1
            weights_path = input_dir + pretrained_id

            if not os.path.exists(input_dir + pretrained_id):
                if not os.path.exists(input_dir):
                    Path(input_dir).mkdir(parents=True, exist_ok=True)

                # download url link
                wget_gdrive_secure(pretrained_url_id, input_dir, filename="weights.zip")

                # unzip
                zipf = zipfile.ZipFile(input_dir + "weights.zip")
                zipf.extractall(input_dir)
                zipf.close()
                print('\n Download model to:', input_dir + pretrained_id)

        state_dict = torch.load(weights_path)
        # 删除 text_model 中的键，只保留 vision 相关的参数
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("text_model")}

        # 下面的true改为了false
        self.load_state_dict(filtered_state_dict, strict=False)
        print('load model weight from:', weights_path)

    def softce_clip_loss(self, logits_per_text, target_pseudo):
        caption_loss = self.ce_loss(logits_per_text, target_pseudo)
        image_loss = self.ce_loss(logits_per_text.T, target_pseudo)
        return (caption_loss + image_loss) / 2.0

    def ce_loss(self, pred_logit, ref):
        ce_loss = torch.nn.functional.cross_entropy(pred_logit, ref)
        return ce_loss

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def fit(self, datalaoders, epochs=30, lr=5e-4, weight_decay=1e-5, scheduler=True, warmup_epoch=1, store_num=5,
            transforms=None):

        # Set optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Set scheduler
        if scheduler:
            from flair.pretraining.utils import get_scheduler_per_iteration
            scheduler = get_scheduler_per_iteration(optimizer, lr, warmup_epoch, len(datalaoders))
        else:
            scheduler = None

        # Training along epochs
        epoch = 1
        while epoch <= epochs:

            # Train epoch
            loss_epoch = self.train_epoch(datalaoders, optimizer, scheduler, transforms, epoch)

            # Display epoch-wise loss
            print('Epoch=%d: ave_loss=%2.5f' % (epoch, loss_epoch))

            # Save model
            if epoch % store_num == 0:
                if self.out_path is not None:
                    if not os.path.isdir(self.out_path):
                        os.mkdir(self.out_path)
                    torch.save(self.state_dict(), self.out_path + self.vision_type + '_epoch' + str(epoch) + '.pth')

            # Update epoch
            epoch += 1

    def train_epoch(self, loader, optimizer, scheduler=None, transforms=None, epoch=1):
        self.train()
        max_grad_norm = 1
        scaler = torch.cuda.amp.GradScaler()  # ✅ 更通用，兼容低版本
        loss_ave = 0.0

        # Set iterator
        epoch_iterator = tqdm(
            loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        # Iterate trough training batches
        for step, (samples, targets) in enumerate(epoch_iterator):
            # Retrieve documents
            images = samples.to(self.device).to(torch.float32)
            # Create text tokens

            # 映射标签到描述（随机选一个描述，也可以拼接所有描述）
            text_list = []
            for label in targets.cpu().tolist():
                desc_list = definitions.get(str(label), ["unknown"])  # 防止找不到key
                # 拼接所有描述为一个字符串输入（也可以只用一个主描述）
                text_list.append(". ".join(desc_list))  # 或 random.choice(desc_list)

            # Tokenize text list
            text_tokens = self.text_model.tokenize(text_list)
            input_ids = text_tokens["input_ids"].to(self.device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(self.device).to(torch.long)

            # Create similarity matrix with soft labels as ground truth
            # Convert targets to list of integers on CPU
            target_list = targets.cpu().tolist()

            # 构建相似矩阵：标签相同则为1，否则为0
            coocurrence = np.array(
                [[1.0 if target_list[i] == target_list[j] else 0.0 for j in range(len(target_list))] for i in
                 range(len(target_list))],
                dtype=np.float32
            )

            # 避免除以0，加上一个极小值epsilon（如果某一行全是0）
            row_sums = coocurrence.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1e-8  # 防止除零错误

            # 归一化
            soft_labels = coocurrence / row_sums

            # 转成 tensor
            target = torch.tensor(soft_labels).to(self.device).to(torch.float32)

            # Forward
            with torch.amp.autocast(device_type='cuda'):

                # Image augmentation
                if transforms is not None:
                    images = transforms(images)

                # Forward vision and text encoder
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)

                # Compute similarity matrix and logits
                logits_per_image = self.compute_logits(img_embeds, text_embeds)
                logits_per_text = logits_per_image.t()

                # Compute cross-entropy loss
                loss = self.softce_clip_loss(logits_per_text, target)

            # Update model with scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Overall losses track
            loss_ave += loss.item()
            torch.cuda.empty_cache()

            # Set description
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) " % (epoch, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            # Update optimizer scheduler
            if scheduler is not None:
                scheduler.step()

        self.eval()
        return loss_ave / len(loader)

    def forward(self, image, text):
        self.eval()

        # Pre-process image
        # image = self.preprocess_image(image)
        # 对 batch 中每张图单独预处理
        image = torch.cat([self.preprocess_image(img) for img in image], dim=0)  # 最终 shape: (B, C, H, W)

        # Pre-process text
        text_input_ids, text_attention_mask = self.preprocess_text(text)

        # Forward vision and text encoder
        with torch.no_grad():
            img_embeds = self.vision_model(image)
            text_embeds = self.text_model(text_input_ids, text_attention_mask)

            # Compute similarity matrix and logits
            logits = self.compute_logits(img_embeds, text_embeds)

            # Compute probabilities
            probs = logits.softmax(dim=-1)

        # return probs.cpu().numpy(), logits.cpu().numpy()
        return probs, logits  # 不再转 numpy
    def preprocess_image(self, image):
        """
        输入: image 是单张图像的 tensor，形状为 (C, H, W)，值范围 [0, 255] 或 [0, 1]。
        输出: 预处理后的图像 tensor，形状为 (1, C, image_size, image_size)
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor.")

        # 确保是 float32 类型
        image = image.float()

        # 如果像素范围是 [0, 255]，则归一化到 [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        # 如果是灰度图，添加 channel 维度 (1, H, W)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 3 and image.shape[0] not in [1, 3]:
            raise ValueError(f"Unexpected channel dimension: {image.shape}")

        # 添加 batch 维度，变成 (1, C, H, W)
        image = image.unsqueeze(0)

        # Resize 保持比例，pad 到目标尺寸
        _, c, h, w = image.shape
        scale = max(h, w) / self.image_size
        new_h, new_w = int(h / scale), int(w / scale)

        image = torch.nn.functional.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

        pad_h = self.image_size - new_h
        pad_w = self.image_size - new_w

        # Pad (left, right, top, bottom) = (0, pad_w, 0, pad_h)
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h))

        # 返回到目标 device
        image = image.to(self.device)

        return image  # shape: (1, C, image_size, image_size)


    def preprocess_text(self, text):

        # Create text prompt
        prompts = [self.caption.replace("[CLS]", category) for category in text]

        # Create text tokens
        text_tokens = self.text_model.tokenize(prompts)
        input_ids = text_tokens["input_ids"].to(self.device).to(torch.long)
        attention_mask = text_tokens["attention_mask"].to(self.device).to(torch.long)

        return input_ids, attention_mask

    def compute_text_embeddings(self, categories, domain_knowledge=False):
        # Obtain text embeddings per class
        text_embeds_dict = {}
        for iKey in range(len(categories)):

            # Replace text prompt with expert knowledge descriptions
            if domain_knowledge and categories[iKey] in list(definitions.keys()):
                descriptions = definitions[categories[iKey]]
                if categories[iKey] not in descriptions:
                    descriptions.append(categories[iKey])
            else:
                descriptions = [categories[iKey]]

            # Forwards prompts trough text encoder
            with torch.no_grad():
                print(descriptions)
                descriptions = [self.caption.replace("[CLS]", iDescription) for iDescription in descriptions]
                text_token = self.text_model.tokenizer(descriptions, truncation=True, padding=True, return_tensors='pt')
                input_ids = text_token["input_ids"].to(self.device).to(torch.long)
                attention_mask = text_token["attention_mask"].to(self.device).to(torch.long)

                text_embeds = self.text_model(input_ids, attention_mask)

            text_embeds_dict[categories[iKey]] = text_embeds.mean(0).unsqueeze(0)

        text_embeds_dict = text_embeds_dict
        text_embeds = torch.concat(list(text_embeds_dict.values()))

        return text_embeds_dict, text_embeds


import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.shared(self.avg_pool(x))
        max = self.shared(self.max_pool(x))
        return self.sigmoid(avg + max)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


class VisionModel(torch.nn.Module):
    def __init__(self, vision_type='resnet', pretrained=True, proj_dim=512, proj_bias=False, projection=True,
                 norm=True):
        super().__init__()
        self.proj_dim = proj_dim

        # Assert vision encoders
        if vision_type not in ['resnet_v1', 'resnet_v2', 'efficientnet']:
            print("Vision model should be one of resnet/efficientnet... using resnet.")
            vision_type = "resnet_v1"

        # Set vision encoder architecture and pretrained weights
        if vision_type == "resnet_v1" or vision_type == "resnet_v2":
            # Set pretrained weights from Imagenet and get model
            if vision_type == "resnet_v1":
                weights = 'IMAGENET1K_V1' if pretrained else None
            elif vision_type == "resnet_v2":
                weights = 'IMAGENET1K_V2' if pretrained else None
            else:
                weights = 'IMAGENET1K_V1' if pretrained else None
            print("Pretrained weights: " + str(weights))
            self.model = CustomResNet(weights=weights)
            # Set number of extracted features
            self.vision_dim = 2048
            # Replace classifier by Identity layer
            self.model.fc = torch.nn.Identity()
        elif vision_type == "efficientnet":
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.model = torchvision.models.efficientnet_b7(weights=weights)
            self.vision_dim = 2096

        # Set output dimension
        if projection:
            self.out_dim = self.proj_dim
        else:
            self.out_dim = self.vision_dim

        # Set projection head
        self.projection_head_vision = ProjectionLayer(layer=torch.nn.Linear(self.vision_dim, self.proj_dim,
                                                                            bias=proj_bias),
                                                      projection=projection, norm=norm)

    def forward(self, pixel_values):
        # Forwards trough vision encoder
        embed = self.model(pixel_values)

        # Compute projection from vision embedding to multi-modal projection
        embed = self.projection_head_vision(embed)
        return embed

    def forward_vis(self, pixel_values):
        # Forwards trough vision encoder
        embed, feature_vis = self.model.forward_vis(pixel_values)

        # Compute projection from vision embedding to multi-modal projection
        embed = self.projection_head_vision(embed)
        return embed, feature_vis

    def forward_inter(self, pixel_values):
        # Forwards trough vision encoder
        embed, inter_1, inter_2, inter_3, inter_4 = self.model.forward_inter(pixel_values)

        # Compute projection from vision embedding to multi-modal projection
        embed = self.projection_head_vision(embed)
        return embed, inter_1, inter_2, inter_3, inter_4


from torchvision.models import ResNet50_Weights
from torchvision.models.resnet import ResNet, Bottleneck

class CustomResNet(ResNet):
    def __init__(self, weights=None):
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3])
        if weights:
            state_dict = ResNet50_Weights[weights].get_state_dict(progress=True)
            self.load_state_dict(state_dict)

        # 加入 CBAM 到每一层后
        self.cbam1 = CBAM(256)   # layer1输出维度
        self.cbam2 = CBAM(512)   # layer2输出维度
        self.cbam3 = CBAM(1024)  # layer3输出维度
        self.cbam4 = CBAM(2048)  # layer4输出维度

    def forward_inter(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        inter_1 = self.cbam1(self.layer1(x))
        inter_2 = self.cbam2(self.layer2(inter_1))
        inter_3 = self.cbam3(self.layer3(inter_2))
        inter_4 = self.cbam4(self.layer4(inter_3))

        x = self.avgpool(inter_4)
        x = torch.flatten(x, 1)

        return x, inter_1, inter_2, inter_3, inter_4



local_path = "./Bio_ClinicalBERT_tokenizer"  # 设定保存路径
save_directory = "./Bio_ClinicalBERT"  # 存放路径
checkpoint_dict = "./checkpoint/bert2/epoch_latest.pt"
class TextModel(torch.nn.Module):
    def __init__(self, bert_type='emilyalsentzer/Bio_ClinicalBERT', proj_dim=512, proj_bias=False, projection=True,
                 norm=True):
        super().__init__()


        config = BertConfig.from_pretrained(save_directory, output_hidden_states=True)  # 需匹配你的BERT架构
        self.bert_model = AutoModel.from_config(config)  # 先用 config 初始化模型
        checkpoint = torch.load(checkpoint_dict, map_location=torch.device("cpu"))  # 加载 checkpoint
        self.bert_model.load_state_dict(checkpoint, strict=False)  # 加载权重
        # Set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        self.tokenizer.model_max_length = 77
        # Set projection head
        self.line_head_text = ProjectionLayer(layer=torch.nn.Linear(768, proj_dim, bias=proj_bias),
                                                    projection=projection, norm=norm)

    def tokenize(self, prompts_list):
        text_tokens = self.tokenizer(prompts_list, truncation=True, padding=True, return_tensors='pt')
        return text_tokens

    def forward(self, input_ids, attention_mask):

        # Forwards trough text encoder
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)

        # Combine last feature layers to compute text embedding
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],
                                          output['hidden_states'][-1]])
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)

        # Compute projection from text embedding to multi-modal projection
        embed = self.line_head_text(embed)
        return embed


# 根据参数决定是否对输入进行归一化和投影操作：
# 初始化时设置是否应用投影和归一化。
# 前向传播时，先判断是否对输入进行模态归一化，再判断是否应用投影层，最后判断是否对投影结果进行归一化。
class ProjectionLayer(torch.nn.Module):
    def __init__(self, layer, projection=True, norm=True):
        super().__init__()

        self.apply_projection = projection
        self.norm_modality = bool(projection * norm)
        self.norm_projection = norm
        self.projection = layer

    def forward(self, x):

        if self.norm_modality:
            x = x / x.norm(dim=-1, keepdim=True)

        if self.apply_projection:
            x = self.projection(x)
            if self.norm_projection:
                x = x / x.norm(dim=-1, keepdim=True)

        return x