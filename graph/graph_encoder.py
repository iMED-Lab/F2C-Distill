import copy
import math
from pickle import STACK_GLOBAL
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, logging, BertConfig
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer

# 每个层级内不同目标之间互相独立,但是与全局节点连接
node = [
    'eye', 'normal', 'other finding',
    'age-related macular degeneration', 'retinal pigment epithelium', 'Patchy dark brown pigment deposits',
    'macular area', 'scattered clustered dots', 'punctate high fluorescence', 'patchy high fluorescence',
    'Central Serous Choroidal Retinopathy', 'macular area', 'high fluorescence spots', 'Ink stain leakage',
    'chimney leakage', 'Fluorescein accumulation', 'retina', 'subcutaneous fluid accumulation', 'circular protrusion',
    'quasi circular protrusion', 'orange red surface', 'grayish yellow surface',
    'diabetes retinopathy', 'retina', 'Spot hemorrhage', 'patchy hemorrhage', 'fibroproliferative membrane',
    'punctate high fluorescence', 'hemorrhagic masking fluorescence', 'Microaneurysm', 'Non-Perfusion',
    'Neovascularization', 'Neovascularization Elsewhere',
    'retinal vein occlusion', 'retina', 'punctate bleeding', 'patchy bleeding', 'flame like bleeding', 'Vein',
    'Obstruction', 'delayed filling', 'tortuous dilation', 'fluorescence masking'
]  # 43 nodes
nodes = '-'.join(node)
node_inds = [0, 5, 6, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             4, 4, 4, 4, 4, 4, 4, 4, 4, 4]  # 43 nodes
node_labels = [0, 6, 7, 1, 2, 2, 2, 2, 2, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
               1, 5, 5, 5, 5, 5, 5, 5, 5, 5]  # labels for each node
node_inds = [each + 1 for each in node_inds]
node_labels = [each + 1 for each in node_labels]  # Standard code uses 1-based indexing, so +1

node_relations = list(range(len(node_inds)))
node_relations = [each + 1 for each in node_relations]

skg = {'nodes': nodes, 'node_inds': node_inds, 'node_labels': node_labels, 'node_relations': node_relations}


def init_tokenizer(args):
    # 使用PubMedBert分词器,否则分词器参数需要大量调整
    tokenizer = AutoTokenizer.from_pretrained(args.PubMed_BERT, local_files_only=True)
    return tokenizer


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 确保 mask 在与 query 相同的设备上
    if mask is not None:
        mask = mask.to(query.device).float()
        scores = scores.float()
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 确保 x 在与模型参数相同的设备上
        x = x.to(self.norm.gamma.device)
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 确保 x 在与 self.gamma 相同的设备上
        x = x.to(self.gamma.device)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    级别编码-位置编码

    在知识图谱编码器中，位置编码的作用不仅仅是为序列中的元素添加位置信息，还需要考虑图结构中的节点类型和节点之间的关系。
    生成基础位置编码：使用传统的正弦和余弦函数生成基础位置编码 pe，通过 x_node_inds 将基础位置编码分配给不同类型的节点。
    位置编码的分配是动态的，可以根据节点的类型或位置进行调整。位置编码需要适应图的拓扑结构，而不是简单的线性序列。

    final_pe[:, 0, :] 会被赋值为 tmp_pe[:, 0]（节点类型 0 的位置编码）。
    final_pe[:, 1, :] 会被赋值为 tmp_pe[:, 1]（节点类型 1 的位置编码）。
    final_pe[:, 2, :] 会被赋值为 tmp_pe[:, 2]（节点类型 2 的位置编码）。
    final_pe[:, 3, :] 和 final_pe[:, 4, :] 都会被赋值为 tmp_pe[:, 3]（节点类型 3 的位置编码）
    位置编码类，用于为知识图谱中的每个节点分配不同类型的位置编码。
    """

    def __init__(self, d_model, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(90, d_model)
        position = torch.arange(0, 90).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, x_node_inds, device):
        final_pe = torch.zeros_like(x)
        for num in range(len(x_node_inds)):
            node_type = x_node_inds[num]
            tmp = self.pe[:, node_type].squeeze(0)
            final_pe[:, num, :] = tmp
        x = x + final_pe.to(device)
        return self.dropout(x)


def generate_mask(inds, labels, relation, x):
    """
    inds、labels 和 relation 都是形状为 [num_entities, nbatches] 的二维张量。
    x 是一个形状为 [nbatches, ...] 的多维张量。
    mask 是一个形状为 [nbatches, num_entities, num_entities] 的三维张量。
    """
    device = x.device  # 获取 x 所在的设备
    inds = inds.unsqueeze(-1)
    labels = labels.unsqueeze(-1)
    relation = relation.unsqueeze(-1)
    inds = inds.float().to(device)  # 将 inds 转换到 x 的设备上
    labels = labels.float().to(device)  # 将 labels 转换到 x 的设备上
    relation = relation.float().to(device)  # 将 relation 转换到 x 的设备上
    # print("labels shape:", labels.shape)
    # print("inds shape:", inds.shape)
    # print("relation shape:", relation.shape)

    nbatches = x.size(0)
    mask = torch.zeros([nbatches, inds.size(0), inds.size(0)], device=device)  # 在 x 的设备上创建 mask

    for i in range(inds.size(0)):
        for j in range(inds.size(0)):
            if i == 1 or j == 1:
                mask[:, i, j] = 1
            if labels[i, :].equal(labels[j, :]) or inds[i, :].equal(inds[j, :]) or relation[i, :].equal(relation[j, :]):
                if labels[i, :].equal(torch.zeros(nbatches, device=device)) == False and \
                        inds[i, :].equal(torch.zeros(nbatches, device=device)) == False and \
                        relation[i, :].equal(torch.zeros(nbatches, device=device)) == False:
                    mask[:, i, j] = 1
    return mask


def process_knowledge_skg(knowledge_skg):
    """
    原本是一个列表内包含32个tensor节点,每个节点包含64列表. 转换之后一个列表包含32个列表,每个列表64个. tensor之后变成tensor整体,可索引shape等.
    Before-knowledge_skg['node_inds'] [tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),...,
    After-knowledge_skg['node_inds'] [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], ...,
    """
    # print("Before-knowledge_skg['node_inds']",knowledge_skg['node_inds'])
    knowledge_skg['node_inds'] = [each.tolist() for each in knowledge_skg['node_inds']]
    # print("After-knowledge_skg['node_inds']",knowledge_skg['node_inds'])
    knowledge_skg['node_labels'] = [each.tolist() for each in knowledge_skg['node_labels']]
    knowledge_skg['node_relations'] = [each.tolist() for each in knowledge_skg['node_relations']]

    knowledge_skg['node_inds'] = torch.tensor(knowledge_skg['node_inds'])
    # print("Before-knowledge_skg['node_inds']",knowledge_skg['node_inds'])
    knowledge_skg['node_labels'] = torch.tensor(knowledge_skg['node_labels'])
    knowledge_skg['node_relations'] = torch.tensor(knowledge_skg['node_relations'])
    """
    转换后的inds,label,relation 均为[nodes_nums,batchsize],直接作为图编码器输入即可. 
    """
    return knowledge_skg


local_path = "./Bio_ClinicalBERT_tokenizer"  # 设定保存路径
save_directory = "./Bio_ClinicalBERT"  # 存放路径
checkpoint_dict = "./checkpoint/bert/epoch_latest.pt"
class Graph_Encoder(nn.Module):
    # 动态图编码器，交叉注意力机制是文章中的graph attention
    def __init__(self,  device, dropout=0.1):
        super(Graph_Encoder, self).__init__()
        c = copy.deepcopy
        self.dropout = nn.Dropout(p=dropout)
        self.encoder = Encoder(
            EncoderLayer(768, c(MultiHeadedAttention(6, 768)), c(PositionwiseFeedForward(768, 1024, 0.1)), dropout),
            2)  # 一个完整的Tranformer单元（MH,MLP），修改了适合图的mask
        # self.embedds = Embeddings(27,768)

        self.pe = PositionalEncoding(768, dropout)  # 级别编码-位置编码

        # self.bert = BertModel.from_pretrained(args.BERT_INIT_WEIGHT).to("cuda")
        print(f"Loading model from checkpoint: yuxunlian")
        config = BertConfig.from_pretrained(save_directory, output_hidden_states=True)  # 需匹配你的BERT架构
        self.bert = AutoModel.from_config(config)  # 先用 config 初始化模型
        checkpoint = torch.load(checkpoint_dict, map_location=torch.device("cpu"))  # 加载 checkpoint
        self.bert.load_state_dict(checkpoint, strict=False)  # 加载权重
        self.bert.to(device)

        #  tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)

        for param in self.bert.parameters():
            param.requires_grad_(False)

        # 只打印可训练参数
        # print_model_parameters(self.text_encoder, print_all=False) # []
        print("BERT parameters have been frozen.")
        # med_config = 'configs/tag_config_sci.json'
        # encoder_config = BertConfig.from_json_file(med_config)
        # encoder_config.encoder_width = 768
        # self.bert = BertModel(config=BertConfig.from_json_file('configs/tag_config_sci_down.json'), add_pooling_layer=False, SKG_know=True)

    def forward(self, x, device):
        """
        输入是：skg = {'nodes':nodes, 'node_inds':node_inds, 'node_labels':node_labels, 'node_relations': node_relations} 字典格式
        """
        # 将skg处理为整体的tensor格式
        x = process_knowledge_skg(x)
        # 知识图谱编码器_能够处理知识图谱的节点，连接，标签等信息，但是并未使用图神经网络
        x_nodes = x['nodes']
        x_node_inds = x['node_inds'].to(device)
        x_node_labels = x['node_labels'].to(device)
        x_relation = x['node_relations'].to(device)
        # print("x_node_inds.shape:",x_node_inds.shape) #x_node_inds.shape: torch.Size([36, 64]) x_node_inds.type: <class 'torch.Tensor'>
        # print("x_node_inds.type:",type(x_node_inds)).
        x = self.get_tag_embeds(x_nodes,
                    device)  # [batch_size, max_len, embedding_dim]  # 实体嵌入-BERT模型,BERT模型初始化,每个节点只包含一个嵌入向量
        # print('1')
        # print("x.shape:", x.shape)
        x = self.pe(x, x_node_inds, device)  # 位置编码表示哪些节点是相同语义，所以赋值为相同的位置编码
        # print('2')
        x = self.bert.embeddings.LayerNorm(x)
        # print('3')
        x = self.bert.embeddings.dropout(x)
        # print('4')
        x = self.encoder(x, generate_mask(x_node_inds, x_node_labels, x_relation, x))  # 交互掩码设计（图的话根据边权关系决定哪些节点交互）
        # print('5')
        return self.dropout(x)  # [batch_size, max_len, embedding_dim]



    def get_tag_embeds(self, nodes, device, nodes_ori=node):
        """
        为给定的知识图谱节点获取嵌入向量，并处理多单词节点的嵌入平均。
        """
        # 统计每个节点的单词数量
        """
        print(len(nodes)): 64
        print("nodes", nodes)
        nodes ['Cornea Epithelium Nerves Nerve Break Nerve Swelling Nerve Loss Dendritic Cells Inflammatory Cells Cell Defect Cell Proliferation Epithelial Cells Cell Hyperplasia Cell Hypoplasia Anterior Elastic Layer Layer Thickness Layer Thickening Layer Irregularity Layer Rupture Wrinkling Stroma Stromal Cells Collagen Fibers Fiber Disarray Fiber Thickening Stromal Lakes Lake Hyperplasia Lake Hypoplasia Posterior Elastic Layer Detachment Endothelium Endothelial Cells Cell Enlargement Cell Reduction Vacuolization Vacuole Hyperplasia Vacuole Hypoplasia',
        'Cornea Epithelium Nerves Nerve Break Nerve Swelling Nerve Loss Dendritic Cells Inflammatory Cells Cell Defect Cell Proliferation Epithelial Cells Cell Hyperplasia Cell Hypoplasia Anterior Elastic Layer Layer Thickness Layer Thickening Layer Irregularity Layer Rupture Wrinkling Stroma Stromal Cells Collagen Fibers Fiber Disarray Fiber Thickening Stromal Lakes Lake Hyperplasia Lake Hypoplasia Posterior Elastic Layer Detachment Endothelium Endothelial Cells Cell Enlargement Cell Reduction Vacuolization Vacuole Hyperplasia Vacuole Hypoplasia', ...]
        """
        token_counts = [len(node.split()) for node in nodes_ori]
        # print("token_counts:{}".format(token_counts))
        # 将所有节点列表连接成一个字符串, 只有字符串才能够送入分词器编码
        # 对输入进行编码
        encoded_input = self.tokenizer(nodes, padding='max_length', truncation=True, max_length=220,
                              return_tensors="pt").to(device)

        # 获取输入的嵌入向量
        embeddings = self.bert.embeddings.word_embeddings(encoded_input.input_ids)
        # print("embedding:{}".format(embeddings.shape))  # [bs, max_len, embeding]
        # 初始化结果列表
        result_embeddings = []

        start_index = 0

        # 根据每个节点的单词数量进行切片和平均
        for count in token_counts:
            if count == 1:
                # 单个单词节点，直接添加
                result_embeddings.append(embeddings[:, start_index : start_index + 1, :])
            else:
                # 多单词节点，计算平均值
                result_embeddings.append(torch.mean(embeddings[:, start_index : start_index + count, :], dim=1, keepdim=True))
            start_index += count

        # 拼接结果
        result_embeddings = torch.cat(result_embeddings, dim=1)
        # print("result_embeddings",result_embeddings.shape) #[bs, nodes_num, embeding]
        # print("result_embeddings.shape:", result_embeddings.shape)
        # print("len(nodes_ori):", len(nodes_ori))

        assert result_embeddings.size(1) == len(nodes_ori),"result_embeddings numbers must be equal to nodes numbers!"

        return result_embeddings
