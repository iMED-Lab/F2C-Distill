import sys

import torch
import torch.nn as nn
from flair import FLAIRModel
from graph.graph_encoder import Graph_Encoder

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

skg = {'nodes': nodes, 'node_inds': torch.tensor(node_inds, dtype=torch.long),
       'node_labels': torch.tensor(node_labels, dtype=torch.long), 'node_relations': torch.tensor(node_relations, dtype=torch.long)}

class FLAIRMultiLayer(nn.Module):
    def __init__(self, args, device, modality='fundus'):
        super().__init__()
        self.modality = modality
        with open("./concepts/concept.txt", "r", encoding="utf-8") as f:
            self.raw_concepts = [line.strip() for line in f if line.strip()]
        
        # self.flair_model = FLAIRModel(device=device, from_checkpoint=True)
        self.flair_model = FLAIRModel(device=device, from_checkpoint=True)
        self.latent_dim = self.flair_model.vision_model.proj_dim
        self.graph_encoder = Graph_Encoder(device=device)
        self.concept_classifier = nn.Linear(93, args.n_classes)
        self.graph_nlp = nn.Linear(768, 512)

        with torch.no_grad():
            text_input_ids, text_attention_mask = self.flair_model.preprocess_text(self.raw_concepts)
            self.embed_concepts = self.flair_model.text_model(text_input_ids, text_attention_mask)
            
        # self.project_1 = nn.Linear(256, self.latent_dim)
        # self.project_2 = nn.Linear(512, self.latent_dim)
        # self.project_3 = nn.Linear(1024, self.latent_dim)
        self.project_4 = nn.Linear(2048, self.latent_dim)

    def forward(self, image):

        embed_images = self.flair_model.vision_model(image)
        skg['nodes'] = skg['nodes'].replace('-', ' ')
        graph_knowledge = self.graph_encoder(skg, image.device)
        # graph_knowledge在第一个维度上取平均姜维
        graph_knowledge = torch.mean(graph_knowledge, dim=0)
        graph_knowledge = self.graph_nlp(graph_knowledge)
        graph_knowledge = graph_knowledge / graph_knowledge.norm(dim=-1, keepdim=True)

        knowledge = torch.cat([graph_knowledge, self.embed_concepts], dim=0)
        concept_sim = embed_images @ knowledge.t()

        sim = self.concept_classifier(concept_sim)
        return sim

    
    def forward_distill(self, image):
        if torch.isnan(image).any() or torch.isinf(image).any():
            print("🔥 Input image contains NaN or Inf!")
            sys.exit(1)
        if self.modality == 'ffa':
            with torch.no_grad():
                embed_images, inter_1, inter_2, inter_3, inter_4 = self.flair_model.vision_model.forward_inter(image)
        elif self.modality == 'fundus':
            embed_images, inter_1, inter_2, inter_3, inter_4 = self.flair_model.vision_model.forward_inter(image)
        # inter_1 = self.project_1(inter_1)
        # inter_2 = self.project_2(inter_2)
        # inter_3 = self.project_3(inter_3)
        inter_4 = torch.nn.functional.adaptive_avg_pool2d(inter_4, (1, 1))  # -> [B, 2048, 1, 1]
        inter_4 = inter_4.view(inter_4.size(0), -1)  # -> [B, 2048]
        inter_4 = self.project_4(inter_4)  # -> [B, latent_dim]


        # concept_sim_1 = inter_1 @ self.embed_concepts.t()
        # concept_sim_2 = inter_2 @ self.embed_concepts.t()
        # concept_sim_3 = inter_3 @ self.embed_concepts.t()
        # concept_sim_4 = inter_4 @ self.embed_concepts.t()
        # for i in range(image.size(0)):
        #     knowledge_skg['nodes'][i] = knowledge_skg['nodes'][i].replace('-', ' ')
        skg['nodes'] = skg['nodes'].replace('-', ' ')
        graph_knowledge = self.graph_encoder(skg, image.device)
        # graph_knowledge在第一个维度上取平均姜维
        graph_knowledge = torch.mean(graph_knowledge, dim=0)
        graph_knowledge = self.graph_nlp(graph_knowledge)
        graph_knowledge = graph_knowledge / graph_knowledge.norm(dim=-1, keepdim=True)
        # graph_knowledge 与 embed_concepts concat
        knowledge = torch.cat([graph_knowledge, self.embed_concepts], dim=0)
        embed_images = self.flair_model.vision_model(image)
        # device = embed_images.device  # 确保用的是模型主设备，比如 cuda:3
        #
        # mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True).to(device)
        #
        # query = embed_images.unsqueeze(0).to(device)  # [1, 64, 512]
        # key_value = graph_knowledge.unsqueeze(0).to(device)  # [1, 81, 512]

        # attn_output, _ = mha(query, key_value, key_value)  # [1, 64, 512]
        # attn_output = attn_output.squeeze(0)  # [64, 512]

        concept_sim_4 = inter_4 @ knowledge.t()
        concept_sim = embed_images @ knowledge.t()
        sim = self.concept_classifier(concept_sim)

        # return sim, torch.stack([concept_sim, concept_sim_1, concept_sim_2, concept_sim_3, concept_sim_4], dim=1)
        return sim, torch.stack([concept_sim, concept_sim_4], dim=1)


    
    def get_concepts_feat(self):
        return self.embed_concepts
    