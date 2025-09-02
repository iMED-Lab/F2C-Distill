from graph.graph_encoder import Graph_Encoder
import torch
import torch.nn as nn
from flair import FLAIRModel

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

class FLAIRConceptClassifier(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        with open("./concepts/concept.txt", "r", encoding="utf-8") as f:
            self.raw_concepts = [line.strip() for line in f if line.strip()]
        self.flair_model = FLAIRModel(device=device, from_checkpoint=True)

        self.graph_encoder = Graph_Encoder(device=device)
        self.concept_classifier = nn.Linear(93, args.n_classes)
        self.graph_nlp = nn.Linear(768, 512)
        text_input_ids, text_attention_mask = self.flair_model.preprocess_text(self.raw_concepts)
        with torch.no_grad():
            self.embed_concepts = self.flair_model.text_model(text_input_ids, text_attention_mask)
        self.latent_dim = self.flair_model.vision_model.proj_dim

    def forward(self, image):
        skg['nodes'] = skg['nodes'].replace('-', ' ')
        graph_knowledge = self.graph_encoder(skg, image.device)
        # graph_knowledge在第一个维度上取平均
        graph_knowledge = torch.mean(graph_knowledge, dim=0)
        graph_knowledge = self.graph_nlp(graph_knowledge)
        graph_knowledge = graph_knowledge / graph_knowledge.norm(dim=-1, keepdim=True)
        embed_images = self.flair_model.vision_model(image)
        knowledge = torch.cat([graph_knowledge, self.embed_concepts], dim=0)
        concept_sim = embed_images @ knowledge.t()
        sim = self.concept_classifier(concept_sim)
        return sim
