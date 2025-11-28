# F2C-Distill

Official implementation of F2C-Distill, a multimodal retinal diagnosis framework that integrates clinical knowledge from 10 common retinal diseases and leverages cross-modal distillation between FFA and Fundus modalities.

## 1. A2_KEBERT Fine-tuning

File: RetinaBERT/A2_KEBERT/main.py
Used for fine-tuning the ophthalmology-specific BERT model (A2-KEBERT).
This module learns medical semantics and clinical concept structures from ophthalmic reports.

Key Functions

Uses UMLS entity definitions and clinical triples (h, r, t).

Constructs textual contrastive learning tasks.

Learns medical entity semantics and structured knowledge embeddings.

Produces an Ophthalmologist Encoder (KE-BERT), capable of clinical entity understanding.

Provides semantic embeddings to support the knowledge graph and multimodal image diagnosis pipeline.

## 2. Pretraining the FFA Teacher Model

File: main_ffa.py
This module pretrains the FFA modality teacher model, which serves as a high-quality supervisory signal for later distillation.

Key Functions

Trains on FFA angiography images.

Learns detailed retinal vascular structure and fine-grained lesion patterns.

Aligns visual features with the Ophthalmology BERT concept space.

Generates a strong image–semantic representation used as the teacher during distillation.

## 3. Distilling the FCT Fundus Student Model

File: main_dis.py
This module distills knowledge from the FFA teacher into a Fundus-based student model (FCT).

Key Functions

Transfers FFA-derived structural and semantic knowledge to Fundus images.

Enables Fundus models to benefit from the richer lesion visibility of FFA.

Enhances performance on:

Early lesion detection

Small structure recognition

Complex, multi-lesion reasoning
