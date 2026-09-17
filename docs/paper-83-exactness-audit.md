# Paper-83 Exactness Audit

Per-paper implementation-exactness inventory over the frozen 83. An audit result below 83/83 ready is a legitimate outcome: the report, artifacts, gap queue, and tests are still produced.

## Summary

```
Total: 83
Ready: 83
Blocked: 0
Out of scope: 0
Papers passing all 17 checks: 83
```

Training remains locked: this audit inventories evidence only and never starts model training.

## Status vocabulary

Only `implementation_ready` grants implementation status. Blocked statuses use the fixed blocker categories; component-level terms (`covered`, `mapped`, `certified_adapter`) are not readiness states.

## Results by status

| Status | Papers |
| --- | --- |
| `implementation_ready` | 83 |

## Failing checks across the campaign

| Check | Failing papers |
| --- | --- |
| paper membership valid | 0 |
| MethodProfile valid | 0 |
| mechanism evidence available | 0 |
| PaperImplementationSpec complete | 0 |
| not generic-only | 0 |
| not alias-only | 0 |
| not metadata-only | 0 |
| not a no-op | 0 |
| paper-specific composition present | 0 |
| runtime hook real | 0 |
| source/runtime fingerprint present | 0 |
| unit tests | 0 |
| non-mock smoke | 0 |
| compatibility tests | 0 |
| rollback path | 0 |
| shared primitives referenced, not copied | 0 |
| core mechanism covered | 0 |

## Blocked papers by category

## Gap queue

```
Gap entries: 0
Blocked papers: 0
```

Every gap entry carries a recommended fix, an estimated scope, and its dependency; see `artifacts/paper_83_gap_queue.yaml`.

## Ready papers

- `arxiv:2103.14259` — OTA: Optimal Transport Assignment for Object Detection
- `arxiv:2104.14082` — Pseudo-IoU: Improving Label Assignment in Anchor-Free Object Detection
- `arxiv:2107.08430` — YOLOX: Exceeding YOLO Series in 2021
- `arxiv:2108.07755` — TOOD: Task-aligned One-stage Object Detection
- `arxiv:2109.05986` — Mutual Supervision for Dense Object Detection
- `arxiv:2203.16250` — PP-YOLOE: An Evolved Version of YOLO
- `arxiv:2208.00817` — DSLA: Dynamic Smooth Label Assignment for Efficient Anchor-Free Object Detection
- `arxiv:2210.11539` — ConfMix: Unsupervised Domain Adaptation for Object Detection via Confidence-Based Mixing
- `arxiv:2212.07784` — RTMDet: An Empirical Study of Designing Real-Time Object Detectors
- `arxiv:2301.01019` — Correlation Loss: Enforcing Correlation between Classification and Localization
- `arxiv:2303.13853` — 2PCNet: Two-Phase Consistency Training for Day-to-Night Unsupervised Domain Adaptive Object Detection
- `arxiv:2303.14404` — Bridging Precision and Confidence: A Train-Time Loss for Calibrating Object Detection
- `arxiv:2309.11331` — Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism
- `arxiv:2503.23220` — Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection
- `arxiv:2507.00721` — UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement
- `arxiv:2603.12409` — ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection
- `arxiv:2603.18541` — Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection
- `arxiv:2603.18757` — DA-Mamba: Learning Domain-Aware State Space Model for Global-Local Alignment in Domain Adaptive Object Detection
- `arxiv:2603.28182` — A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning Matters and Parallel Decoder Helps
- `cvf:cvpr2021:Dai_General_Instance_Distillation_for_Object_Detection` — General Instance Distillation for Object Detection
- `cvf:cvpr2021:Guo_Distilling_Object_Detectors_via_Decoupled_Features` — Distilling Object Detectors via Decoupled Features
- `cvf:cvpr2021:Hu_Dense_Relation_Distillation_With_Context-Aware_Aggregation_for_Few-Shot_Object_Detection` — Dense Relation Distillation With Context-Aware Aggregation for Few-Shot Object Detection
- `cvf:cvpr2021:VS_MeGA-CDA_Memory_Guided_Attention_for_Category-Aware_Unsupervised_Domain_Adaptive_Object` — MeGA-CDA: Memory Guided Attention for Category-Aware Unsupervised Domain Adaptive Object Detection
- `cvf:cvpr2021:Zhang_RPN_Prototype_Alignment_for_Domain_Adaptive_Object_Detector` — RPN Prototype Alignment for Domain Adaptive Object Detector
- `cvf:cvpr2022:Feng_Overcoming_Catastrophic_Forgetting_in_Incremental_Object_Detection_via_Elastic_Response` — Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation
- `cvf:cvpr2022:Guo_Scale-Equivalent_Distillation_for_Semi-Supervised_Object_Detection` — Scale-Equivalent Distillation for Semi-Supervised Object Detection
- `cvf:cvpr2022:He_Cross_Domain_Object_Detection_by_Target-Perceived_Dual_Branch_Distillation` — Cross Domain Object Detection by Target-Perceived Dual Branch Distillation
- `cvf:cvpr2022:Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection` — Cross-Domain Adaptive Teacher for Object Detection
- `cvf:cvpr2022:Li_SIGMA_Semantic-Complete_Graph_Matching_for_Domain_Adaptive_Object_Detection` — SIGMA: Semantic-Complete Graph Matching for Domain Adaptive Object Detection
- `cvf:cvpr2022:Wu_Single-Domain_Generalized_Object_Detection_in_Urban_Scene_via_Cyclic-Disentangled_Self-Distillation` — Single-Domain Generalized Object Detection in Urban Scene via Cyclic-Disentangled Self-Distillation
- `cvf:cvpr2022:Wu_Target-Relevant_Knowledge_Preservation_for_Multi-Source_Domain_Adaptive_Object_Detection` — Target-Relevant Knowledge Preservation for Multi-Source Domain Adaptive Object Detection
- `cvf:cvpr2022:Zhao_Task-Specific_Inconsistency_Alignment_for_Domain_Adaptive_Object_Detection` — Task-Specific Inconsistency Alignment for Domain Adaptive Object Detection
- `cvf:cvpr2022:Zheng_Localization_Distillation_for_Dense_Object_Detection` — Localization Distillation for Dense Object Detection
- `cvf:cvpr2022:Zhou_Multi-Granularity_Alignment_Domain_Adaptation_for_Object_Detection` — Multi-Granularity Alignment Domain Adaptation for Object Detection
- `cvf:cvpr2023:Cao_Contrastive_Mean_Teacher_for_Domain_Adaptive_Object_Detectors` — Contrastive Mean Teacher for Domain Adaptive Object Detectors
- `cvf:cvpr2023:Gao_AsyFOD_An_Asymmetric_Adaptation_Paradigm_for_Few-Shot_Domain_Adaptive_Object` — AsyFOD: An Asymmetric Adaptation Paradigm for Few-Shot Domain Adaptive Object Detection
- `cvf:cvpr2023:Liu_CIGAR_Cross-Modality_Graph_Reasoning_for_Domain_Adaptive_Object_Detection` — CIGAR: Cross-Modality Graph Reasoning for Domain Adaptive Object Detection
- `cvf:cvpr2023:VS_Instance_Relation_Graph_Guided_Source-Free_Domain_Adaptive_Object_Detection` — Instance Relation Graph Guided Source-Free Domain Adaptive Object Detection
- `cvf:cvpr2023:Wang_Object-Aware_Distillation_Pyramid_for_Open-Vocabulary_Object_Detection` — Object-Aware Distillation Pyramid for Open-Vocabulary Object Detection
- `cvf:cvpr2023:Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector` — ScaleKD: Distilling Scale-Aware Knowledge in Small Object Detector
- `cvf:cvpr2024:Du_Boosting_Object_Detection_with_Zero-Shot_Day-Night_Domain_Adaptation` — Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation
- `cvf:cvpr2024:Kennerley_CAT_Exploiting_Inter-Class_Dynamics_for_Domain_Adaptive_Object_Detection` — CAT: Exploiting Inter-Class Dynamics for Domain Adaptive Object Detection
- `cvf:cvpr2024:Nakamura_Active_Domain_Adaptation_with_False_Negative_Prediction_for_Object_Detection` — Active Domain Adaptation with False Negative Prediction for Object Detection
- `cvf:cvpr2024:Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection` — CrossKD: Cross-Head Knowledge Distillation for Object Detection
- `cvf:cvpr2024:Yang_Active_Object_Detection_with_Knowledge_Aggregation_and_Distillation_from_Large` — Active Object Detection with Knowledge Aggregation and Distillation from Large Models
- `cvf:cvpr2025:Li_SEEN-DA_SEmantic_ENtropy_guided_Domain-aware_Attention_for_Domain_Adaptive_Object` — SEEN-DA: SEmantic ENtropy guided Domain-aware Attention for Domain Adaptive Object Detection
- `cvf:cvpr2025:Liu_Distinguish_Then_Exploit_Source-free_Open_Set_Domain_Adaptation_via_Weight` — Distinguish Then Exploit: Source-free Open Set Domain Adaptation via Weight Barcode Estimation and Sparse Label Assignment
- `cvf:iccv2021:Chen_Deep_Structured_Instance_Graph_for_Distilling_Object_Detectors` — Deep Structured Instance Graph for Distilling Object Detectors
- `cvf:iccv2021:Chen_Dual_Bipartite_Graph_Learning_A_General_Approach_for_Domain_Adaptive` — Dual Bipartite Graph Learning: A General Approach for Domain Adaptive Object Detection
- `cvf:iccv2021:Tian_Knowledge_Mining_and_Transferring_for_Domain_Adaptive_Object_Detection` — Knowledge Mining and Transferring for Domain Adaptive Object Detection
- `cvf:iccv2021:Yao_G-DetKD_Towards_General_Distillation_Framework_for_Object_Detectors_via_Contrastive` — G-DetKD: Towards General Distillation Framework for Object Detectors via Contrastive and Semantic-Guided Feature Imitation
- `cvf:iccv2021:Yao_Multi-Source_Domain_Adaptation_for_Object_Detection` — Multi-Source Domain Adaptation for Object Detection
- `cvf:iccv2023:Gao_CSDA_Learning_Category-Scale_Joint_Feature_for_Domain_Adaptive_Object_Detection` — CSDA: Learning Category-Scale Joint Feature for Domain Adaptive Object Detection
- `cvf:iccv2023:Kang_Alleviating_Catastrophic_Forgetting_of_Incremental_Object_Detection_via_Within-Class_and` — Alleviating Catastrophic Forgetting of Incremental Object Detection via Within-Class and Between-Class Knowledge Distillation
- `cvf:iccv2023:Lao_UniKD_Universal_Knowledge_Distillation_for_Mimicking_Homogeneous_or_Heterogeneous_Object` — UniKD: Universal Knowledge Distillation for Mimicking Homogeneous or Heterogeneous Object Detectors
- `cvf:iccv2023:Wu_Spatial_Self-Distillation_for_Object_Detection_with_Inaccurate_Bounding_Boxes` — Spatial Self-Distillation for Object Detection with Inaccurate Bounding Boxes
- `cvf:iccv2023:Yang_Bridging_Cross-task_Protocol_Inconsistency_for_Distillation_in_Dense_Object_Detection` — Bridging Cross-task Protocol Inconsistency for Distillation in Dense Object Detection
- `cvf:iccv2023:Zhao_Masked_Retraining_Teacher-Student_Framework_for_Domain_Adaptive_Object_Detection` — Masked Retraining Teacher-Student Framework for Domain Adaptive Object Detection
- `cvf:iccv2025:Cui_Debiased_Teacher_for_Day-to-Night_Domain_Adaptive_Object_Detection` — Debiased Teacher for Day-to-Night Domain Adaptive Object Detection
- `cvf:iccv2025:He_Dual-Rate_Dynamic_Teacher_for_Source-Free_Domain_Adaptive_Object_Detection` — Dual-Rate Dynamic Teacher for Source-Free Domain Adaptive Object Detection
- `ecva:eccv2022:1356` — Prediction-Guided Distillation for Dense Object Detection
- `ecva:eccv2022:2285` — HEAD: HEtero-Assists Distillation for Heterogeneous Object Detectors
- `ecva:eccv2022:2717` — Distilling Object Detectors with Global Knowledge
- `ecva:eccv2022:3523` — Multi-faceted Distillation of Base-Novel Commonality for Few-Shot Object Detection
- `ecva:eccv2022:3958` — Unsupervised Domain Adaptation for One-Stage Object Detector Using Offsets to Bounding Box
- `ecva:eccv2022:6004` — Few-Shot Object Detection by Knowledge Distillation Using Bag-of-Visual-Words Representations
- `ecva:eccv2022:6328` — GLAMD: Global and Local Attention Mask Distillation for Object Detectors
- `ecva:eccv2024:11200` — Distilling Knowledge from Large-Scale Image Models for Object Detection
- `ecva:eccv2024:11254` — Enhancing Source-Free Domain Adaptive Object Detection with Low-confidence Pseudo Label Distillation
- `ecva:eccv2024:6619` — Multi-scale Cross Distillation for Object Detection in Aerial Images
- `ecva:eccv2024:7083` — Simplifying Source-Free Domain Adaptation for Object Detection: Effective Self-Training Strategies and Performance Insights
- `neurips:2021:082a8bbf2c357c09f26675f9cf5bcba3-Abstract` — Distilling Image Classifiers in Object Detectors
- `neurips:2021:29c0c0ee223856f336d7ea8052057753-Abstract` — Distilling Object Detectors with Feature Richness
- `neurips:2021:892c91e0a653ba19df81a90f89d99bcd-Abstract` — Instance-Conditional Knowledge Distillation for Object Detection
- `neurips:2021:c0cccc24dd23ded67404f5e511c342b0-Abstract` — SSAL: Synergizing between Self-Training and Adversarial Learning for Domain Adaptive Object Detection
- `neurips:2022:18c0102cb7f1a02c14f0929089b2e576-Abstract-Conference` — Structural Knowledge Distillation for Object Detection
- `neurips:2022:631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference` — PKD: General Distillation Framework for Object Detectors via Pearson Correlation Coefficient
- `neurips:2024:6b6492cd06db22bac024506e9ed0925e-Abstract-Conference` — Towards Unsupervised Model Selection for Domain Adaptive Object Detection
- `neurips:2024:89d0d5c2f720921df93bbb8fef514571-Abstract-Conference` — Domain Adaptation for Large-Vocabulary Object Detectors
- `neurips:2024:bb71b5567ee985e0a4cee54ade19275c-Abstract-Conference` — DA-Ada: Learning Domain-Aware Adapter for Domain Adaptive Object Detection
- `neurips:2025:6460e378f24da3a79f20ac2640732a00-Abstract-Conference` — ELDET: Early-Learning Distillation with Noisy Labels for Object Detection
- `papernotes:black-box_domain_adaptation_for_object_detection_with_retention-driven_knowledge` — Black-Box Domain Adaptation for Object Detection with Retention-Driven Knowledge Compression
- `papernotes:expert-teacher-student_collaborative_learning_for_domain_adaptive_object_detecti` — Expert-Teacher-Student Collaborative Learning for Domain Adaptive Object Detection

