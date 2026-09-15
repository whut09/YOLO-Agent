# Paper-83 Data-side Status

This report audits only data-side mechanisms. It does not train a model and does not promote shared adapters to paper implementations.

- Frozen papers: 83
- Data-side papers in scope: 0
- Ready data-side routes: 0
- Blocked for missing evidence: 0
- Out of scope: 83
- Manifest membership hash: `1d77fc0df6c9e348b99399368f16453f337640d9d35263fd6fecb5b71e95bce0`
- Plan source: `E:\codex\YOLO-Agent\configs\research\paper_83_engineering_plan.yaml`

## Scope Decision

The current frozen engineering plan contains no paper whose primary or secondary domain is one of `data_quality`, `annotation`, `sampling`, `augmentation`, `preprocessing`, `long_tail`, or `active_learning`.

The checked-in data primitives are therefore not attributed to any of the frozen 83 papers. A later plan must supply an exact paper-specific mechanism, config, and evidence before a row can be marked ready.

## Per-paper Audit

| Paper | Domains | Scope | Status | Mechanisms | Runtime routes | Blockers |
|---|---|---|---|---|---|---|
| `arxiv:2103.14259` | assignment | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2104.14082` | bbox_loss | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2107.08430` | assignment | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2108.07755` | head | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2109.05986` | bbox_loss | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2203.16250` | assignment | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2208.00817` | assignment | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2210.11539` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2212.07784` | neck | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2301.01019` | bbox_loss | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2303.13853` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2303.14404` | bbox_loss | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2309.11331` | feature_fusion | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2503.23220` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2507.00721` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2603.12409` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2603.18541` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2603.18757` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `arxiv:2603.28182` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2021:Dai_General_Instance_Distillation_for_Object_Detection` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2021:Guo_Distilling_Object_Detectors_via_Decoupled_Features` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2021:Hu_Dense_Relation_Distillation_With_Context-Aware_Aggregation_for_Few-Shot_Object_Detection` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2021:VS_MeGA-CDA_Memory_Guided_Attention_for_Category-Aware_Unsupervised_Domain_Adaptive_Object` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2021:Zhang_RPN_Prototype_Alignment_for_Domain_Adaptive_Object_Detector` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2022:Feng_Overcoming_Catastrophic_Forgetting_in_Incremental_Object_Detection_via_Elastic_Response` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2022:Guo_Scale-Equivalent_Distillation_for_Semi-Supervised_Object_Detection` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2022:He_Cross_Domain_Object_Detection_by_Target-Perceived_Dual_Branch_Distillation` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2022:Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2022:Li_SIGMA_Semantic-Complete_Graph_Matching_for_Domain_Adaptive_Object_Detection` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2022:Wu_Single-Domain_Generalized_Object_Detection_in_Urban_Scene_via_Cyclic-Disentangled_Self-Distillation` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2022:Wu_Target-Relevant_Knowledge_Preservation_for_Multi-Source_Domain_Adaptive_Object_Detection` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2022:Zhao_Task-Specific_Inconsistency_Alignment_for_Domain_Adaptive_Object_Detection` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2022:Zheng_Localization_Distillation_for_Dense_Object_Detection` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2022:Zhou_Multi-Granularity_Alignment_Domain_Adaptation_for_Object_Detection` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2023:Cao_Contrastive_Mean_Teacher_for_Domain_Adaptive_Object_Detectors` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2023:Gao_AsyFOD_An_Asymmetric_Adaptation_Paradigm_for_Few-Shot_Domain_Adaptive_Object` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2023:Liu_CIGAR_Cross-Modality_Graph_Reasoning_for_Domain_Adaptive_Object_Detection` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2023:VS_Instance_Relation_Graph_Guided_Source-Free_Domain_Adaptive_Object_Detection` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2023:Wang_Object-Aware_Distillation_Pyramid_for_Open-Vocabulary_Object_Detection` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2023:Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2024:Du_Boosting_Object_Detection_with_Zero-Shot_Day-Night_Domain_Adaptation` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2024:Kennerley_CAT_Exploiting_Inter-Class_Dynamics_for_Domain_Adaptive_Object_Detection` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2024:Nakamura_Active_Domain_Adaptation_with_False_Negative_Prediction_for_Object_Detection` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2024:Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2024:Yang_Active_Object_Detection_with_Knowledge_Aggregation_and_Distillation_from_Large` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2025:Li_SEEN-DA_SEmantic_ENtropy_guided_Domain-aware_Attention_for_Domain_Adaptive_Object` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:cvpr2025:Liu_Distinguish_Then_Exploit_Source-free_Open_Set_Domain_Adaptation_via_Weight` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:iccv2021:Chen_Deep_Structured_Instance_Graph_for_Distilling_Object_Detectors` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:iccv2021:Chen_Dual_Bipartite_Graph_Learning_A_General_Approach_for_Domain_Adaptive` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:iccv2021:Tian_Knowledge_Mining_and_Transferring_for_Domain_Adaptive_Object_Detection` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:iccv2021:Yao_G-DetKD_Towards_General_Distillation_Framework_for_Object_Detectors_via_Contrastive` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:iccv2021:Yao_Multi-Source_Domain_Adaptation_for_Object_Detection` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:iccv2023:Gao_CSDA_Learning_Category-Scale_Joint_Feature_for_Domain_Adaptive_Object_Detection` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:iccv2023:Kang_Alleviating_Catastrophic_Forgetting_of_Incremental_Object_Detection_via_Within-Class_and` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:iccv2023:Lao_UniKD_Universal_Knowledge_Distillation_for_Mimicking_Homogeneous_or_Heterogeneous_Object` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:iccv2023:Wu_Spatial_Self-Distillation_for_Object_Detection_with_Inaccurate_Bounding_Boxes` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:iccv2023:Yang_Bridging_Cross-task_Protocol_Inconsistency_for_Distillation_in_Dense_Object_Detection` | distillation | out-of-scope | out_of_scope | none | none | none |
| `cvf:iccv2023:Zhao_Masked_Retraining_Teacher-Student_Framework_for_Domain_Adaptive_Object_Detection` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:iccv2025:Cui_Debiased_Teacher_for_Day-to-Night_Domain_Adaptive_Object_Detection` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `cvf:iccv2025:He_Dual-Rate_Dynamic_Teacher_for_Source-Free_Domain_Adaptive_Object_Detection` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `ecva:eccv2022:1356` | distillation | out-of-scope | out_of_scope | none | none | none |
| `ecva:eccv2022:2285` | distillation | out-of-scope | out_of_scope | none | none | none |
| `ecva:eccv2022:2717` | distillation | out-of-scope | out_of_scope | none | none | none |
| `ecva:eccv2022:3523` | distillation | out-of-scope | out_of_scope | none | none | none |
| `ecva:eccv2022:3958` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `ecva:eccv2022:6004` | distillation | out-of-scope | out_of_scope | none | none | none |
| `ecva:eccv2022:6328` | distillation | out-of-scope | out_of_scope | none | none | none |
| `ecva:eccv2024:11200` | distillation | out-of-scope | out_of_scope | none | none | none |
| `ecva:eccv2024:11254` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `ecva:eccv2024:6619` | distillation | out-of-scope | out_of_scope | none | none | none |
| `ecva:eccv2024:7083` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `neurips:2021:082a8bbf2c357c09f26675f9cf5bcba3-Abstract` | distillation | out-of-scope | out_of_scope | none | none | none |
| `neurips:2021:29c0c0ee223856f336d7ea8052057753-Abstract` | distillation | out-of-scope | out_of_scope | none | none | none |
| `neurips:2021:892c91e0a653ba19df81a90f89d99bcd-Abstract` | distillation | out-of-scope | out_of_scope | none | none | none |
| `neurips:2021:c0cccc24dd23ded67404f5e511c342b0-Abstract` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `neurips:2022:18c0102cb7f1a02c14f0929089b2e576-Abstract-Conference` | distillation | out-of-scope | out_of_scope | none | none | none |
| `neurips:2022:631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference` | distillation | out-of-scope | out_of_scope | none | none | none |
| `neurips:2024:6b6492cd06db22bac024506e9ed0925e-Abstract-Conference` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `neurips:2024:89d0d5c2f720921df93bbb8fef514571-Abstract-Conference` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `neurips:2024:bb71b5567ee985e0a4cee54ade19275c-Abstract-Conference` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `neurips:2025:6460e378f24da3a79f20ac2640732a00-Abstract-Conference` | distillation | out-of-scope | out_of_scope | none | none | none |
| `papernotes:black-box_domain_adaptation_for_object_detection_with_retention-driven_knowledge` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |
| `papernotes:expert-teacher-student_collaborative_learning_for_domain_adaptive_object_detecti` | domain_adaptation | out-of-scope | out_of_scope | none | none | none |

`ready` means the exact route's CPU behavior probe passed and the paper supplied planning/evidence/config references. It is not a mAP result or a claim of exact reproduction.
