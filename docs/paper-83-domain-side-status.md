# Paper-83 Distillation/Domain-Adaptation Status

This report audits teacher, domain-adaptation, and semi-supervised
mechanisms. It does not train a model. Generic teacher-student and
domain-alignment branches are shared primitives; each paper's status
is bound to its own paper-specific route and behavior evidence.

- Frozen papers: 83
- Teacher/domain papers in scope: 72
- Distillation papers: 32
- Domain-adaptation papers: 40
- Semi-supervised papers: 0
- Ready domain-side routes: 58
- Blocked for missing evidence: 14
- Out of scope: 11
- Manifest membership hash: `1d77fc0df6c9e348b99399368f16453f337640d9d35263fd6fecb5b71e95bce0`
- Plan source: `E:\codex\YOLO-Agent\configs\research\paper_83_engineering_plan.yaml`

## Per-paper Audit

| Paper | Domain | Scope | Status | Routes | Branch | Assets blocked | Blockers |
|---|---|---|---|---|---|---|---|
| `arxiv:2103.14259` | assignment | out-of-scope | out_of_scope | none | none | - | none |
| `arxiv:2104.14082` | bbox_loss | out-of-scope | out_of_scope | none | none | - | none |
| `arxiv:2107.08430` | assignment | out-of-scope | out_of_scope | none | none | - | none |
| `arxiv:2108.07755` | head | out-of-scope | out_of_scope | none | none | - | none |
| `arxiv:2109.05986` | bbox_loss | out-of-scope | out_of_scope | none | none | - | none |
| `arxiv:2203.16250` | assignment | out-of-scope | out_of_scope | none | none | - | none |
| `arxiv:2208.00817` | assignment | out-of-scope | out_of_scope | none | none | - | none |
| `arxiv:2210.11539` | domain_adaptation | domain-side | ready | domain_adaptation.2210_11539 | adversarial_alignment | 14 | none |
| `arxiv:2212.07784` | neck | out-of-scope | out_of_scope | none | none | - | none |
| `arxiv:2301.01019` | bbox_loss | out-of-scope | out_of_scope | none | none | - | none |
| `arxiv:2303.13853` | domain_adaptation | domain-side | ready | domain_adaptation.2303_13853 | feature_alignment | 14 | none |
| `arxiv:2303.14404` | bbox_loss | out-of-scope | out_of_scope | none | none | - | none |
| `arxiv:2309.11331` | feature_fusion | out-of-scope | out_of_scope | none | none | - | none |
| `arxiv:2503.23220` | domain_adaptation | domain-side | ready | domain_adaptation.2503_23220 | pseudo_label_adaptation | 14 | none |
| `arxiv:2507.00721` | domain_adaptation | domain-side | ready | domain_adaptation.2507_00721 | feature_alignment | 14 | none |
| `arxiv:2603.12409` | domain_adaptation | domain-side | ready | domain_adaptation.2603_12409 | feature_alignment | 14 | none |
| `arxiv:2603.18541` | domain_adaptation | domain-side | ready | domain_adaptation.2603_18541 | contrastive_domain_alignment | 14 | none |
| `arxiv:2603.18757` | domain_adaptation | domain-side | ready | domain_adaptation.2603_18757 | feature_alignment | 14 | none |
| `arxiv:2603.28182` | domain_adaptation | domain-side | ready | domain_adaptation.2603_28182 | active_domain_adaptation | 14 | none |
| `cvf:cvpr2021:Dai_General_Instance_Distillation_for_Object_Detection` | distillation | domain-side | ready | distillation.general_instance | relation_distillation | 12 | none |
| `cvf:cvpr2021:Guo_Distilling_Object_Detectors_via_Decoupled_Features` | distillation | domain-side | ready | distillation.decoupled_features | feature_distillation | 12 | none |
| `cvf:cvpr2021:Hu_Dense_Relation_Distillation_With_Context-Aware_Aggregation_for_Few-Shot_Object_Detection` | distillation | domain-side | ready | distillation.dense_relation_fewshot | relation_distillation | 13 | none |
| `cvf:cvpr2021:VS_MeGA-CDA_Memory_Guided_Attention_for_Category-Aware_Unsupervised_Domain_Adaptive_Object` | domain_adaptation | domain-side | ready | domain_adaptation.mega_cda | feature_alignment | 15 | none |
| `cvf:cvpr2021:Zhang_RPN_Prototype_Alignment_for_Domain_Adaptive_Object_Detector` | domain_adaptation | domain-side | ready | domain_adaptation.rpn_prototype_alignment | adversarial_alignment | 15 | none |
| `cvf:cvpr2022:Feng_Overcoming_Catastrophic_Forgetting_in_Incremental_Object_Detection_via_Elastic_Response` | distillation | domain-side | ready | distillation.elastic_response_incremental | quality_aware_distillation | 13 | none |
| `cvf:cvpr2022:Guo_Scale-Equivalent_Distillation_for_Semi-Supervised_Object_Detection` | distillation | domain-side | ready | distillation.scale_equivalent | feature_distillation | 13 | none |
| `cvf:cvpr2022:He_Cross_Domain_Object_Detection_by_Target-Perceived_Dual_Branch_Distillation` | distillation | domain-side | ready | distillation.target_perceived_dual_branch | cross_domain_teacher | 12 | none |
| `cvf:cvpr2022:Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection` | domain_adaptation | domain-side | ready | domain_adaptation.adaptive_teacher | cross_domain_teacher | 16 | none |
| `cvf:cvpr2022:Li_SIGMA_Semantic-Complete_Graph_Matching_for_Domain_Adaptive_Object_Detection` | domain_adaptation | domain-side | ready | domain_adaptation.sigma_graph_matching | feature_alignment | 15 | none |
| `cvf:cvpr2022:Wu_Single-Domain_Generalized_Object_Detection_in_Urban_Scene_via_Cyclic-Disentangled_Self-Distillation` | distillation | domain-side | ready | distillation.cyclic_disentangled | feature_distillation | 12 | none |
| `cvf:cvpr2022:Wu_Target-Relevant_Knowledge_Preservation_for_Multi-Source_Domain_Adaptive_Object_Detection` | domain_adaptation | domain-side | ready | domain_adaptation.multi_source_knowledge | feature_alignment | 15 | none |
| `cvf:cvpr2022:Zhao_Task-Specific_Inconsistency_Alignment_for_Domain_Adaptive_Object_Detection` | domain_adaptation | domain-side | ready | domain_adaptation.inconsistency_alignment | adversarial_alignment | 15 | none |
| `cvf:cvpr2022:Zheng_Localization_Distillation_for_Dense_Object_Detection` | distillation | domain-side | ready | distillation.localization | localization_distillation | 11 | none |
| `cvf:cvpr2022:Zhou_Multi-Granularity_Alignment_Domain_Adaptation_for_Object_Detection` | domain_adaptation | domain-side | ready | domain_adaptation.multi_granularity | feature_alignment | 15 | none |
| `cvf:cvpr2023:Cao_Contrastive_Mean_Teacher_for_Domain_Adaptive_Object_Detectors` | domain_adaptation | domain-side | ready | domain_adaptation.contrastive_mean_teacher | contrastive_domain_alignment | 15 | none |
| `cvf:cvpr2023:Gao_AsyFOD_An_Asymmetric_Adaptation_Paradigm_for_Few-Shot_Domain_Adaptive_Object` | domain_adaptation | domain-side | ready | domain_adaptation.asyfod | pseudo_label_adaptation | 15 | none |
| `cvf:cvpr2023:Liu_CIGAR_Cross-Modality_Graph_Reasoning_for_Domain_Adaptive_Object_Detection` | domain_adaptation | domain-side | ready | domain_adaptation.cigar_graph | feature_alignment | 15 | none |
| `cvf:cvpr2023:VS_Instance_Relation_Graph_Guided_Source-Free_Domain_Adaptive_Object_Detection` | domain_adaptation | domain-side | ready | domain_adaptation.source_free_irg | source_free_adaptation | 15 | none |
| `cvf:cvpr2023:Wang_Object-Aware_Distillation_Pyramid_for_Open-Vocabulary_Object_Detection` | distillation | domain-side | ready | distillation.object_aware_pyramid | feature_distillation | 13 | none |
| `cvf:cvpr2023:Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector` | distillation | domain-side | ready | distillation.scalekd | feature_distillation | 13 | none |
| `cvf:cvpr2024:Du_Boosting_Object_Detection_with_Zero-Shot_Day-Night_Domain_Adaptation` | domain_adaptation | domain-side | ready | domain_adaptation.zero_shot_day_night | feature_alignment | 15 | none |
| `cvf:cvpr2024:Kennerley_CAT_Exploiting_Inter-Class_Dynamics_for_Domain_Adaptive_Object_Detection` | domain_adaptation | domain-side | ready | domain_adaptation.cat_interclass | feature_alignment | 15 | none |
| `cvf:cvpr2024:Nakamura_Active_Domain_Adaptation_with_False_Negative_Prediction_for_Object_Detection` | domain_adaptation | domain-side | ready | domain_adaptation.active_false_negative | active_domain_adaptation | 15 | none |
| `cvf:cvpr2024:Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection` | distillation | domain-side | ready | distillation.crosskd | logits_distillation | 13 | none |
| `cvf:cvpr2024:Yang_Active_Object_Detection_with_Knowledge_Aggregation_and_Distillation_from_Large` | distillation | domain-side | ready | distillation.active_knowledge_aggregation | teacher_ensemble | 13 | none |
| `cvf:cvpr2025:Li_SEEN-DA_SEmantic_ENtropy_guided_Domain-aware_Attention_for_Domain_Adaptive_Object` | domain_adaptation | domain-side | ready | domain_adaptation.seen_da | feature_alignment | 15 | none |
| `cvf:cvpr2025:Liu_Distinguish_Then_Exploit_Source-free_Open_Set_Domain_Adaptation_via_Weight` | domain_adaptation | domain-side | ready | domain_adaptation.source_free_open_set | source_free_adaptation | 15 | none |
| `cvf:iccv2021:Chen_Deep_Structured_Instance_Graph_for_Distilling_Object_Detectors` | distillation | domain-side | ready | distillation.structured_instance_graph | relation_distillation | 12 | none |
| `cvf:iccv2021:Chen_Dual_Bipartite_Graph_Learning_A_General_Approach_for_Domain_Adaptive` | domain_adaptation | domain-side | ready | domain_adaptation.dual_bipartite_graph | feature_alignment | 15 | none |
| `cvf:iccv2021:Tian_Knowledge_Mining_and_Transferring_for_Domain_Adaptive_Object_Detection` | domain_adaptation | domain-side | ready | domain_adaptation.knowledge_mining | cross_domain_teacher | 16 | none |
| `cvf:iccv2021:Yao_G-DetKD_Towards_General_Distillation_Framework_for_Object_Detectors_via_Contrastive` | distillation | domain-side | ready | distillation.gdetkd | contrastive_distillation | 12 | none |
| `cvf:iccv2021:Yao_Multi-Source_Domain_Adaptation_for_Object_Detection` | domain_adaptation | domain-side | ready | domain_adaptation.multi_source | feature_alignment | 15 | none |
| `cvf:iccv2023:Gao_CSDA_Learning_Category-Scale_Joint_Feature_for_Domain_Adaptive_Object_Detection` | domain_adaptation | domain-side | ready | domain_adaptation.csda | feature_alignment | 15 | none |
| `cvf:iccv2023:Kang_Alleviating_Catastrophic_Forgetting_of_Incremental_Object_Detection_via_Within-Class_and` | distillation | domain-side | ready | distillation.incremental_within_class | logits_distillation | 13 | none |
| `cvf:iccv2023:Lao_UniKD_Universal_Knowledge_Distillation_for_Mimicking_Homogeneous_or_Heterogeneous_Object` | distillation | domain-side | ready | distillation.unikd | logits_distillation | 13 | none |
| `cvf:iccv2023:Wu_Spatial_Self-Distillation_for_Object_Detection_with_Inaccurate_Bounding_Boxes` | distillation | domain-side | ready | distillation.spatial_self | attention_distillation | 12 | none |
| `cvf:iccv2023:Yang_Bridging_Cross-task_Protocol_Inconsistency_for_Distillation_in_Dense_Object_Detection` | distillation | domain-side | ready | distillation.cross_task_protocol | logits_distillation | 12 | none |
| `cvf:iccv2023:Zhao_Masked_Retraining_Teacher-Student_Framework_for_Domain_Adaptive_Object_Detection` | domain_adaptation | domain-side | ready | domain_adaptation.masked_retraining_teacher | cross_domain_teacher | 16 | none |
| `cvf:iccv2025:Cui_Debiased_Teacher_for_Day-to-Night_Domain_Adaptive_Object_Detection` | domain_adaptation | domain-side | ready | domain_adaptation.debiased_teacher | cross_domain_teacher | 16 | none |
| `cvf:iccv2025:He_Dual-Rate_Dynamic_Teacher_for_Source-Free_Domain_Adaptive_Object_Detection` | domain_adaptation | domain-side | ready | domain_adaptation.dual_rate_source_free | source_free_adaptation | 15 | none |
| `ecva:eccv2022:1356` | distillation | domain-side | blocked_missing_evidence | distillation.1356 | none | 11 | blocked_missing_evidence:paper_specific_domain_mechanism<br>blocked_missing_evidence:paper_specific_domain_config<br>blocked_missing_evidence:generic_branch_not_paper_route:distillation.1356<br>blocked_missing_evidence:behavior:distillation.1356:no_behavior_probe_for_identity_recovery |
| `ecva:eccv2022:2285` | distillation | domain-side | blocked_missing_evidence | distillation.2285 | none | 1 | blocked_missing_evidence:paper_specific_domain_mechanism<br>blocked_missing_evidence:paper_specific_domain_config<br>blocked_missing_evidence:generic_branch_not_paper_route:distillation.2285<br>blocked_missing_evidence:behavior:distillation.2285:no_behavior_probe_for_identity_recovery |
| `ecva:eccv2022:2717` | distillation | domain-side | blocked_missing_evidence | distillation.2717 | none | 12 | blocked_missing_evidence:paper_specific_domain_mechanism<br>blocked_missing_evidence:paper_specific_domain_config<br>blocked_missing_evidence:generic_branch_not_paper_route:distillation.2717<br>blocked_missing_evidence:behavior:distillation.2717:no_behavior_probe_for_identity_recovery |
| `ecva:eccv2022:3523` | distillation | domain-side | blocked_missing_evidence | distillation.3523 | none | 12 | blocked_missing_evidence:paper_specific_domain_mechanism<br>blocked_missing_evidence:paper_specific_domain_config<br>blocked_missing_evidence:generic_branch_not_paper_route:distillation.3523<br>blocked_missing_evidence:behavior:distillation.3523:no_behavior_probe_for_identity_recovery |
| `ecva:eccv2022:3958` | domain_adaptation | domain-side | ready | domain_adaptation.3958 | adversarial_alignment | 15 | none |
| `ecva:eccv2022:6004` | distillation | domain-side | blocked_missing_evidence | distillation.6004 | none | 12 | blocked_missing_evidence:paper_specific_domain_mechanism<br>blocked_missing_evidence:paper_specific_domain_config<br>blocked_missing_evidence:generic_branch_not_paper_route:distillation.6004<br>blocked_missing_evidence:behavior:distillation.6004:no_behavior_probe_for_identity_recovery |
| `ecva:eccv2022:6328` | distillation | domain-side | blocked_missing_evidence | distillation.6328 | none | 11 | blocked_missing_evidence:paper_specific_domain_mechanism<br>blocked_missing_evidence:paper_specific_domain_config<br>blocked_missing_evidence:generic_branch_not_paper_route:distillation.6328<br>blocked_missing_evidence:behavior:distillation.6328:no_behavior_probe_for_identity_recovery |
| `ecva:eccv2024:11200` | distillation | domain-side | blocked_missing_evidence | distillation.11200 | none | 12 | blocked_missing_evidence:paper_specific_domain_mechanism<br>blocked_missing_evidence:paper_specific_domain_config<br>blocked_missing_evidence:generic_branch_not_paper_route:distillation.11200<br>blocked_missing_evidence:behavior:distillation.11200:no_behavior_probe_for_identity_recovery |
| `ecva:eccv2024:11254` | domain_adaptation | domain-side | ready | domain_adaptation.11254 | cross_domain_teacher | 16 | none |
| `ecva:eccv2024:6619` | distillation | domain-side | blocked_missing_evidence | distillation.6619 | none | 12 | blocked_missing_evidence:paper_specific_domain_mechanism<br>blocked_missing_evidence:paper_specific_domain_config<br>blocked_missing_evidence:generic_branch_not_paper_route:distillation.6619<br>blocked_missing_evidence:behavior:distillation.6619:no_behavior_probe_for_identity_recovery |
| `ecva:eccv2024:7083` | domain_adaptation | domain-side | ready | domain_adaptation.7083 | feature_alignment | 15 | none |
| `neurips:2021:082a8bbf2c357c09f26675f9cf5bcba3-Abstract` | distillation | domain-side | blocked_missing_evidence | distillation.082a8bbf2c357c09f26675f9cf5bcba3_abstract | none | 11 | blocked_missing_evidence:paper_specific_domain_mechanism<br>blocked_missing_evidence:paper_specific_domain_config<br>blocked_missing_evidence:generic_branch_not_paper_route:distillation.082a8bbf2c357c09f26675f9cf5bcba3_abstract<br>blocked_missing_evidence:behavior:distillation.082a8bbf2c357c09f26675f9cf5bcba3_abstract:no_behavior_probe_for_identity_recovery |
| `neurips:2021:29c0c0ee223856f336d7ea8052057753-Abstract` | distillation | domain-side | blocked_missing_evidence | distillation.29c0c0ee223856f336d7ea8052057753_abstract | none | 11 | blocked_missing_evidence:paper_specific_domain_mechanism<br>blocked_missing_evidence:paper_specific_domain_config<br>blocked_missing_evidence:generic_branch_not_paper_route:distillation.29c0c0ee223856f336d7ea8052057753_abstract<br>blocked_missing_evidence:behavior:distillation.29c0c0ee223856f336d7ea8052057753_abstract:no_behavior_probe_for_identity_recovery |
| `neurips:2021:892c91e0a653ba19df81a90f89d99bcd-Abstract` | distillation | domain-side | blocked_missing_evidence | distillation.892c91e0a653ba19df81a90f89d99bcd_abstract | none | 12 | blocked_missing_evidence:paper_specific_domain_mechanism<br>blocked_missing_evidence:paper_specific_domain_config<br>blocked_missing_evidence:generic_branch_not_paper_route:distillation.892c91e0a653ba19df81a90f89d99bcd_abstract<br>blocked_missing_evidence:behavior:distillation.892c91e0a653ba19df81a90f89d99bcd_abstract:no_behavior_probe_for_identity_recovery |
| `neurips:2021:c0cccc24dd23ded67404f5e511c342b0-Abstract` | domain_adaptation | domain-side | ready | domain_adaptation.c0cccc24dd23ded67404f5e511c342b0_abstract | contrastive_domain_alignment | 15 | none |
| `neurips:2022:18c0102cb7f1a02c14f0929089b2e576-Abstract-Conference` | distillation | domain-side | blocked_missing_evidence | distillation.18c0102cb7f1a02c14f0929089b2e576_abstract_confer | none | 12 | blocked_missing_evidence:paper_specific_domain_mechanism<br>blocked_missing_evidence:paper_specific_domain_config<br>blocked_missing_evidence:generic_branch_not_paper_route:distillation.18c0102cb7f1a02c14f0929089b2e576_abstract_confer<br>blocked_missing_evidence:behavior:distillation.18c0102cb7f1a02c14f0929089b2e576_abstract_confer:no_behavior_probe_for_identity_recovery |
| `neurips:2022:631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference` | distillation | domain-side | blocked_missing_evidence | distillation.631ad9ae3174bf4d6c0f6fdca77335a4_abstract_confer | none | 11 | blocked_missing_evidence:paper_specific_domain_mechanism<br>blocked_missing_evidence:paper_specific_domain_config<br>blocked_missing_evidence:generic_branch_not_paper_route:distillation.631ad9ae3174bf4d6c0f6fdca77335a4_abstract_confer<br>blocked_missing_evidence:behavior:distillation.631ad9ae3174bf4d6c0f6fdca77335a4_abstract_confer:no_behavior_probe_for_identity_recovery |
| `neurips:2024:6b6492cd06db22bac024506e9ed0925e-Abstract-Conference` | domain_adaptation | domain-side | ready | domain_adaptation.6b6492cd06db22bac024506e9ed0925e_abstract_confer | pseudo_label_adaptation | 15 | none |
| `neurips:2024:89d0d5c2f720921df93bbb8fef514571-Abstract-Conference` | domain_adaptation | domain-side | ready | domain_adaptation.89d0d5c2f720921df93bbb8fef514571_abstract_confer | active_domain_adaptation | 15 | none |
| `neurips:2024:bb71b5567ee985e0a4cee54ade19275c-Abstract-Conference` | domain_adaptation | domain-side | ready | domain_adaptation.bb71b5567ee985e0a4cee54ade19275c_abstract_confer | feature_alignment | 15 | none |
| `neurips:2025:6460e378f24da3a79f20ac2640732a00-Abstract-Conference` | distillation | domain-side | blocked_missing_evidence | distillation.6460e378f24da3a79f20ac2640732a00_abstract_confer | none | 11 | blocked_missing_evidence:paper_specific_domain_mechanism<br>blocked_missing_evidence:paper_specific_domain_config<br>blocked_missing_evidence:generic_branch_not_paper_route:distillation.6460e378f24da3a79f20ac2640732a00_abstract_confer<br>blocked_missing_evidence:behavior:distillation.6460e378f24da3a79f20ac2640732a00_abstract_confer:no_behavior_probe_for_identity_recovery |
| `papernotes:black-box_domain_adaptation_for_object_detection_with_retention-driven_knowledge` | domain_adaptation | domain-side | ready | domain_adaptation.black_box_retention | source_free_adaptation | 14 | none |
| `papernotes:expert-teacher-student_collaborative_learning_for_domain_adaptive_object_detecti` | domain_adaptation | domain-side | ready | domain_adaptation.expert_teacher_student | cross_domain_teacher | 15 | none |

## Method Notes

- Required teacher/checkpoint/data assets are recorded per paper;
  `blocked_for_real_reproduction=true` marks assets the first training
  run must provide. It never blocks code-level implementation readiness
  because every mechanism runs on synthetic teachers/students in the
  probe suite. No in-scope paper depends on an unobtainable fixed
  external model.
- Behavior evidence comes from real CPU probes: mechanism forward and
  backward without teacher gradients, EMA decay arithmetic, integer
  buffer copy, pseudo-label confidence filtering with fail-closed
  empty-set behavior, and gradient-reversal direction (forward
  passthrough, negated scaled backward).
- Readiness is per paper and per route: an 11-mechanism shared
  distillation library or 8-strategy domain library passing probes
  never marks a paper ready without its own route, config, and
  evidence. Identity-recovery routes (no certified branch) are
  blocked, not silently promoted.
- No training is started by this audit.
