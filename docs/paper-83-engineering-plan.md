# Paper-83 Engineering Plan

This document is an implementation work plan, not a reproduction or training result.

- Frozen papers: 83
- Membership hash: `1d77fc0df6c9e348b99399368f16453f337640d9d35263fd6fecb5b71e95bce0`
- Implementation-ready at source audit: 9
- Batches: 11

Shared primitives are dependencies only; each row remains a separate paper implementation contract.

## Batches

| Batch | Focus | Papers | Shared primitives |
|---|---|---:|---|
| `paper83-assignment-01` | assignment | 7 | none |
| `paper83-bbox-loss-01` | bbox_loss | 4 | none |
| `paper83-distillation-01` | distillation | 8 | none |
| `paper83-distillation-02` | distillation | 8 | none |
| `paper83-distillation-03` | distillation | 8 | distillation.yolo26_teacher_student |
| `paper83-distillation-04` | distillation | 8 | distillation.yolo26_teacher_student |
| `paper83-domain-adaptation-01` | domain_adaptation | 8 | none |
| `paper83-domain-adaptation-02` | domain_adaptation | 8 | none |
| `paper83-domain-adaptation-03` | domain_adaptation | 8 | none |
| `paper83-domain-adaptation-04` | domain_adaptation | 8 | none |
| `paper83-domain-adaptation-05` | domain_adaptation | 8 | none |

## Per-Paper Plan

| Paper | Domain | Readiness at audit | Mechanisms | Missing parts | Disposition |
|---|---|---|---|---|---|
| `arxiv:2103.14259` | assignment | implementation_ready | assigner.optimal_transport<br>optimal_transport_assigner | none | blocked_runtime |
| `arxiv:2104.14082` | bbox_loss | implementation_ready | loss.quality.pseudo_iou<br>pseudo_iou | none | blocked_runtime |
| `arxiv:2107.08430` | assignment | implementation_ready | assigner.optimal_transport<br>ota_assignment | none | blocked_runtime |
| `arxiv:2108.07755` | head | code_bound | detection_head.task_aligned | non_mock_smoke_validation<br>unit_validation | blocked_runtime |
| `arxiv:2109.05986` | bbox_loss | code_bound | loss.2109_05986 | non_mock_smoke_validation<br>unit_validation | implementation_request |
| `arxiv:2203.16250` | assignment | implementation_ready | assigner.task_aligned<br>task_aligned_assigner | none | blocked_runtime |
| `arxiv:2208.00817` | assignment | implementation_ready | assigner.dynamic_smooth_label<br>dynamic_smooth_labels | none | blocked_runtime |
| `arxiv:2210.11539` | domain_adaptation | code_bound | domain_adaptation.2210_11539 | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `arxiv:2212.07784` | neck | implementation_ready | large_kernel_depthwise_conv | none | runtime_ready |
| `arxiv:2301.01019` | bbox_loss | implementation_ready | correlation_loss<br>loss.quality.correlation | none | blocked_runtime |
| `arxiv:2303.13853` | domain_adaptation | code_bound | domain_adaptation.2303_13853 | non_mock_smoke_validation<br>unit_validation | implementation_request |
| `arxiv:2303.14404` | bbox_loss | implementation_ready | confidence_calibration<br>loss.calibration.bpc<br>train_time_loss | none | blocked_runtime |
| `arxiv:2309.11331` | feature_fusion | code_bound | feature_pyramid.multi_scale | non_mock_smoke_validation<br>unit_validation | blocked_runtime |
| `arxiv:2503.23220` | domain_adaptation | code_bound | domain_adaptation.2503_23220 | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `arxiv:2507.00721` | domain_adaptation | code_bound | domain_adaptation.2507_00721 | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `arxiv:2603.12409` | domain_adaptation | code_bound | domain_adaptation.2603_12409 | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `arxiv:2603.18541` | domain_adaptation | code_bound | domain_adaptation.2603_18541 | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `arxiv:2603.18757` | domain_adaptation | code_bound | domain_adaptation.2603_18757 | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `arxiv:2603.28182` | domain_adaptation | code_bound | domain_adaptation.2603_28182 | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2021:Dai_General_Instance_Distillation_for_Object_Detection` | distillation | code_bound | distillation.general_instance | non_mock_smoke_validation<br>unit_validation | blocked_runtime |
| `cvf:cvpr2021:Guo_Distilling_Object_Detectors_via_Decoupled_Features` | distillation | code_bound | distillation.decoupled_features | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2021:Hu_Dense_Relation_Distillation_With_Context-Aware_Aggregation_for_Few-Shot_Object_Detection` | distillation | code_bound | distillation.dense_relation_fewshot | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | blocked_runtime |
| `cvf:cvpr2021:VS_MeGA-CDA_Memory_Guided_Attention_for_Category-Aware_Unsupervised_Domain_Adaptive_Object` | domain_adaptation | code_bound | domain_adaptation.mega_cda | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2021:Zhang_RPN_Prototype_Alignment_for_Domain_Adaptive_Object_Detector` | domain_adaptation | code_bound | domain_adaptation.rpn_prototype_alignment | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2022:Feng_Overcoming_Catastrophic_Forgetting_in_Incremental_Object_Detection_via_Elastic_Response` | distillation | code_bound | distillation.elastic_response_incremental | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | blocked_runtime |
| `cvf:cvpr2022:Guo_Scale-Equivalent_Distillation_for_Semi-Supervised_Object_Detection` | distillation | code_bound | distillation.scale_equivalent | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2022:He_Cross_Domain_Object_Detection_by_Target-Perceived_Dual_Branch_Distillation` | distillation | code_bound | distillation.target_perceived_dual_branch | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2022:Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection` | domain_adaptation | code_bound | domain_adaptation.adaptive_teacher | non_mock_smoke_validation<br>unit_validation | implementation_request |
| `cvf:cvpr2022:Li_SIGMA_Semantic-Complete_Graph_Matching_for_Domain_Adaptive_Object_Detection` | domain_adaptation | code_bound | domain_adaptation.sigma_graph_matching | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2022:Wu_Single-Domain_Generalized_Object_Detection_in_Urban_Scene_via_Cyclic-Disentangled_Self-Distillation` | distillation | code_bound | distillation.cyclic_disentangled | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2022:Wu_Target-Relevant_Knowledge_Preservation_for_Multi-Source_Domain_Adaptive_Object_Detection` | domain_adaptation | code_bound | domain_adaptation.multi_source_knowledge | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2022:Zhao_Task-Specific_Inconsistency_Alignment_for_Domain_Adaptive_Object_Detection` | domain_adaptation | code_bound | domain_adaptation.inconsistency_alignment | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2022:Zheng_Localization_Distillation_for_Dense_Object_Detection` | distillation | implementation_ready | distillation.localization | none | blocked_runtime |
| `cvf:cvpr2022:Zhou_Multi-Granularity_Alignment_Domain_Adaptation_for_Object_Detection` | domain_adaptation | code_bound | domain_adaptation.multi_granularity | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2023:Cao_Contrastive_Mean_Teacher_for_Domain_Adaptive_Object_Detectors` | domain_adaptation | code_bound | domain_adaptation.contrastive_mean_teacher | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2023:Gao_AsyFOD_An_Asymmetric_Adaptation_Paradigm_for_Few-Shot_Domain_Adaptive_Object` | domain_adaptation | code_bound | domain_adaptation.asyfod | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2023:Liu_CIGAR_Cross-Modality_Graph_Reasoning_for_Domain_Adaptive_Object_Detection` | domain_adaptation | code_bound | domain_adaptation.cigar_graph | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2023:VS_Instance_Relation_Graph_Guided_Source-Free_Domain_Adaptive_Object_Detection` | domain_adaptation | code_bound | domain_adaptation.source_free_irg | non_mock_smoke_validation<br>unit_validation | implementation_request |
| `cvf:cvpr2023:Wang_Object-Aware_Distillation_Pyramid_for_Open-Vocabulary_Object_Detection` | distillation | code_bound | distillation.object_aware_pyramid | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2023:Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector` | distillation | code_bound | distillation.scalekd | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2024:Du_Boosting_Object_Detection_with_Zero-Shot_Day-Night_Domain_Adaptation` | domain_adaptation | code_bound | domain_adaptation.zero_shot_day_night | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2024:Kennerley_CAT_Exploiting_Inter-Class_Dynamics_for_Domain_Adaptive_Object_Detection` | domain_adaptation | code_bound | domain_adaptation.cat_interclass | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2024:Nakamura_Active_Domain_Adaptation_with_False_Negative_Prediction_for_Object_Detection` | domain_adaptation | code_bound | domain_adaptation.active_false_negative | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2024:Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection` | distillation | code_bound | distillation.crosskd | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2024:Yang_Active_Object_Detection_with_Knowledge_Aggregation_and_Distillation_from_Large` | distillation | code_bound | distillation.active_knowledge_aggregation | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2025:Li_SEEN-DA_SEmantic_ENtropy_guided_Domain-aware_Attention_for_Domain_Adaptive_Object` | domain_adaptation | code_bound | domain_adaptation.seen_da | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:cvpr2025:Liu_Distinguish_Then_Exploit_Source-free_Open_Set_Domain_Adaptation_via_Weight` | domain_adaptation | code_bound | domain_adaptation.source_free_open_set | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:iccv2021:Chen_Deep_Structured_Instance_Graph_for_Distilling_Object_Detectors` | distillation | code_bound | distillation.structured_instance_graph | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:iccv2021:Chen_Dual_Bipartite_Graph_Learning_A_General_Approach_for_Domain_Adaptive` | domain_adaptation | code_bound | domain_adaptation.dual_bipartite_graph | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:iccv2021:Tian_Knowledge_Mining_and_Transferring_for_Domain_Adaptive_Object_Detection` | domain_adaptation | code_bound | domain_adaptation.knowledge_mining | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:iccv2021:Yao_G-DetKD_Towards_General_Distillation_Framework_for_Object_Detectors_via_Contrastive` | distillation | code_bound | distillation.gdetkd | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:iccv2021:Yao_Multi-Source_Domain_Adaptation_for_Object_Detection` | domain_adaptation | code_bound | domain_adaptation.multi_source | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:iccv2023:Gao_CSDA_Learning_Category-Scale_Joint_Feature_for_Domain_Adaptive_Object_Detection` | domain_adaptation | code_bound | domain_adaptation.csda | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:iccv2023:Kang_Alleviating_Catastrophic_Forgetting_of_Incremental_Object_Detection_via_Within-Class_and` | distillation | code_bound | distillation.incremental_within_class | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:iccv2023:Lao_UniKD_Universal_Knowledge_Distillation_for_Mimicking_Homogeneous_or_Heterogeneous_Object` | distillation | code_bound | distillation.unikd | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:iccv2023:Wu_Spatial_Self-Distillation_for_Object_Detection_with_Inaccurate_Bounding_Boxes` | distillation | code_bound | distillation.spatial_self | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:iccv2023:Yang_Bridging_Cross-task_Protocol_Inconsistency_for_Distillation_in_Dense_Object_Detection` | distillation | code_bound | distillation.cross_task_protocol | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:iccv2023:Zhao_Masked_Retraining_Teacher-Student_Framework_for_Domain_Adaptive_Object_Detection` | domain_adaptation | code_bound | domain_adaptation.masked_retraining_teacher | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:iccv2025:Cui_Debiased_Teacher_for_Day-to-Night_Domain_Adaptive_Object_Detection` | domain_adaptation | code_bound | domain_adaptation.debiased_teacher | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `cvf:iccv2025:He_Dual-Rate_Dynamic_Teacher_for_Source-Free_Domain_Adaptive_Object_Detection` | domain_adaptation | code_bound | domain_adaptation.dual_rate_source_free | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `ecva:eccv2022:1356` | distillation | profiled | none | paper_specific_mechanism_interpretation | evidence_recovery |
| `ecva:eccv2022:2285` | distillation | profiled | none | paper_specific_mechanism_interpretation | evidence_recovery |
| `ecva:eccv2022:2717` | distillation | profiled | none | paper_specific_mechanism_interpretation | evidence_recovery |
| `ecva:eccv2022:3523` | distillation | profiled | none | paper_specific_mechanism_interpretation | evidence_recovery |
| `ecva:eccv2022:3958` | domain_adaptation | code_bound | domain_adaptation.3958 | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `ecva:eccv2022:6004` | distillation | profiled | none | paper_specific_mechanism_interpretation | evidence_recovery |
| `ecva:eccv2022:6328` | distillation | profiled | none | paper_specific_mechanism_interpretation | evidence_recovery |
| `ecva:eccv2024:11200` | distillation | profiled | none | paper_specific_mechanism_interpretation | evidence_recovery |
| `ecva:eccv2024:11254` | domain_adaptation | code_bound | domain_adaptation.11254 | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `ecva:eccv2024:6619` | distillation | profiled | none | paper_specific_mechanism_interpretation | evidence_recovery |
| `ecva:eccv2024:7083` | domain_adaptation | code_bound | domain_adaptation.7083 | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `neurips:2021:082a8bbf2c357c09f26675f9cf5bcba3-Abstract` | distillation | profiled | none | paper_specific_mechanism_interpretation | evidence_recovery |
| `neurips:2021:29c0c0ee223856f336d7ea8052057753-Abstract` | distillation | profiled | none | paper_specific_mechanism_interpretation | evidence_recovery |
| `neurips:2021:892c91e0a653ba19df81a90f89d99bcd-Abstract` | distillation | profiled | none | paper_specific_mechanism_interpretation | evidence_recovery |
| `neurips:2021:c0cccc24dd23ded67404f5e511c342b0-Abstract` | domain_adaptation | code_bound | domain_adaptation.c0cccc24dd23ded67404f5e511c342b0_abstract | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `neurips:2022:18c0102cb7f1a02c14f0929089b2e576-Abstract-Conference` | distillation | profiled | none | paper_specific_mechanism_interpretation | evidence_recovery |
| `neurips:2022:631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference` | distillation | profiled | none | paper_specific_mechanism_interpretation | evidence_recovery |
| `neurips:2024:6b6492cd06db22bac024506e9ed0925e-Abstract-Conference` | domain_adaptation | code_bound | domain_adaptation.6b6492cd06db22bac024506e9ed0925e_abstract_confer | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `neurips:2024:89d0d5c2f720921df93bbb8fef514571-Abstract-Conference` | domain_adaptation | code_bound | domain_adaptation.89d0d5c2f720921df93bbb8fef514571_abstract_confer | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `neurips:2024:bb71b5567ee985e0a4cee54ade19275c-Abstract-Conference` | domain_adaptation | code_bound | domain_adaptation.bb71b5567ee985e0a4cee54ade19275c_abstract_confer | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `neurips:2025:6460e378f24da3a79f20ac2640732a00-Abstract-Conference` | distillation | profiled | none | paper_specific_mechanism_interpretation | evidence_recovery |
| `papernotes:black-box_domain_adaptation_for_object_detection_with_retention-driven_knowledge` | domain_adaptation | code_bound | domain_adaptation.black_box_retention | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |
| `papernotes:expert-teacher-student_collaborative_learning_for_domain_adaptive_object_detecti` | domain_adaptation | code_bound | domain_adaptation.expert_teacher_student | non_mock_smoke_validation<br>unit_validation<br>yolo26_compatibility_validation | implementation_request |

`implementation_ready` is intentionally not inferred from a shared adapter, a recipe, a mock smoke, or a paper claim.
