# Paper Implementation Readiness

This is a paper-level implementation audit. It is not an exact paper reproduction, a pilot result, or a training result.

- Frozen papers: 83
- Manifest membership hash: `1d77fc0df6c9e348b99399368f16453f337640d9d35263fd6fecb5b71e95bce0`
- Implementation ready: 69
- Blocked: 14
- Not audited: 0

## Readiness Counts

| Readiness | Papers |
|---|---:|
| `cataloged` | 0 |
| `profiled` | 0 |
| `spec_complete` | 0 |
| `code_bound` | 14 |
| `runtime_integrated` | 0 |
| `unit_tested` | 0 |
| `smoke_passed` | 0 |
| `implementation_ready` | 69 |
| `pilot_reproduced` | 0 |
| `full_reproduced` | 0 |
| `confirmed_multi_seed` | 0 |

## Per-Paper Audit

| Paper ID | Profile | Readiness | Evidence class | Components | Adapters | Blockers |
|---|---|---|---|---|---|---|
| `arxiv:2103.14259` | `method-profile-7ae602a1f36a9e73b861` | implementation_ready | paper_specific | assigner.optimal_transport | assigner.optimal_transport | none |
| `arxiv:2104.14082` | `method-profile-c992f0017f0d90af0dfd` | implementation_ready | paper_specific | loss.quality.pseudo_iou | loss.quality.pseudo_iou | none |
| `arxiv:2107.08430` | `method-profile-fcbb3dde628bb7a2f7e4` | implementation_ready | paper_specific | assigner.optimal_transport | assigner.optimal_transport | none |
| `arxiv:2108.07755` | `method-profile-482ac2857e6c8acca601` | implementation_ready | paper_specific | detection_head.task_aligned | detection_head.task_aligned | none |
| `arxiv:2109.05986` | `method-profile-8ef600fcd9467f725290` | implementation_ready | paper_specific | loss.2109_05986 | loss.2109_05986 | none |
| `arxiv:2203.16250` | `method-profile-e6b9e22ea9447c9f508f` | implementation_ready | paper_specific | assigner.task_aligned | assigner.task_aligned | none |
| `arxiv:2208.00817` | `method-profile-c14d3a2b5f3983a65043` | implementation_ready | paper_specific | assigner.dynamic_smooth_label | assigner.dynamic_smooth_label | none |
| `arxiv:2210.11539` | `method-profile-febf9fd8db62295552ae` | implementation_ready | paper_specific | domain_adaptation.2210_11539 | domain_adaptation.2210_11539 | none |
| `arxiv:2212.07784` | `method-profile-8499fc8a32a471be62e1` | implementation_ready | paper_specific | neck.rtmdet_large_kernel | neck.rtmdet_large_kernel | none |
| `arxiv:2301.01019` | `method-profile-d85d404f39c75276392a` | implementation_ready | paper_specific | loss.quality.correlation | loss.quality.correlation | none |
| `arxiv:2303.13853` | `method-profile-b27388cb2e5e4d8faddb` | implementation_ready | paper_specific | domain_adaptation.2303_13853 | domain_adaptation.2303_13853 | none |
| `arxiv:2303.14404` | `method-profile-9c39f45fbb6b74688ae2` | implementation_ready | paper_specific | loss.calibration.bpc | loss.calibration.bpc | none |
| `arxiv:2309.11331` | `method-profile-20c825b0bf7f3a368bb6` | implementation_ready | paper_specific | feature_pyramid.multi_scale | feature_pyramid.multi_scale | none |
| `arxiv:2503.23220` | `method-profile-4d53dc6c401e209aa71b` | implementation_ready | paper_specific | domain_adaptation.2503_23220 | domain_adaptation.2503_23220 | none |
| `arxiv:2507.00721` | `method-profile-c9cb9d77a040a9d16d09` | implementation_ready | paper_specific | domain_adaptation.2507_00721 | domain_adaptation.2507_00721 | none |
| `arxiv:2603.12409` | `method-profile-889150037242e120da81` | implementation_ready | paper_specific | domain_adaptation.2603_12409 | domain_adaptation.2603_12409 | none |
| `arxiv:2603.18541` | `method-profile-1185d4bce4fc7c925931` | implementation_ready | paper_specific | domain_adaptation.2603_18541 | domain_adaptation.2603_18541 | none |
| `arxiv:2603.18757` | `method-profile-82110adeaa63d335e243` | implementation_ready | paper_specific | domain_adaptation.2603_18757 | domain_adaptation.2603_18757 | none |
| `arxiv:2603.28182` | `method-profile-951c8a49c122141a48dc` | implementation_ready | paper_specific | domain_adaptation.2603_28182 | domain_adaptation.2603_28182 | none |
| `cvf:cvpr2021:Dai_General_Instance_Distillation_for_Object_Detection` | `method-profile-c954bd070bfb8556e5a5` | implementation_ready | paper_specific | distillation.general_instance | distillation.general_instance | none |
| `cvf:cvpr2021:Guo_Distilling_Object_Detectors_via_Decoupled_Features` | `method-profile-aecae9c159db13c0f539` | implementation_ready | paper_specific | distillation.decoupled_features | distillation.decoupled_features | none |
| `cvf:cvpr2021:Hu_Dense_Relation_Distillation_With_Context-Aware_Aggregation_for_Few-Shot_Object_Detection` | `method-profile-97a94b0731fda34fb274` | implementation_ready | paper_specific | distillation.dense_relation_fewshot | distillation.dense_relation_fewshot | none |
| `cvf:cvpr2021:VS_MeGA-CDA_Memory_Guided_Attention_for_Category-Aware_Unsupervised_Domain_Adaptive_Object` | `method-profile-2102b896e4100c4ec1dd` | implementation_ready | paper_specific | domain_adaptation.mega_cda | domain_adaptation.mega_cda | none |
| `cvf:cvpr2021:Zhang_RPN_Prototype_Alignment_for_Domain_Adaptive_Object_Detector` | `method-profile-963cc2cea75b5eddc8b4` | implementation_ready | paper_specific | domain_adaptation.rpn_prototype_alignment | domain_adaptation.rpn_prototype_alignment | none |
| `cvf:cvpr2022:Feng_Overcoming_Catastrophic_Forgetting_in_Incremental_Object_Detection_via_Elastic_Response` | `method-profile-813ef4b961ee7d36c2c1` | implementation_ready | paper_specific | distillation.elastic_response_incremental | distillation.elastic_response_incremental | none |
| `cvf:cvpr2022:Guo_Scale-Equivalent_Distillation_for_Semi-Supervised_Object_Detection` | `method-profile-e0be6401ba8e40082dee` | implementation_ready | paper_specific | distillation.scale_equivalent | distillation.scale_equivalent | none |
| `cvf:cvpr2022:He_Cross_Domain_Object_Detection_by_Target-Perceived_Dual_Branch_Distillation` | `method-profile-5b806a22b2c399e1d5d3` | implementation_ready | paper_specific | distillation.target_perceived_dual_branch | distillation.target_perceived_dual_branch | none |
| `cvf:cvpr2022:Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection` | `method-profile-49f9590c3f756a156586` | implementation_ready | paper_specific | domain_adaptation.adaptive_teacher | domain_adaptation.adaptive_teacher | none |
| `cvf:cvpr2022:Li_SIGMA_Semantic-Complete_Graph_Matching_for_Domain_Adaptive_Object_Detection` | `method-profile-c5605da6e8115eabbf12` | implementation_ready | paper_specific | domain_adaptation.sigma_graph_matching | domain_adaptation.sigma_graph_matching | none |
| `cvf:cvpr2022:Wu_Single-Domain_Generalized_Object_Detection_in_Urban_Scene_via_Cyclic-Disentangled_Self-Distillation` | `method-profile-306354accf077a2355a2` | implementation_ready | paper_specific | distillation.cyclic_disentangled | distillation.cyclic_disentangled | none |
| `cvf:cvpr2022:Wu_Target-Relevant_Knowledge_Preservation_for_Multi-Source_Domain_Adaptive_Object_Detection` | `method-profile-3050828b3dff4210b0c5` | implementation_ready | paper_specific | domain_adaptation.multi_source_knowledge | domain_adaptation.multi_source_knowledge | none |
| `cvf:cvpr2022:Zhao_Task-Specific_Inconsistency_Alignment_for_Domain_Adaptive_Object_Detection` | `method-profile-62616df455f4e51e55d3` | implementation_ready | paper_specific | domain_adaptation.inconsistency_alignment | domain_adaptation.inconsistency_alignment | none |
| `cvf:cvpr2022:Zheng_Localization_Distillation_for_Dense_Object_Detection` | `method-profile-cfe41a21371f823b4820` | implementation_ready | paper_specific | distillation.localization | distillation.localization | none |
| `cvf:cvpr2022:Zhou_Multi-Granularity_Alignment_Domain_Adaptation_for_Object_Detection` | `method-profile-01c27d2d263efb09c2ce` | implementation_ready | paper_specific | domain_adaptation.multi_granularity | domain_adaptation.multi_granularity | none |
| `cvf:cvpr2023:Cao_Contrastive_Mean_Teacher_for_Domain_Adaptive_Object_Detectors` | `method-profile-98ecb36ab0192b0037f3` | implementation_ready | paper_specific | domain_adaptation.contrastive_mean_teacher | domain_adaptation.contrastive_mean_teacher | none |
| `cvf:cvpr2023:Gao_AsyFOD_An_Asymmetric_Adaptation_Paradigm_for_Few-Shot_Domain_Adaptive_Object` | `method-profile-86ab43219dd6a40cee41` | implementation_ready | paper_specific | domain_adaptation.asyfod | domain_adaptation.asyfod | none |
| `cvf:cvpr2023:Liu_CIGAR_Cross-Modality_Graph_Reasoning_for_Domain_Adaptive_Object_Detection` | `method-profile-42f70a4a8d2640f0ec2f` | implementation_ready | paper_specific | domain_adaptation.cigar_graph | domain_adaptation.cigar_graph | none |
| `cvf:cvpr2023:VS_Instance_Relation_Graph_Guided_Source-Free_Domain_Adaptive_Object_Detection` | `method-profile-e0eed3d12a393bff59a3` | implementation_ready | paper_specific | domain_adaptation.source_free_irg | domain_adaptation.source_free_irg | none |
| `cvf:cvpr2023:Wang_Object-Aware_Distillation_Pyramid_for_Open-Vocabulary_Object_Detection` | `method-profile-317fe67f15a06c3e6d10` | implementation_ready | paper_specific | distillation.object_aware_pyramid | distillation.object_aware_pyramid | none |
| `cvf:cvpr2023:Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector` | `method-profile-0a827aeed9aec321357a` | implementation_ready | paper_specific | distillation.scalekd | distillation.scalekd | none |
| `cvf:cvpr2024:Du_Boosting_Object_Detection_with_Zero-Shot_Day-Night_Domain_Adaptation` | `method-profile-16f22e2571695f4b1952` | implementation_ready | paper_specific | domain_adaptation.zero_shot_day_night | domain_adaptation.zero_shot_day_night | none |
| `cvf:cvpr2024:Kennerley_CAT_Exploiting_Inter-Class_Dynamics_for_Domain_Adaptive_Object_Detection` | `method-profile-fef5e21d59f498b69e25` | implementation_ready | paper_specific | domain_adaptation.cat_interclass | domain_adaptation.cat_interclass | none |
| `cvf:cvpr2024:Nakamura_Active_Domain_Adaptation_with_False_Negative_Prediction_for_Object_Detection` | `method-profile-973b0177625bd1c8ee41` | implementation_ready | paper_specific | domain_adaptation.active_false_negative | domain_adaptation.active_false_negative | none |
| `cvf:cvpr2024:Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection` | `method-profile-cfab371f0c96bd993a99` | implementation_ready | paper_specific | distillation.crosskd | distillation.crosskd | none |
| `cvf:cvpr2024:Yang_Active_Object_Detection_with_Knowledge_Aggregation_and_Distillation_from_Large` | `method-profile-e866bd5e18525c05d5f2` | implementation_ready | paper_specific | distillation.active_knowledge_aggregation | distillation.active_knowledge_aggregation | none |
| `cvf:cvpr2025:Li_SEEN-DA_SEmantic_ENtropy_guided_Domain-aware_Attention_for_Domain_Adaptive_Object` | `method-profile-1a3b0f2ee57f5322e676` | implementation_ready | paper_specific | domain_adaptation.seen_da | domain_adaptation.seen_da | none |
| `cvf:cvpr2025:Liu_Distinguish_Then_Exploit_Source-free_Open_Set_Domain_Adaptation_via_Weight` | `method-profile-a96c33cc73cd1c9e9923` | implementation_ready | paper_specific | domain_adaptation.source_free_open_set | domain_adaptation.source_free_open_set | none |
| `cvf:iccv2021:Chen_Deep_Structured_Instance_Graph_for_Distilling_Object_Detectors` | `method-profile-d2770fb0ee89e2a94e51` | implementation_ready | paper_specific | distillation.structured_instance_graph | distillation.structured_instance_graph | none |
| `cvf:iccv2021:Chen_Dual_Bipartite_Graph_Learning_A_General_Approach_for_Domain_Adaptive` | `method-profile-a39936c465d0191d4c8a` | implementation_ready | paper_specific | domain_adaptation.dual_bipartite_graph | domain_adaptation.dual_bipartite_graph | none |
| `cvf:iccv2021:Tian_Knowledge_Mining_and_Transferring_for_Domain_Adaptive_Object_Detection` | `method-profile-62b66f8edfbd557b1736` | implementation_ready | paper_specific | domain_adaptation.knowledge_mining | domain_adaptation.knowledge_mining | none |
| `cvf:iccv2021:Yao_G-DetKD_Towards_General_Distillation_Framework_for_Object_Detectors_via_Contrastive` | `method-profile-332922df25d659789428` | implementation_ready | paper_specific | distillation.gdetkd | distillation.gdetkd | none |
| `cvf:iccv2021:Yao_Multi-Source_Domain_Adaptation_for_Object_Detection` | `method-profile-3a7e6480afd96d363e20` | implementation_ready | paper_specific | domain_adaptation.multi_source | domain_adaptation.multi_source | none |
| `cvf:iccv2023:Gao_CSDA_Learning_Category-Scale_Joint_Feature_for_Domain_Adaptive_Object_Detection` | `method-profile-76ac5cc6d089c031a645` | implementation_ready | paper_specific | domain_adaptation.csda | domain_adaptation.csda | none |
| `cvf:iccv2023:Kang_Alleviating_Catastrophic_Forgetting_of_Incremental_Object_Detection_via_Within-Class_and` | `method-profile-142922e24104d01fd86a` | implementation_ready | paper_specific | distillation.incremental_within_class | distillation.incremental_within_class | none |
| `cvf:iccv2023:Lao_UniKD_Universal_Knowledge_Distillation_for_Mimicking_Homogeneous_or_Heterogeneous_Object` | `method-profile-d1ce7ec5b4db1d5c1791` | implementation_ready | paper_specific | distillation.unikd | distillation.unikd | none |
| `cvf:iccv2023:Wu_Spatial_Self-Distillation_for_Object_Detection_with_Inaccurate_Bounding_Boxes` | `method-profile-e3f4a9f318080f9ffc66` | implementation_ready | paper_specific | distillation.spatial_self | distillation.spatial_self | none |
| `cvf:iccv2023:Yang_Bridging_Cross-task_Protocol_Inconsistency_for_Distillation_in_Dense_Object_Detection` | `method-profile-df55a83786197423bc7f` | implementation_ready | paper_specific | distillation.cross_task_protocol | distillation.cross_task_protocol | none |
| `cvf:iccv2023:Zhao_Masked_Retraining_Teacher-Student_Framework_for_Domain_Adaptive_Object_Detection` | `method-profile-1bc20779c1f31638ad3c` | implementation_ready | paper_specific | domain_adaptation.masked_retraining_teacher | domain_adaptation.masked_retraining_teacher | none |
| `cvf:iccv2025:Cui_Debiased_Teacher_for_Day-to-Night_Domain_Adaptive_Object_Detection` | `method-profile-2626bd9cd057c76efdc5` | implementation_ready | paper_specific | domain_adaptation.debiased_teacher | domain_adaptation.debiased_teacher | none |
| `cvf:iccv2025:He_Dual-Rate_Dynamic_Teacher_for_Source-Free_Domain_Adaptive_Object_Detection` | `method-profile-98370172ed9618b5c6dd` | implementation_ready | paper_specific | domain_adaptation.dual_rate_source_free | domain_adaptation.dual_rate_source_free | none |
| `ecva:eccv2022:1356` | `method-profile-448a1980c5def94231ed` | code_bound | paper_specific | distillation.pgd_prediction_guided | distillation.pgd_prediction_guided | component_runtime_evidence_missing:distillation.pgd_prediction_guided<br>component_smoke_evidence_missing:distillation.pgd_prediction_guided<br>component_unit_evidence_missing:distillation.pgd_prediction_guided |
| `ecva:eccv2022:2285` | `method-profile-82575d80fba8d321a640` | code_bound | paper_specific | distillation.head_hetero_assist | distillation.head_hetero_assist | component_runtime_evidence_missing:distillation.head_hetero_assist<br>component_smoke_evidence_missing:distillation.head_hetero_assist<br>component_unit_evidence_missing:distillation.head_hetero_assist |
| `ecva:eccv2022:2717` | `method-profile-902852a98db72b50db57` | code_bound | paper_specific | distillation.global_kd_prototype | distillation.global_kd_prototype | component_runtime_evidence_missing:distillation.global_kd_prototype<br>component_smoke_evidence_missing:distillation.global_kd_prototype<br>component_unit_evidence_missing:distillation.global_kd_prototype |
| `ecva:eccv2022:3523` | `method-profile-cf5d46c67b03349eefdd` | code_bound | paper_specific | distillation.base_novel_commonality | distillation.base_novel_commonality | component_runtime_evidence_missing:distillation.base_novel_commonality<br>component_smoke_evidence_missing:distillation.base_novel_commonality<br>component_unit_evidence_missing:distillation.base_novel_commonality |
| `ecva:eccv2022:3958` | `method-profile-5e7ac24ee76dc651314d` | implementation_ready | paper_specific | domain_adaptation.3958 | domain_adaptation.3958 | none |
| `ecva:eccv2022:6004` | `method-profile-058d0df5273565fa63c9` | code_bound | paper_specific | distillation.bovw_consistency | distillation.bovw_consistency | component_runtime_evidence_missing:distillation.bovw_consistency<br>component_smoke_evidence_missing:distillation.bovw_consistency<br>component_unit_evidence_missing:distillation.bovw_consistency |
| `ecva:eccv2022:6328` | `method-profile-9e23fa7f561b0ab993a8` | code_bound | paper_specific | distillation.glamd_attention_mask | distillation.glamd_attention_mask | component_runtime_evidence_missing:distillation.glamd_attention_mask<br>component_smoke_evidence_missing:distillation.glamd_attention_mask<br>component_unit_evidence_missing:distillation.glamd_attention_mask |
| `ecva:eccv2024:11200` | `method-profile-dd0653c55a78de58b8e4` | code_bound | paper_specific | distillation.dlim_det_query | distillation.dlim_det_query | component_runtime_evidence_missing:distillation.dlim_det_query<br>component_smoke_evidence_missing:distillation.dlim_det_query<br>component_unit_evidence_missing:distillation.dlim_det_query |
| `ecva:eccv2024:11254` | `method-profile-3618252b5d116fc80476` | implementation_ready | paper_specific | domain_adaptation.11254 | domain_adaptation.11254 | none |
| `ecva:eccv2024:6619` | `method-profile-54ebbfb46261c9648b4a` | code_bound | paper_specific | distillation.mscd_cross_scale | distillation.mscd_cross_scale | component_runtime_evidence_missing:distillation.mscd_cross_scale<br>component_smoke_evidence_missing:distillation.mscd_cross_scale<br>component_unit_evidence_missing:distillation.mscd_cross_scale |
| `ecva:eccv2024:7083` | `method-profile-14c62d67c6de86e66204` | implementation_ready | paper_specific | domain_adaptation.7083 | domain_adaptation.7083 | none |
| `neurips:2021:082a8bbf2c357c09f26675f9cf5bcba3-Abstract` | `method-profile-bfeb652e3a94e633b4d4` | code_bound | paper_specific | distillation.classifier_kd | distillation.classifier_kd | component_runtime_evidence_missing:distillation.classifier_kd<br>component_smoke_evidence_missing:distillation.classifier_kd<br>component_unit_evidence_missing:distillation.classifier_kd |
| `neurips:2021:29c0c0ee223856f336d7ea8052057753-Abstract` | `method-profile-6344699126f2c7f1fdc1` | code_bound | paper_specific | distillation.frs_richness | distillation.frs_richness | component_runtime_evidence_missing:distillation.frs_richness<br>component_smoke_evidence_missing:distillation.frs_richness<br>component_unit_evidence_missing:distillation.frs_richness |
| `neurips:2021:892c91e0a653ba19df81a90f89d99bcd-Abstract` | `method-profile-78f5ccc79c628c3f1643` | code_bound | paper_specific | distillation.icd_instance_conditional | distillation.icd_instance_conditional | component_runtime_evidence_missing:distillation.icd_instance_conditional<br>component_smoke_evidence_missing:distillation.icd_instance_conditional<br>component_unit_evidence_missing:distillation.icd_instance_conditional |
| `neurips:2021:c0cccc24dd23ded67404f5e511c342b0-Abstract` | `method-profile-122ef1834c80308a9f27` | implementation_ready | paper_specific | domain_adaptation.c0cccc24dd23ded67404f5e511c342b0_abstract | domain_adaptation.c0cccc24dd23ded67404f5e511c342b0_abstract | none |
| `neurips:2022:18c0102cb7f1a02c14f0929089b2e576-Abstract-Conference` | `method-profile-fa385dcb9d0d699d4f77` | code_bound | paper_specific | distillation.structural_kd | distillation.structural_kd | component_runtime_evidence_missing:distillation.structural_kd<br>component_smoke_evidence_missing:distillation.structural_kd<br>component_unit_evidence_missing:distillation.structural_kd |
| `neurips:2022:631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference` | `method-profile-7e8076c79a494b6a5d42` | code_bound | paper_specific | distillation.pkd_pearson | distillation.pkd_pearson | component_runtime_evidence_missing:distillation.pkd_pearson<br>component_smoke_evidence_missing:distillation.pkd_pearson<br>component_unit_evidence_missing:distillation.pkd_pearson |
| `neurips:2024:6b6492cd06db22bac024506e9ed0925e-Abstract-Conference` | `method-profile-507b1e840547bcaa2529` | implementation_ready | paper_specific | domain_adaptation.6b6492cd06db22bac024506e9ed0925e_abstract_confer | domain_adaptation.6b6492cd06db22bac024506e9ed0925e_abstract_confer | none |
| `neurips:2024:89d0d5c2f720921df93bbb8fef514571-Abstract-Conference` | `method-profile-8e52b40cb4fb2c9e303f` | implementation_ready | paper_specific | domain_adaptation.89d0d5c2f720921df93bbb8fef514571_abstract_confer | domain_adaptation.89d0d5c2f720921df93bbb8fef514571_abstract_confer | none |
| `neurips:2024:bb71b5567ee985e0a4cee54ade19275c-Abstract-Conference` | `method-profile-225db28282babc2688f6` | implementation_ready | paper_specific | domain_adaptation.bb71b5567ee985e0a4cee54ade19275c_abstract_confer | domain_adaptation.bb71b5567ee985e0a4cee54ade19275c_abstract_confer | none |
| `neurips:2025:6460e378f24da3a79f20ac2640732a00-Abstract-Conference` | `method-profile-f04bbf6411a5e9310647` | code_bound | paper_specific | distillation.eldet_early_learning | distillation.eldet_early_learning | component_runtime_evidence_missing:distillation.eldet_early_learning<br>component_smoke_evidence_missing:distillation.eldet_early_learning<br>component_unit_evidence_missing:distillation.eldet_early_learning |
| `papernotes:black-box_domain_adaptation_for_object_detection_with_retention-driven_knowledge` | `method-profile-d8c63ac588aa9eb26bb8` | implementation_ready | paper_specific | domain_adaptation.black_box_retention | domain_adaptation.black_box_retention | none |
| `papernotes:expert-teacher-student_collaborative_learning_for_domain_adaptive_object_detecti` | `method-profile-fa58108169052c5b8f66` | implementation_ready | paper_specific | domain_adaptation.expert_teacher_student | domain_adaptation.expert_teacher_student | none |

A paper is `implementation_ready` only when its own paper-specific configuration and non-mock validation satisfy the strict boundary. `pilot_reproduced` remains a later state.