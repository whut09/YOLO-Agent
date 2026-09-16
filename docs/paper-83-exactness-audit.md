# Paper-83 Exactness Audit

Per-paper implementation-exactness inventory over the frozen 83. An audit result below 83/83 ready is a legitimate outcome: the report, artifacts, gap queue, and tests are still produced.

## Summary

```
Total: 83
Ready: 9
Blocked: 74
Out of scope: 0
Papers passing all 17 checks: 9
```

Training remains locked: this audit inventories evidence only and never starts model training.

## Status vocabulary

Only `implementation_ready` grants implementation status. Blocked statuses use the fixed blocker categories; component-level terms (`covered`, `mapped`, `certified_adapter`) are not readiness states.

## Results by status

| Status | Papers |
| --- | --- |
| `implementation_ready` | 9 |
| `blocked_missing_code` | 14 |
| `blocked_runtime` | 60 |

## Failing checks across the campaign

| Check | Failing papers |
| --- | --- |
| paper membership valid | 0 |
| MethodProfile valid | 0 |
| mechanism evidence available | 60 |
| PaperImplementationSpec complete | 14 |
| not generic-only | 14 |
| not alias-only | 0 |
| not metadata-only | 0 |
| not a no-op | 60 |
| paper-specific composition present | 14 |
| runtime hook real | 60 |
| source/runtime fingerprint present | 0 |
| unit tests | 60 |
| non-mock smoke | 60 |
| compatibility tests | 52 |
| rollback path | 0 |
| shared primitives referenced, not copied | 0 |
| core mechanism covered | 14 |

## Blocked papers by category

### `blocked_missing_code` (14 papers)

- `ecva:eccv2022:1356` — failed: core_mechanism_covered, implementation_spec_complete, not_generic_only, paper_specific_composition
- `ecva:eccv2022:2285` — failed: core_mechanism_covered, implementation_spec_complete, not_generic_only, paper_specific_composition
- `ecva:eccv2022:2717` — failed: core_mechanism_covered, implementation_spec_complete, not_generic_only, paper_specific_composition
- `ecva:eccv2022:3523` — failed: core_mechanism_covered, implementation_spec_complete, not_generic_only, paper_specific_composition
- `ecva:eccv2022:6004` — failed: core_mechanism_covered, implementation_spec_complete, not_generic_only, paper_specific_composition
- `ecva:eccv2022:6328` — failed: core_mechanism_covered, implementation_spec_complete, not_generic_only, paper_specific_composition
- `ecva:eccv2024:11200` — failed: core_mechanism_covered, implementation_spec_complete, not_generic_only, paper_specific_composition
- `ecva:eccv2024:6619` — failed: core_mechanism_covered, implementation_spec_complete, not_generic_only, paper_specific_composition
- `neurips:2021:082a8bbf2c357c09f26675f9cf5bcba3-Abstract` — failed: core_mechanism_covered, implementation_spec_complete, not_generic_only, paper_specific_composition
- `neurips:2021:29c0c0ee223856f336d7ea8052057753-Abstract` — failed: core_mechanism_covered, implementation_spec_complete, not_generic_only, paper_specific_composition
- `neurips:2021:892c91e0a653ba19df81a90f89d99bcd-Abstract` — failed: core_mechanism_covered, implementation_spec_complete, not_generic_only, paper_specific_composition
- `neurips:2022:18c0102cb7f1a02c14f0929089b2e576-Abstract-Conference` — failed: core_mechanism_covered, implementation_spec_complete, not_generic_only, paper_specific_composition
- `neurips:2022:631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference` — failed: core_mechanism_covered, implementation_spec_complete, not_generic_only, paper_specific_composition
- `neurips:2025:6460e378f24da3a79f20ac2640732a00-Abstract-Conference` — failed: core_mechanism_covered, implementation_spec_complete, not_generic_only, paper_specific_composition

### `blocked_runtime` (60 papers)

- `arxiv:2108.07755` — failed: mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `arxiv:2109.05986` — failed: mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `arxiv:2210.11539` — failed: mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `arxiv:2303.13853` — failed: mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `arxiv:2309.11331` — failed: mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `arxiv:2503.23220` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `arxiv:2507.00721` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `arxiv:2603.12409` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `arxiv:2603.18541` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `arxiv:2603.18757` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `arxiv:2603.28182` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2021:Dai_General_Instance_Distillation_for_Object_Detection` — failed: mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2021:Guo_Distilling_Object_Detectors_via_Decoupled_Features` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2021:Hu_Dense_Relation_Distillation_With_Context-Aware_Aggregation_for_Few-Shot_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2021:VS_MeGA-CDA_Memory_Guided_Attention_for_Category-Aware_Unsupervised_Domain_Adaptive_Object` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2021:Zhang_RPN_Prototype_Alignment_for_Domain_Adaptive_Object_Detector` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2022:Feng_Overcoming_Catastrophic_Forgetting_in_Incremental_Object_Detection_via_Elastic_Response` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2022:Guo_Scale-Equivalent_Distillation_for_Semi-Supervised_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2022:He_Cross_Domain_Object_Detection_by_Target-Perceived_Dual_Branch_Distillation` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2022:Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection` — failed: mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2022:Li_SIGMA_Semantic-Complete_Graph_Matching_for_Domain_Adaptive_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2022:Wu_Single-Domain_Generalized_Object_Detection_in_Urban_Scene_via_Cyclic-Disentangled_Self-Distillation` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2022:Wu_Target-Relevant_Knowledge_Preservation_for_Multi-Source_Domain_Adaptive_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2022:Zhao_Task-Specific_Inconsistency_Alignment_for_Domain_Adaptive_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2022:Zhou_Multi-Granularity_Alignment_Domain_Adaptation_for_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2023:Cao_Contrastive_Mean_Teacher_for_Domain_Adaptive_Object_Detectors` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2023:Gao_AsyFOD_An_Asymmetric_Adaptation_Paradigm_for_Few-Shot_Domain_Adaptive_Object` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2023:Liu_CIGAR_Cross-Modality_Graph_Reasoning_for_Domain_Adaptive_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2023:VS_Instance_Relation_Graph_Guided_Source-Free_Domain_Adaptive_Object_Detection` — failed: mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2023:Wang_Object-Aware_Distillation_Pyramid_for_Open-Vocabulary_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2023:Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2024:Du_Boosting_Object_Detection_with_Zero-Shot_Day-Night_Domain_Adaptation` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2024:Kennerley_CAT_Exploiting_Inter-Class_Dynamics_for_Domain_Adaptive_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2024:Nakamura_Active_Domain_Adaptation_with_False_Negative_Prediction_for_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2024:Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2024:Yang_Active_Object_Detection_with_Knowledge_Aggregation_and_Distillation_from_Large` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2025:Li_SEEN-DA_SEmantic_ENtropy_guided_Domain-aware_Attention_for_Domain_Adaptive_Object` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:cvpr2025:Liu_Distinguish_Then_Exploit_Source-free_Open_Set_Domain_Adaptation_via_Weight` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:iccv2021:Chen_Deep_Structured_Instance_Graph_for_Distilling_Object_Detectors` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:iccv2021:Chen_Dual_Bipartite_Graph_Learning_A_General_Approach_for_Domain_Adaptive` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:iccv2021:Tian_Knowledge_Mining_and_Transferring_for_Domain_Adaptive_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:iccv2021:Yao_G-DetKD_Towards_General_Distillation_Framework_for_Object_Detectors_via_Contrastive` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:iccv2021:Yao_Multi-Source_Domain_Adaptation_for_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:iccv2023:Gao_CSDA_Learning_Category-Scale_Joint_Feature_for_Domain_Adaptive_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:iccv2023:Kang_Alleviating_Catastrophic_Forgetting_of_Incremental_Object_Detection_via_Within-Class_and` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:iccv2023:Lao_UniKD_Universal_Knowledge_Distillation_for_Mimicking_Homogeneous_or_Heterogeneous_Object` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:iccv2023:Wu_Spatial_Self-Distillation_for_Object_Detection_with_Inaccurate_Bounding_Boxes` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:iccv2023:Yang_Bridging_Cross-task_Protocol_Inconsistency_for_Distillation_in_Dense_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:iccv2023:Zhao_Masked_Retraining_Teacher-Student_Framework_for_Domain_Adaptive_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:iccv2025:Cui_Debiased_Teacher_for_Day-to-Night_Domain_Adaptive_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `cvf:iccv2025:He_Dual-Rate_Dynamic_Teacher_for_Source-Free_Domain_Adaptive_Object_Detection` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `ecva:eccv2022:3958` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `ecva:eccv2024:11254` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `ecva:eccv2024:7083` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `neurips:2021:c0cccc24dd23ded67404f5e511c342b0-Abstract` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `neurips:2024:6b6492cd06db22bac024506e9ed0925e-Abstract-Conference` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `neurips:2024:89d0d5c2f720921df93bbb8fef514571-Abstract-Conference` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `neurips:2024:bb71b5567ee985e0a4cee54ade19275c-Abstract-Conference` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `papernotes:black-box_domain_adaptation_for_object_detection_with_retention-driven_knowledge` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present
- `papernotes:expert-teacher-student_collaborative_learning_for_domain_adaptive_object_detecti` — failed: compatibility_tests_present, mechanism_evidence_available, non_mock_smoke_present, not_no_op, runtime_hook_real, unit_tests_present

## Gap queue

```
Gap entries: 408
Blocked papers: 74
blocked_compatibility: 52
blocked_missing_code: 56
blocked_missing_evidence: 60
blocked_runtime: 120
blocked_test: 120
```

Every gap entry carries a recommended fix, an estimated scope, and its dependency; see `artifacts/paper_83_gap_queue.yaml`.

## Ready papers

- `arxiv:2103.14259` — OTA: Optimal Transport Assignment for Object Detection
- `arxiv:2104.14082` — Pseudo-IoU: Improving Label Assignment in Anchor-Free Object Detection
- `arxiv:2107.08430` — YOLOX: Exceeding YOLO Series in 2021
- `arxiv:2203.16250` — PP-YOLOE: An Evolved Version of YOLO
- `arxiv:2208.00817` — DSLA: Dynamic Smooth Label Assignment for Efficient Anchor-Free Object Detection
- `arxiv:2212.07784` — RTMDet: An Empirical Study of Designing Real-Time Object Detectors
- `arxiv:2301.01019` — Correlation Loss: Enforcing Correlation between Classification and Localization
- `arxiv:2303.14404` — Bridging Precision and Confidence: A Train-Time Loss for Calibrating Object Detection
- `cvf:cvpr2022:Zheng_Localization_Distillation_for_Dense_Object_Detection` — Localization Distillation for Dense Object Detection

