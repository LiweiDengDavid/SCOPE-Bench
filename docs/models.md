# Model References

All models currently implemented in NexusRec.

`Via` means the migration or reference source. It does not automatically mean benchmark equivalence.

## Centralized CF (ID-only)


| Model    | Paper                                                                                                                  | Venue                           | Official Code                                                                                           | Via     |
| -------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------- | ------- |
| BPR      | Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback."                                     | UAI 2009                        | —                                                                                                       | RecBole |
| LightGCN | Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation."                  | SIGIR 2020                      | [kuandeng/LightGCN](https://github.com/kuandeng/LightGCN)                                               | RecBole |
| NCF      | Xiangnan He et al. "Neural Collaborative Filtering." (implemented with the paper's full NeuMF architecture)             | WWW 2017                        | [hexiangnan/neuralcollaborativefiltering](https://github.com/hexiangnan/neural_collaborative_filtering) | RecBole |
| SGL      | Jiancan Wu et al. "Self-supervised Graph Learning for Recommendation."                                                 | SIGIR 2021                      | [wujcan/SGL-Torch](https://github.com/wujcan/SGL-Torch)                                                 | RecBole |
| MultiVAE | Dawen Liang et al. "Variational Autoencoders for Collaborative Filtering."                                             | WWW 2018                        | [dawenl/vaecf](https://github.com/dawenl/vae_cf)                                                        | RecBole |
| RecVAE   | Ilya Shenbin et al. "RecVAE: A New Variational Autoencoder for Top-N Recommendations with Implicit Feedback."          | WSDM 2020                       | [ilya-shenbin/RecVAE](https://github.com/ilya-shenbin/RecVAE)                                           | RecBole |
| DiffRec  | Wenjie Wang et al. "Diffusion Recommender Model."                                                                      | SIGIR 2023                      | [YiyanXu/DiffRec](https://github.com/YiyanXu/DiffRec)                                                   | RecBole |
| FlowCF   | Chengkai Liu et al. "Flow Matching for Collaborative Filtering."                                                       | KDD 2025                        | [chengkai-liu/FlowCF](https://github.com/chengkai-liu/FlowCF)                                           | FlowCF  |
| CFDiff   | Yu Hou et al. "Collaborative Filtering Based on Diffusion Models: Unveiling the Potential of High-Order Connectivity." | SIGIR 2024                      | [jackfrost168/CFDiff](https://github.com/jackfrost168/CF_Diff)                                          | CFDiff  |


---

## Centralized Multimodal


| Model      | Paper                                                                                                                                 | Venue         | Official Code                                                                                                   | Via   |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------- | ----- |
| VBPR       | Ruining He, Julian McAuley. "VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback."                                      | AAAI 2016     | —                                                                                                               | MMRec |
| BM3        | Xin Zhou et al. "Bootstrap Latent Representations for Multi-modal Recommendation."                                                    | WWW 2023      | [enoche/BM3](https://github.com/enoche/BM3)                                                                     | MMRec |
| FREEDOM    | Xin Zhou et al. "A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation."                        | MM 2023       | [enoche/FREEDOM](https://github.com/enoche/FREEDOM)                                                             | MMRec |
| LATTICE    | Jinghao Zhang et al. "Mining Latent Structures for Multimedia Recommendation."                                                        | MM 2021       | [CRIPAC-DIG/LATTICE](https://github.com/CRIPAC-DIG/LATTICE)                                                     | MMRec |
| DRAGON     | Hongyu Zhou et al. "Enhancing Dyadic Relations with Homogeneous Graphs for Multimodal Recommendation."                                | ECAI 2023     | [hongyurain/DRAGON](https://github.com/hongyurain/DRAGON)                                                       | MMRec |
| DualGNN    | Qifan Wang et al. "Dual Graph Neural Network for Multimedia Recommendation."                                                          | IEEE TMM 2021 | [wqf321/dualgnn](https://github.com/wqf321/dualgnn)                                                             | MMRec |
| GRCN       | Yinwei Wei et al. "Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback."                         | MM 2020       | [weiyinwei/GRCN](https://github.com/weiyinwei/GRCN)                                                             | MMRec |
| MGCN       | Penghang Yu et al. "Multi-View Graph Convolutional Network for Multimedia Recommendation."                                            | MM 2023       | [NJUPT-MCC/MGCN](https://github.com/NJUPT-MCC/MGCN)                                                             | MMRec |
| MVGAE      | Jing Yi, Zhenzhong Chen. "Multi-Modal Variational Graph Auto-Encoder for Recommendation Systems."                                     | IEEE TMM 2021 | —                                                                                                               | MMRec |
| MMGCN      | Yinwei Wei et al. "MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video."                      | MM 2019       | [weiyinwei/MMGCN](https://github.com/weiyinwei/MMGCN)                                                           | MMRec |
| SLMRec     | Zhulin Tao et al. "Self-supervised Learning for Multimedia Recommendation."                                                           | IEEE TMM 2022 | [zltao/SLMRec](https://github.com/zltao/SLMRec)                                                                 | MMRec |
| DiffMM     | Yangqin Jiang et al. "Multi-Modal Diffusion Model for Recommendation."                                                                | MM 2024       | [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM)                                                                 | MMRec |
| IDFREE     | Based on a retracted paper. Implementation retained for reference; results should not be used in published comparisons.               | —             | —                                                                                                               | MMRec |
| ItemKNNCBF | Maurizio Ferrari Dacrema et al. "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches." | RecSys 2019   | [MaurizioFD/RecSys2019DeepLearningEvaluation](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation) | MMRec |

### Additional multimodal implementations

These implementations were added after the original NexusRec model table. “Source snapshot” means that the imported implementation documents its paper/code provenance in the model module, but the local source did not contain a verified public repository URL; a release URL should be added only after it is confirmed.

| Model | Paper / description | Venue | Reference code |
| --- | --- | --- | --- |
| BGCC | Behavior-Guided Candidate Calibration for Multimodal Recommendation | AAAI 2026 | [LIZESHENG13/bridge](https://github.com/LIZESHENG13/bridge) |
| BeFA | A General Behavior-driven Feature Adapter for Multimedia Recommendation | AAAI 2025 | [fqldom/BeFA](https://github.com/fqldom/BeFA) |
| CCDRec | Curriculum Conditional Diffusion Recommendation | AAAI 2025 | Source snapshot |
| CM3 | Calibrating Multimodal Recommendation | arXiv 2025 | [enoche/CM3](https://github.com/enoche/CM3) |
| COHESION | Co-optimized Heterogeneous Fusion for Multimodal Recommendation | SIGIR 2025 | Source snapshot |
| DA_MRS | Improving Multi-modal Recommender Systems by Denoising and Aligning Multi-modal Content and User Feedback | KDD 2024 | [GuipengXu/MA-MRS](https://github.com/GuipengXu/MA-MRS) |
| DGMRec | Disentangling and Generating Modalities for Recommendation in Missing Modality Scenarios | SIGIR 2025 | [HimTo-Kim/DGMRec](https://github.com/HimTo-Kim/DGMRec) |
| FITMM | Frequency-aware Information-bottleneck for MultiModal Recommendation | ACM MM 2025 | Source snapshot |
| LGMRec | Local and Global Graph Learning for Multimodal Recommendation | AAAI 2024 | [georgeguo-cn/LGMRec](https://github.com/georgeguo-cn/LGMRec) |
| MENTOR | Multi-level Self-supervised Learning for Multimodal Recommendation | AAAI 2025 | [Jinfeng-Xu/MENTOR](https://github.com/Jinfeng-Xu/MENTOR) |
| MIG_GT | Modality-Independent Graph Neural Networks with Global Transformers for Multimodal Recommendation | AAAI 2025 | [CrawlScript/MIG-GT](https://github.com/CrawlScript/MIG-GT) |
| PGL | Mind Individual Information! Principal Graph Learning for Multimedia Recommendation | AAAI 2025 | [demonph10/PGL](https://github.com/demonph10/PGL) |
| R2MR | Review and Rewrite: Modality-consensus Multimodal Recommendation | KDD 2025 | Source snapshot |
| REARM | Refining Contrastive Learning and Homography Relations for Multi-Modal Recommendation | ACM MM 2025 | [ACM Digital Library](https://dl.acm.org/doi/10.1145/3746027.3755779) |
| STAIR | Manipulating Collaborative and Multimodal Information for E-Commerce Recommendation | AAAI 2025 | [yizhenzhong/STAIR](https://github.com/yizhenzhong/STAIR) |
| TMLP | Topological Multi-Layer Perceptron for Multimodal Recommendation | AAAI 2025 | Source snapshot |
| TimeMM | Time-aware Multi-scale MultiModal Recommendation | SIGIR 2026 | Source snapshot |


---

## Federated (ID-only)


| Model   | Paper                                                                                                                       | Venue                        | Official Code                                                               | Via                    |
| ------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | --------------------------------------------------------------------------- | ---------------------- |
| FedAvg  | Brendan McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data."                         | AISTATS 2017                 | —                                                                           | Personalized FedRecSys |
| FCF     | Muhammad Ammad-Ud-Din et al. "Federated Collaborative Filtering for Privacy-Preserving Personalized Recommendation System." | arXiv 2019                   | —                                                                           | Personalized FedRecSys |
| FedNCF  | Vasileios Perifanis, Pavlos S. Efraimidis. "Federated Neural Collaborative Filtering."                                      | Knowledge-Based Systems 2022 | —                                                                           | Personalized FedRecSys |
| FedRAP  | Zhiwei Li et al. "Federated Recommendation with Additive Personalization."                                                  | ICLR 2024                    | [mtics/FedRAP](https://github.com/mtics/FedRAP)                             | Personalized FedRecSys |
| PFedRec | Chunxu Zhang et al. "Dual Personalization on Federated Recommendation."                                                     | IJCAI 2023                   | [Zhangcx19/IJCAI-23-PFedRec](https://github.com/Zhangcx19/IJCAI-23-PFedRec) | Personalized FedRecSys |


---

## Federated Multimodal

The FedVLR-family host models keep the original personalized fusion path.
MMPFedRec extends the PFedRec personalized head with the same lightweight
item-side multimodal fusion interface used by the FedVLR family.


| Model    | Paper                                                                                                  | Venue     | Official Code                                   | Via    |
| -------- | ------------------------------------------------------------------------------------------------------ | --------- | ----------------------------------------------- | ------ |
| MMFedAvg | Zhiwei Li et al. "Federated Vision-Language-Recommendation with Personalized Fusion." arXiv:2410.08478 | AAAI 2026 | [mtics/FedVLR](https://github.com/mtics/FedVLR) | FedVLR |
| MMFedNCF | Zhiwei Li et al. "Federated Vision-Language-Recommendation with Personalized Fusion." arXiv:2410.08478 | AAAI 2026 | [mtics/FedVLR](https://github.com/mtics/FedVLR) | FedVLR |
| MMFedRAP | Zhiwei Li et al. "Federated Vision-Language-Recommendation with Personalized Fusion." arXiv:2410.08478 | AAAI 2026 | [mtics/FedVLR](https://github.com/mtics/FedVLR) | FedVLR |
| MMFCF    | Zhiwei Li et al. "Federated Vision-Language-Recommendation with Personalized Fusion." arXiv:2410.08478 | AAAI 2026 | [mtics/FedVLR](https://github.com/mtics/FedVLR) | FedVLR |
| MMPFedRec | Chunxu Zhang et al. "Dual Personalization on Federated Recommendation."; Zhiwei Li et al. "Federated Vision-Language-Recommendation with Personalized Fusion." | IJCAI 2023 / AAAI 2026 | [Zhangcx19/IJCAI-23-PFedRec](https://github.com/Zhangcx19/IJCAI-23-PFedRec); [mtics/FedVLR](https://github.com/mtics/FedVLR) | PFedRec + FedVLR |


---

## Sequential


| Model    | Paper                                                                                                             | Venue     | Official Code                                         | Via     |
| -------- | ----------------------------------------------------------------------------------------------------------------- | --------- | ----------------------------------------------------- | ------- |
| BERT4Rec | Fei Sun et al. "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer." | CIKM 2019 | [FeiSun/BERT4Rec](https://github.com/FeiSun/BERT4Rec) | RecBole |
| GRU4Rec  | Balázs Hidasi et al. "Session-based Recommendations with Recurrent Neural Networks."                              | ICLR 2016 | [hidasib/GRU4Rec](https://github.com/hidasib/GRU4Rec) | RecBole |
| SASRec   | Wang-Cheng Kang, Julian McAuley. "Self-Attentive Sequential Recommendation."                                      | ICDM 2018 | [kang205/SASRec](https://github.com/kang205/SASRec)   | RecBole |
| HM4SR    | Zihao Li et al. "Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation."           | WWW 2025  | [SStarCCat/HM4SR](https://github.com/SStarCCat/HM4SR) | FedVLR  |


---

## Source Repositories
| Source                 | URL                                                                                    | Coverage                                                                                                                               |
| ---------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| MMRec                  | [enoche/MMRec](https://github.com/enoche/MMRec)                                        | Centralized multimodal models                                                                                                          |
| RecBole                | [RUCAIBox/RecBole](https://github.com/RUCAIBox/RecBole)                                | Centralized CF baselines (BPR, LightGCN, NCF, SGL, MultiVAE, RecVAE, DiffRec); sequential baselines (BERT4Rec, GRU4Rec, SASRec) |
| Personalized FedRecSys | [Zhangcx19/PersonalizedFedRecSys](https://github.com/Zhangcx19/Personalized_FedRecSys) | Federated ID-only models                                                                                                               |
| FedVLR                 | [mtics/FedVLR](https://github.com/mtics/FedVLR)                                        | Federated multimodal models                                                                                                            |
| FlowCF                 | [chengkai-liu/FlowCF](https://github.com/chengkai-liu/FlowCF)                          | FlowCF                                                                                                                                 |
| CFDiff                 | [jackfrost168/CFDiff](https://github.com/jackfrost168/CF_Diff)                         | CFDiff                                                                                                                                 |
