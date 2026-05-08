<div align="center">

<h1>Awesome Pruning</h1>

<p>
  <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
  <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fghimiredhikura%2FAwasome-Pruning&label=Visitors&countColor=%23263759" alt="Visitors">
  <img src="https://img.shields.io/badge/Last%20Updated-2026--05--08-0A7EA4" alt="Last Updated">
  <img src="https://img.shields.io/badge/Resources-Papers%20%7C%20Surveys%20%7C%20Code-0F766E" alt="Resources">
</p>

<p><em>A curated list of neural network pruning papers, surveys, toolboxes, and code resources across CNNs, ViTs, LLMs, VLMs/MLLMs, diffusion, and deployment-aware compression.</em></p>

<p><strong>Maintained by Deepak Ghimire and the community</strong></p>

<p>Inspired by <a href="https://github.com/he-y/Awesome-Pruning">he-y/Awesome-Pruning</a></p>

</div>

> [!NOTE]
> Pull requests for missing papers, code links, and corrections are welcome.
> If this repository helps your work, please cite it using the BibTeX in [Citation](#citation).
> **Last updated:** 2026-05-08. Recent additions emphasize pruning for LLMs, VLMs/MLLMs, ViTs, diffusion/3DGS, structured sparsity, semi-structured sparsity, and deployment-oriented compression.

## Contents

| Section | Quick Links |
|:--|:--|
| **[Conference Publications](#conference-publications)** | [`2026`](#2026) [`2025`](#2025) [`2024`](#2024) [`2023`](#2023) [`2022`](#2022) [`2021`](#2021) [`2020`](#2020) [`2019`](#2019) [`2018`](#2018) [`2017`](#2017) |
| **[Journal Publications](#journal-publications)** | [`2026`](#2026-1) [`2025`](#2025-1) [`2024`](#2024-1) [`2023`](#2023-1) [`2022`](#2022-1) [`2021`](#2021-1) [`2020`](#2020-1) |
| **[Survey Articles](#survey-articles)** | [`2020~2026`](#survey-articles) |
| **[Other Publications](#other-publications)** | [`2022~2026`](#other-publications) |
| **[Pruning Software and Toolbox](#pruning-software-and-toolbox)** | [`2019~2026`](#pruning-software-and-toolbox) |
| **[Citation](#citation)** | [`BibTeX`](#citation) |

## Legend

| Symbol | Meaning |
|:--:|:--|
| `U` | Unstructured / weight pruning |
| `S` | Structured / filter / channel / neuron / head / layer pruning |
| `SS` | Semi-structured sparsity, e.g., N:M / 2:4 |
| `T` | Token / patch / KV-cache pruning or token reduction |
| `D` | Dynamic / input-adaptive pruning |
| `Q` | Joint pruning and quantization |
| `A` | Official / author implementation |
| `O` | Unofficial / third-party implementation |
| `-` | Code not found or not clearly public |

## Quick Recent Trends

| Trend | Representative Methods |
|:--|:--|
| LLM structural pruning | LLM-Pruner, Sheared LLaMA, LoRAPrune, 2SSP, PAT, D2 Prune, ARMOR |
| LLM unstructured / semi-structured pruning | SparseGPT, Wanda, RIA, Wanda++, SLoRB, PermLLM |
| VLM / MLLM visual-token pruning | ATP-LLaVA, DivPrune, PACT, TopV, HiMAP, DyCoke, SGL |
| General automatic structured pruning | DepGraph / Torch-Pruning, OTOv2/HESSO, GETA |
| ViT token pruning / merging | ToMe, Token Cropr, Zero-TPrune, Token Fusion |
| Hardware/deployment-aware compression | HALP, GETA, NPAS, APQ, 2:4 pruning, pruning+quantization |

## Conference Publications

**<h3 align="center">2026</h3>**

| Venue | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| `ICLR` | [ARMOR: High-Performance Semi-Structured Pruning via Block-Diagonal Masking](https://openreview.net/forum?id=8NE554wv0m) | LLM | `SS` | - | Uses block-diagonal structure for high-performance semi-structured pruning. |
| `AAAI` | [D2 Prune: Sparsifying Large Language Models via Dual Distribution-Aware Calibration](https://ojs.aaai.org/index.php/AAAI/article/view/39932) | LLM | `U/SS` | - | Addresses activation distribution shift and long-tail activation behavior. |
| `EACL Industry` | [Iterative Structured Pruning for Large Language Models](https://aclanthology.org/2026.eacl-industry.1/) | LLM | `S` | - | Iterative structured pruning with hybrid calibration for downstream generalization. |
| `CPAL` | [Understanding Neural Network Pruning via Infinite Width Graph Limits](https://openreview.net/forum?id=HJkdDRmUzi) | Theory | `U/S` | - | Theoretical graphon perspective on sparse networks induced by pruning. |
| `arXiv` | [Efficient Post-Training Pruning of Large Language Models with Statistical Correction](https://arxiv.org/abs/2602.07375) | LLM | `U/S` | - | Post-training pruning with statistical correction. |
| `arXiv` | [From Local to Global: Revisiting Structured Pruning for Large Language Models](https://arxiv.org/abs/2510.18030) | LLM | `S` | - | Global/task-aware view of structured pruning; arXiv version updated in 2026. |
| `arXiv` | [GETA-3DGS: Automatic Joint Structured Pruning and Quantization for 3D Gaussian Splatting](https://arxiv.org/abs/2605.02086) | 3DGS | `S/Q` | - | Extends joint pruning-quantization ideas to 3D Gaussian Splatting. |

**<h3 align="center">2025</h3>**

| Venue | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| `CVPR` | [Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression](https://openaccess.thecvf.com/content/CVPR2025/html/Qu_Automatic_Joint_Structured_Pruning_and_Quantization_for_Efficient_Neural_Network_CVPR_2025_paper.html) | General | `S/Q` | [PyTorch[A]](https://github.com/microsoft/geta) | GETA; architecture-agnostic joint structured pruning + mixed-precision QAT. |
| `CVPR` | [ATP-LLaVA: Adaptive Token Pruning for Large Vision Language Models](https://openaccess.thecvf.com/content/CVPR2025/html/Ye_ATP-LLaVA_Adaptive_Token_Pruning_for_Large_Vision_Language_Models_CVPR_2025_paper.html) | LVLM | `T/D` | - | Layer-wise and instance-wise adaptive visual-token pruning. |
| `CVPR` | [DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models](https://openaccess.thecvf.com/content/CVPR2025/html/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.html) | MLLM | `T` | [PyTorch[A]](https://github.com/vbdi/divprune) | Training-free, calibration-free diversity-based visual-token selection. |
| `CVPR` | [Token Cropr: Faster ViTs for Quite a Few Tasks](https://openaccess.thecvf.com/content/CVPR2025/html/Bergner_Token_Cropr_Faster_ViTs_for_Quite_a_Few_Tasks_CVPR_2025_paper.html) | ViT | `T/D` | [PyTorch[A]](https://github.com/benbergner/cropr) | End-to-end token pruner; reported 1.5×–4× speedups across vision tasks. |
| `CVPR` | [PACT: Pruning and Clustering-Based Token Reduction for Faster Visual Language Models](https://openaccess.thecvf.com/content/CVPR2025/html/Dhouib_PACT_Pruning_and_Clustering-Based_Token_Reduction_for_Faster_Visual_Language_CVPR_2025_paper.html) | VLM | `T` | [PyTorch[A]](https://github.com/orailix/PACT) | Combines pruning of irrelevant tokens with clustering/merging of redundant tokens. |
| `CVPR` | [TopV: Compatible Token Pruning with Inference Time Optimization for Fast and Low-Memory Multimodal Vision Language Model](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_TopV_Compatible_Token_Pruning_with_Inference_Time_Optimization_for_Fast_CVPR_2025_paper.html) | MLLM | `T/D` | - | Prefill-stage token pruning compatible with FlashAttention and KV-cache reduction. |
| `CVPR` | [Lifting the Veil on Visual Information Flow in MLLMs: Unlocking Efficient Visual Token Pruning](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_Lifting_the_Veil_on_Visual_Information_Flow_in_MLLMs_Unlocking_CVPR_2025_paper.html) | MLLM | `T/D` | - | Introduces HiMAP, a modality-aware plug-and-play visual-token pruning method. |
| `CVPR` | [Libra-Merging: Importance-Redundancy and Pruning-Merging Trade-off for Acceleration Plug-in in Large Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Libra-Merging_Importance-redundancy_and_Pruning-merging_Trade-off_for_Acceleration_Plug-in_in_Large_CVPR_2025_paper.html) | LVLM | `T` | - | Balances pruning and merging to handle token importance/redundancy trade-off. |
| `CVPR` | [DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models](https://openaccess.thecvf.com/content/CVPR2025/html/Tao_DyCoke_Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_CVPR_2025_paper.html) | Video-LLM | `T/D` | - | Training-free temporal compression and dynamic KV-cache reduction. |
| `CVPR` | [A Stitch in Time Saves Nine: Small VLM is a Precise Guidance for Accelerating Large VLMs](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_A_Stitch_in_Time_Saves_Nine_Small_VLM_is_a_CVPR_2025_paper.html) | VLM | `T/D` | - | Small VLM guidance for visual-token pruning in a larger VLM. |
| `ICLR` | [The Unreasonable Ineffectiveness of the Deeper Layers](https://openreview.net/forum?id=ngmEcEer8a) | LLM | `S` | - | Studies layer pruning and redundancy in deeper LLM layers. |
| `ICLR Workshop` | [2SSP: A Two-Stage Framework for Structured Pruning of LLMs](https://arxiv.org/abs/2501.17771) | LLM | `S` | [PyTorch[A]](https://github.com/FabrizioSandri/2SSP) | Combines width pruning and depth pruning for LLMs. |
| `OpenReview` | [MoreauPruner: Robust Structured Pruning of Large Language Models](https://openreview.net/forum?id=Y0qmwm6tgy) | LLM | `S` | - | Robustness-oriented structured pruning under weight perturbations. |
| `OpenReview` | [FASP: Fast and Accurate Structured Pruning of Large Language Models](https://openreview.net/forum?id=f4b0YVwKUO) | LLM | `S` | - | Fast structured pruning using interlinked sequential-layer structure. |
| `OpenReview` | [HESSO: Towards Automatic Efficient and User Friendly Structured Pruning](https://openreview.net/forum?id=LXlTdn9hY9) | General | `S` | - | OTO-style automatic structured pruning workflow. |
| `AAAI` | [Toward Adaptive Large Language Models Structured Pruning](https://ojs.aaai.org/index.php/AAAI/article/view/34078) | LLM | `S/D` | - | Multi-granularity adaptive structured pruning for LLMs. |
| `AAAI` | [PAT: Pruning-Aware Tuning for Large Language Models](https://github.com/kriskrisliu/PAT) | LLM | `S` | [PyTorch[A]](https://github.com/kriskrisliu/PAT) | Tunes LLMs while considering later pruning. |
| `AAAI` | [Pruning Large Language Models with Semi-Structural Adaptive Sparse Training](https://ojs.aaai.org/index.php/AAAI/article/view/34592) | LLM | `SS` | - | Semi-structured LLM pruning with adaptive sparse training; includes SLoRB. |
| `COLING` | [Enhancing One-Shot Pruned Pre-trained Language Models Through Sparse-Dense-Sparse Strategy](https://aclanthology.org/2025.coling-main.117/) | PLM | `U` | - | SDS: Sparse-Dense-Sparse restoration strategy after one-shot pruning. |
| `ACL Findings` | [Wanda++: Pruning Large Language Models via Regional Gradients](https://aclanthology.org/2025.findings-acl.224/) | LLM | `U/SS` | - | Adds efficient block/regional gradient information to Wanda-like pruning. |
| `EMNLP` | [On Pruning State-Space LLMs](https://aclanthology.org/2025.emnlp-main.950/) | SSM-LLM | `S/U` | - | Studies pruning behavior for state-space language models. |
| `BMVC` | [Explainability-Aware Structured Pruning for Efficient Neural Networks](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_850/paper.pdf) | CNN/ViT | `S` | - | Structured pruning guided by explainability. |
| `arXiv` | [NIRVANA: Structured Pruning Reimagined for Large Language Models](https://arxiv.org/abs/2509.14230) | LLM | `S` | - | Differentiates attention/MLP structure in LLM pruning. |
| `arXiv` | [Týr-the-Pruner: Structural Pruning LLMs via Global Sparsity Allocation Search](https://arxiv.org/abs/2503.09657) | LLM | `S` | - | Searches global sparsity allocation across layers. |
| `arXiv` | [Pruning Large Language Models by Identifying and Preserving Functional Networks](https://arxiv.org/abs/2508.05239) | LLM | `S` | [PyTorch[A]](https://github.com/WhatAboutMyStar/LLM_ACTIVATION) | Brain-network-inspired functional-neuron preservation. |
| `arXiv` | [Pruning Weights but Not Truth: Safeguarding Truthfulness While Pruning LLMs](https://arxiv.org/abs/2509.00096) | LLM Safety | `U/S` | - | Truthfulness-aware pruning using activation outlier signals. |
| `arXiv` | [PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models](https://openreview.net/forum?id=V13dSX1wAs) | LLM | `SS` | - | Channel permutation for improved N:M sparse pruning. |
| `arXiv` | [DSA: Discovering Sparsity Allocation for Layer-wise Pruning of Large Language Models](https://openreview.net/forum?id=rgtrYVC9n4) | LLM | `U/S` | - | Automated sparsity allocation for layer-wise LLM pruning. |
| `arXiv` | [OATS: Outlier-Aware Pruning Through Sparse and Low-Rank Decomposition](https://openreview.net/forum?id=DLDuVbxORA) | Transformer | `U/S` | - | Compresses transformer weights as sparse + low-rank components. |


**<h3 align="center">2024</h3>**

| Venue | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| `ICLR` | [Towards Meta-Pruning via Optimal Transport](https://openreview.net/forum?id=sMoifbuxjB) | General | `S` | [PyTorch[A]](https://github.com/alexandertheus/Intra-Fusion) | Structured pruning for hardware-friendly compression. |
| `ICLR` | [Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework](https://openreview.net/forum?id=eoSeaK4QJo) | SNN | `U` | [PyTorch[A]](https://github.com/xyshi2000/Unstructured-Pruning) | Unstructured sparsity / weight pruning method. |
| `ICLR` | [Masks, Signs, And Learning Rate Rewinding](https://openreview.net/forum?id=qODvxQ8TXW) | General | `S` | [PyTorch[A]](https://github.com/xyshi2000/Unstructured-Pruning) | Structured pruning for hardware-friendly compression. |
| `ICLR` | [Scaling Laws for Sparsely-Connected Foundation Models](https://openreview.net/forum?id=i9K2ZWkYIP) | Sparsity Theory | `S` | [PyTorch[A]](https://github.com/google-research/jaxpruner/tree/main/jaxpruner/projects/bigsparse) | Structured pruning for hardware-friendly compression. |
| `ICLR` | [Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging](https://openreview.net/forum?id=xx0ITyHp3u) | Sparsity Theory | `S` | - | Structured pruning for hardware-friendly compression. |
| `ICLR` | [Adaptive Sharpness-Aware Pruning for Robust Sparse Networks](https://openreview.net/forum?id=QFYVVwiAM8) | Sparsity Theory | `S` | - | Structured pruning for hardware-friendly compression. |
| `ICLR` | [What Makes a Good Prune? Maximal Unstructured Pruning for Maximal Cosine Similarity](https://openreview.net/forum?id=jsvvPVVzwf) | Sparsity Theory | `U` | [PyTorch[A]](https://github.com/gmw99/what_makes_a_good_prune) | Unstructured sparsity / weight pruning method. |
| `ICLR` | [In defense of parameter sharing for model-compression](https://openreview.net/forum?id=ypAT2ixD4X) | General | `S/U` | - | - |
| `ICLR` | [ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models](https://openreview.net/forum?id=iIT02bAKzv) | LLM | `U` | - | Unstructured sparsity / weight pruning method. |
| `ICLR` | [Data-independent Module-aware Pruning for Hierarchical Vision Transformers](https://openreview.net/forum?id=7Ol6foUi1G) | ViT/Transformer | `S` | [PyTorch[A]](https://github.com/he-y/Data-independent-Module-Aware-Pruning) | Structured pruning for hardware-friendly compression. |
| `ICLR` | [SWAP: Sparse Entropic Wasserstein Regression for Robust Network Pruning](https://openreview.net/forum?id=LJWizuuBUy) | Sparsity Theory | `S` | - | Structured pruning for hardware-friendly compression. |
| `ICLR` | [Sparse Weight Averaging with Multiple Particles for Iterative Magnitude Pruning](https://openreview.net/forum?id=Y9t7MqZtCR) | Sparsity Theory | `U` | - | Unstructured sparsity / weight pruning method. |
| `ICLR` | [Synergistic Patch Pruning for Vision Transformer: Unifying Intra- & Inter-Layer Patch Importance](https://openreview.net/forum?id=COO51g41Q4) | ViT/Transformer | `S` | - | Token/patch reduction for transformer acceleration. |
| `ICLR` | [FedP3: Federated Personalized and Privacy-friendly Network Pruning under Model Heterogeneity](https://openreview.net/forum?id=hbHwZYqk9T) | Federated/Privacy | `S` | - | Pruning method for federated or distributed settings. |
| `ICLR` | [The Need for Speed: Pruning Transformers with One Recipe](https://openreview.net/forum?id=MVmT6uQ3cQ) | ViT/Transformer | `S` | [PyTorch[A]](https://github.com/Skhaki18/optin-transformer-pruning) | Structured pruning for hardware-friendly compression. |
| `ICLR` | [SAS: Structured Activation Sparsification](https://openreview.net/forum?id=vZfi5to2Xl) | General | `S` | [PyTorch[A]](https://github.com/DensoITLab/sas_) | Structured pruning for hardware-friendly compression. |
| `CVPR` | [OrthCaps: An Orthogonal CapsNet with Sparse Attention Routing and Pruning](https://arxiv.org/abs/2403.13351) | Sparsity Theory | `S` | [PyTorch[A]](https://github.com/ornamentt/OrthCap) | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Zero-TPrune: Zero-Shot Token Pruning through Leveraging of the Attention Graph in Pre-Trained Transformers](https://arxiv.org/abs/2305.17328) | ViT/Transformer | `S` | [PyTorch[A]](https://jha-lab.github.io/zerotprune/) | Token/patch reduction for transformer acceleration. |
| `CVPR` | [Finding Lottery Tickets in Vision Models via Data-driven Spectral Foresight Pruning](https://github.com/iurada/px-ntk-pruning) | Sparsity Theory | `S` | [PyTorch[A]](https://github.com/iurada/px-ntk-pruning) | Structured pruning for hardware-friendly compression. |
| `CVPR` | [BilevelPruning: Unified Dynamic and Static Channel Pruning for Convolutional Neural Networks](https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_BilevelPruning_Unified_Dynamic_and_Static_Channel_Pruning_for_Convolutional_Neural_CVPR_2024_paper.pdf) | CNN | `S` | - | Input-adaptive or dynamic pruning strategy. |
| `CVPR` | [FedMef: Towards Memory-efficient Federated Dynamic Pruning](https://arxiv.org/pdf/2403.14737.pdf) | Federated/Privacy | `S` | - | Pruning method for federated or distributed settings. |
| `CVPR` | Resource-Efficient Transformer Pruning for Finetuning of Large Models | ViT/Transformer | `S` | - | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Device-Wise Federated Network Pruning](https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_Device-Wise_Federated_Network_Pruning_CVPR_2024_paper.pdf) | Federated/Privacy | `S` | - | Pruning method for federated or distributed settings. |
| `CVPR` | [Auto-Train-Once: Controller Network Guided Automatic Network Pruning from Scratch](https://arxiv.org/abs/2403.14729) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Jointly Training and Pruning CNNs via Learnable Agent Guidance and Alignment](https://arxiv.org/abs/2403.14729) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Diversity-aware Channel Pruning for StyleGAN Compression](https://arxiv.org/abs/2403.13548) | GAN | `S` | [PyTorch[A]](https://jiwoogit.github.io/DCP-GAN_site/) | Structured pruning for hardware-friendly compression. |
| `CVPR` | [MADTP: Multimodal Alignment-Guided Dynamic Token Pruning for Accelerating Vision-Language Transformer](https://arxiv.org/abs/2403.02991) | VLM/MLLM | `S` | [PyTorch[A]](https://github.com/double125/MADTP) | Recent method for pruning or sparsifying large language models. |
| `AAAI` | [Dynamic Feature Pruning and Consolidation for Occluded Person Re-Identification](https://arxiv.org/abs/2211.14742) | General | `S` | - | Input-adaptive or dynamic pruning strategy. |
| `AAAI` | [REPrune: Channel Pruning via Kernel Representative Selection](https://arxiv.org/abs/2211.14742) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| `AAAI` | [Revisiting Gradient Pruning: A Dual Realization for Defending against Gradient Attacks](https://arxiv.org/abs/2401.16687) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| `AAAI` | [IRPruneDet: Efficient Infrared Small Target Detection via Wavelet Structure-Regularized Soft Channel Pruning](https://ojs.aaai.org/index.php/AAAI/article/view/28551) | Detection/Segmentation | `S` | - | Structured pruning for hardware-friendly compression. |
| `AAAI` | [EPSD: Early Pruning with Self-Distillation for Efficient Model Compression](https://arxiv.org/abs/2402.00084) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| `WACV` | [Pruning from Scratch via Shared Pruning Module and Nuclear norm-based Regularization](https://openaccess.thecvf.com/content/WACV2024/papers/Lee_Pruning_From_Scratch_via_Shared_Pruning_Module_and_Nuclear_Norm-Based_WACV_2024_paper.pdf) | General | `S` | [PyTorch[A]](https://github.com/jsleeg98/NuSPM) | Structured pruning for hardware-friendly compression. |
| `WACV` | [Towards Better Structured Pruning Saliency by Reorganizing Convolution](https://openaccess.thecvf.com/content/WACV2024/papers/Sun_Towards_Better_Structured_Pruning_Saliency_by_Reorganizing_Convolution_WACV_2024_paper.pdf) | GAN | `S` | [PyTorch[A]](https://github.com/AlexSunNik/SPSRC) | Structured pruning for hardware-friendly compression. |
| `WACV` | [Torque based Structured Pruning for Deep Neural Network](https://openaccess.thecvf.com/content/WACV2024/papers/Gupta_Torque_Based_Structured_Pruning_for_Deep_Neural_Network_WACV_2024_paper.pdf) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| `WACV` | [Revisiting Token Pruning for Object Detection and Instance Segmentation](https://openaccess.thecvf.com/content/WACV2024/html/Liu_Revisiting_Token_Pruning_for_Object_Detection_and_Instance_Segmentation_WACV_2024_paper.html) | ViT/Transformer | `S` | [PyTorch[A]](https://github.com/uzh-rpg/svit/) | Token/patch reduction for transformer acceleration. |
| `WACV` | [Token Fusion: Bridging the Gap Between Token Pruning and Token Merging](https://openaccess.thecvf.com/content/WACV2024/html/Kim_Token_Fusion_Bridging_the_Gap_Between_Token_Pruning_and_Token_WACV_2024_paper.html) | ViT/Transformer | `S` | - | Token/patch reduction for transformer acceleration. |
| `WACV` | [PATROL: Privacy-Oriented Pruning for Collaborative Inference Against Model Inversion Attacks](https://openaccess.thecvf.com/content/WACV2024/html/Ding_PATROL_Privacy-Oriented_Pruning_for_Collaborative_Inference_Against_Model_Inversion_Attacks_WACV_2024_paper.html) | Federated/Privacy | `S` | - | Structured pruning for hardware-friendly compression. |
| `ICLR` | [Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning](https://openreview.net/forum?id=09iOdaeOzp) | LLM | `S` | [PyTorch[A]](https://github.com/princeton-nlp/LLM-Shearing) | Structured pruning + continued pretraining for smaller LLMs. |
| `ICLR` | [Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models](https://openreview.net/forum?id=Tr0lPx9woF) | LLM | `U` | - | RIA + channel permutation for post-training LLM pruning and N:M semi-structured sparsity. |
| `ACL Findings` | [LoRAPrune: Structured Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning](https://aclanthology.org/2024.findings-acl.178/) | LLM | `S` | [PyTorch[A]](https://github.com/aim-uofa/LoRAPrune) | LoRA-guided structured pruning with reduced memory cost. |
| `ECCV` | [FastV: An Image is Worth 1/2 Tokens After Layer 2](https://arxiv.org/abs/2403.06764) | LVLM | `T` | [PyTorch[A]](https://github.com/pkunlp-icler/FastV) | Plug-and-play visual-token pruning for LMM inference acceleration. |


**<h3 align="center">2023</h3>**

| Venue | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| `NIPS` | [Diff-Pruning: Structural Pruning for Diffusion Models](https://arxiv.org/abs/2305.10924) | Diffusion | `S` | [PyTorch[A]](https://github.com/VainF/Diff-Pruning) | Structured pruning for hardware-friendly compression. |
| `NIPS` | [LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627) | LLM | `S` | [PyTorch[A]](https://github.com/horseee/LLM-Pruner) | Recent method for pruning or sparsifying large language models. |
| `ICCV` | [Automatic Network Pruning via Hilbert-Schmidt Independence Criterion Lasso under Information Bottleneck Principle](https://openaccess.thecvf.com/content/ICCV2023/html/Guo_Automatic_Network_Pruning_via_Hilbert-Schmidt_Independence_Criterion_Lasso_under_Information_ICCV_2023_paper.html) | General | `S` | [PyTorch[A]](https://github.com/sunggo/APIB) | Structured pruning for hardware-friendly compression. |
| `ICCV` | [Unified Data-Free Compression: Pruning and Quantization without Fine-Tuning](https://openaccess.thecvf.com/content/ICCV2023/html/Bai_Unified_Data-Free_Compression_Pruning_and_Quantization_without_Fine-Tuning_ICCV_2023_paper.html) | General | `S` | [PyTorch[A]](https://github.com/Dtudy/UDFC) | Combines pruning with quantization/compression. |
| `ICCV` | [Structural Alignment for Network Pruning through Partial Regularization](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_Structural_Alignment_for_Network_Pruning_through_Partial_Regularization_ICCV_2023_paper.html) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| `ICCV` | [Differentiable Transportation Pruning](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Differentiable_Transportation_Pruning_ICCV_2023_paper.html) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| `ICCV` | [Dynamic Token Pruning in Plain Vision Transformers for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2023/html/Tang_Dynamic_Token_Pruning_in_Plain_Vision_Transformers_for_Semantic_Segmentation_ICCV_2023_paper.html) | ViT/Transformer | `S` | [PyTorch[A]](https://github.com/zbwxp/Dynamic-Token-Pruning) | Token/patch reduction for transformer acceleration. |
| `ICCV` | [Towards Fairness-aware Adversarial Network Pruning](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Towards_Fairness-aware_Adversarial_Network_Pruning_ICCV_2023_paper.html) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| `ICCV` | [Efficient Joint Optimization of Layer-Adaptive Weight Pruning in Deep Neural Networks](https://openaccess.thecvf.com/content/ICCV2023/html/Xu_Efficient_Joint_Optimization_of_Layer-Adaptive_Weight_Pruning_in_Deep_Neural_ICCV_2023_paper.html) | General | `S` | [PyTorch[A]](https://github.com/Akimoto-Cris/RD_VIT_PRUNE) | Structured pruning for hardware-friendly compression. |
| `CVPR` | [DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900) | General | `S` | [PyTorch[A]](https://github.com/VainF/Torch-Pruning) | Structured pruning for hardware-friendly compression. |
| `CVPR` | [X-Pruner: eXplainable Pruning for Vision Transformers](https://arxiv.org/abs/2303.04935) | ViT/Transformer | `U/S` | - | - |
| `CVPR` | [Joint Token Pruning and Squeezing Towards More Aggressive Compression of Vision Transformers](https://arxiv.org/abs/2304.10716) | ViT/Transformer | `S` | [PyTorch[A]](https://github.com/megvii-research/TPS-CVPR2023) | Token/patch reduction for transformer acceleration. |
| `CVPR` | [Global Vision Transformer Pruning with Hessian-Aware Saliency](https://arxiv.org/abs/2110.04869) | ViT/Transformer | `S` | - | Structured pruning for hardware-friendly compression. |
| `CVPR` | [CP3: Channel Pruning Plug-in for Point-based Networks](https://arxiv.org/abs/2303.13097) | 3D/Point Cloud | `S` | - | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Training Debiased Subnetworks With Contrastive Weight Pruning](https://openaccess.thecvf.com/content/CVPR2023/html/Park_Training_Debiased_Subnetworks_With_Contrastive_Weight_Pruning_CVPR_2023_paper.html) | General | `U` | - | Unstructured sparsity / weight pruning method. |
| `CVPR` | [Pruning Parameterization With Bi-Level Optimization for Efficient Semantic Segmentation on the Edge](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Pruning_Parameterization_With_Bi-Level_Optimization_for_Efficient_Semantic_Segmentation_on_CVPR_2023_paper.html) | Detection/Segmentation | `S` | - | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Structural Alignment for Network Pruning through Partial Regularization](https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_Structural_Alignment_for_Network_Pruning_through_Partial_Regularization_ICCV_2023_paper.pdf) | General | `S` | [PyTorch[A]](https://github.com/xidongwu/AutoTrainOnce) | Structured pruning for hardware-friendly compression. |
| `ICLR` | [JaxPruner: A concise library for sparsity research](https://arxiv.org/abs/2304.14082) | Sparsity Theory | `U/S` | [PyTorch[A]](https://github.com/google-research/jaxpruner) | - |
| `ICLR` | [OTOv2: Automatic, Generic, User-Friendly](https://openreview.net/forum?id=7ynoX1ojPMt) | General | `S` | [PyTorch[A]](https://github.com/tianyic/only_train_once) | Structured pruning for hardware-friendly compression. |
| `ICLR` | [How I Learned to Stop Worrying and Love Retraining](https://openreview.net/forum?id=_nF5imFKQI) | General | `U` | [PyTorch[A]](https://github.com/ZIB-IOL/BIMP) | Unstructured sparsity / weight pruning method. |
| `ICLR` | [Token Merging: Your ViT But Faster ](https://openreview.net/forum?id=JroZRaRw7Eu) | ViT/Transformer | `U/S` | [PyTorch[A]](https://github.com/facebookresearch/ToMe) | Token/patch reduction for transformer acceleration. |
| `ICLR` | [Revisiting Pruning at Initialization Through the Lens of Ramanujan Graphs](https://openreview.net/forum?id=uVcDssQff_) | Sparsity Theory | `U` | [PyTorch[A]](https://github.com/VITA-Group/ramanujan-on-pai) (soon...) | Unstructured sparsity / weight pruning method. |
| `ICLR` | [Unmasking the Lottery Ticket Hypothesis: What's Encoded in a Winning Ticket's Mask?](https://openreview.net/forum?id=xSsW2Am-ukZ) | Sparsity Theory | `U` | - | Unstructured sparsity / weight pruning method. |
| `ICLR` | [NTK-SAP: Improving neural network pruning by aligning training dynamics](https://openreview.net/forum?id=-5EWhW_4qWP) | General | `U` | - | Unstructured sparsity / weight pruning method. |
| `ICLR` | [DFPC: Data flow driven pruning of coupled channels without data](https://openreview.net/forum?id=mhnHqRqcjYU) | CNN | `S` | [PyTorch[A]](https://github.com/TanayNarshana/DFPC-Pruning) | Data-free or calibration-light pruning method. |
| `ICLR` | [TVSPrune - Pruning Non-discriminative filters via Total Variation separability of intermediate representations without fine tuning](https://openreview.net/forum?id=sZI1Oj9KBKy) | CNN | `S` | [PyTorch[A]](https://github.com/chaimurti/TVSPrune) | Structured pruning for hardware-friendly compression. |
| `ICLR` | [Pruning Deep Neural Networks from a Sparsity Perspective](https://openreview.net/forum?id=i-DleYh34BM) | Sparsity Theory | `U` | [PyTorch[A]](https://openreview.net/attachment?id=i-DleYh34BM&name=supplementary_material) | Unstructured sparsity / weight pruning method. |
| `ICLR` | [A Unified Framework of Soft Threshold Pruning](https://openreview.net/forum?id=cCFqcrq0d8) | General | `U` | [PyTorch[A]](https://openreview.net/attachment?id=cCFqcrq0d8&name=supplementary_material) | Unstructured sparsity / weight pruning method. |
| `WACV` | [Calibrating Deep Neural Networks Using Explicit Regularisation and Dynamic Data Pruning](https://openaccess.thecvf.com/content/WACV2023/html/Patra_Calibrating_Deep_Neural_Networks_Using_Explicit_Regularisation_and_Dynamic_Data_WACV_2023_paper.html) | General | `S` | - | Input-adaptive or dynamic pruning strategy. |
| `WACV` | [Attend Who Is Weak: Pruning-Assisted Medical Image Localization Under Sophisticated and Implicit Imbalances](https://openaccess.thecvf.com/content/WACV2023/html/Jaiswal_Attend_Who_Is_Weak_Pruning-Assisted_Medical_Image_Localization_Under_Sophisticated_WACV_2023_paper.html) | Medical Imaging | `S` | - | Structured pruning for hardware-friendly compression. |
| [`ICASSP`](https://2023.ieeeicassp.org/important-dates/) | [WHC: Weighted Hybrid Criterion for Filter Pruning on Convolutional Neural Networks](https://arxiv.org/abs/2302.08185) | CNN | `S` | [PyTorch[A]](https://github.com/ShaowuChen/WHC) | Structured pruning for hardware-friendly compression. |
| `ICML` | [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://proceedings.mlr.press/v202/frantar23a.html) | LLM | `U/SS` | [PyTorch[A]](https://github.com/IST-DASLab/sparsegpt) | First widely used one-shot pruning method for 10B–100B+ parameter GPT-family models. |
| `arXiv` | [Wanda: A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695) | LLM | `U/SS` | [PyTorch[A]](https://github.com/locuslab/wanda) | Weight magnitude × activation norm; no retraining or weight update. |
| `arXiv` | [LoRAPrune: Structured Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2305.18403) | LLM | `S` | [PyTorch[A]](https://github.com/aim-uofa/LoRAPrune) | Early version later appearing in ACL Findings 2024. |

<h3 align="center">2022</h3>

| Venue | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| `CVPR` | [Interspace Pruning: Using Adaptive Filter Representations To Improve Training of Sparse CNNs](https://openaccess.thecvf.com/content/CVPR2022/html/Wimmer_Interspace_Pruning_Using_Adaptive_Filter_Representations_To_Improve_Training_of_CVPR_2022_paper.html) | Sparsity Theory | `U` | - | Unstructured sparsity / weight pruning method. |
| `CVPR` | [Revisiting Random Channel Pruning for Neural Network Compression](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Revisiting_Random_Channel_Pruning_for_Neural_Network_Compression_CVPR_2022_paper.html) | CNN | `S` | [PyTorch[A]](https://github.com/ofsoundof/random_channel_pruning) (soon...) | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Fire Together Wire Together: A Dynamic Pruning Approach With Self-Supervised Mask Prediction](https://openaccess.thecvf.com/content/CVPR2022/html/Elkerdawy_Fire_Together_Wire_Together_A_Dynamic_Pruning_Approach_With_Self-Supervised_CVPR_2022_paper.html) | General | `S` | [PyTorch[A]](https://github.com/selkerdawy/FTWT) | Input-adaptive or dynamic pruning strategy. |
| `CVPR` | [When to Prune? A Policy towards Early Structural Pruning](https://openaccess.thecvf.com/content/CVPR2022/html/Shen_When_To_Prune_A_Policy_Towards_Early_Structural_Pruning_CVPR_2022_paper.html) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Dreaming to Prune Image Deraining Networks](https://openaccess.thecvf.com/content/CVPR2022/papers/Zou_Dreaming_To_Prune_Image_Deraining_Networks_CVPR_2022_paper.pdf) | Low-level Vision | `S` | - | Structured pruning for hardware-friendly compression. |
| `ICLR` | [SOSP: Efficiently Capturing Global Correlations by Second-Order Structured Pruning](https://openreview.net/forum?id=t5EmXZ3ZLR) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| `ICLR` | [Learning Pruning-Friendly Networks via Frank-Wolfe: One-Shot, Any-Sparsity, And No Retraining](https://openreview.net/forum?id=O1DEtITim__) | Sparsity Theory | `U` | [PyTorch[A]](https://github.com/VITA-Group/SFW-Once-for-All-Pruning) | Unstructured sparsity / weight pruning method. |
| `ICLR` | [Revisit Kernel Pruning with Lottery Regulated Grouped Convolutions](https://openreview.net/forum?id=LdEhiMG9WLO) | Sparsity Theory | `S` | [PyTorch[A]](https://github.com/choH/lottery_regulated_grouped_kernel_pruning) | Structured pruning for hardware-friendly compression. |
| `ICLR` | [Dual Lottery Ticket Hypothesis](https://openreview.net/forum?id=fOsN52jn25l) | Sparsity Theory | `U` | [PyTorch[A]](https://github.com/yueb17/DLTH) | Unstructured sparsity / weight pruning method. |
| `NIPS` | [SAViT: Structure-Aware Vision Transformer Pruning via Collaborative Optimization](https://openreview.net/forum?id=w5DacXWzQ-Q) | ViT/Transformer | `S` | [PyTorch[A]](https://github.com/hikvision-research/SAViT)(soon...) | Structured pruning for hardware-friendly compression. |
| `NIPS` | [Structural Pruning via Latency-Saliency Knapsack](https://openreview.net/forum?id=cUOR-_VsavA) | Deployment | `S` | [PyTorch[A]](https://github.com/NVlabs/HALP) | Structured pruning for hardware-friendly compression. |
| `ACCV` | [Filter Pruning via Automatic Pruning Rate Search⋆](https://openaccess.thecvf.com/content/ACCV2022/html/Sun_Filter_Pruning_via_Automatic_Pruning_Rate_Search_ACCV_2022_paper.html) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| `ACCV` | [Network Pruning via Feature Shift Minimization](https://openaccess.thecvf.com/content/ACCV2022/html/Duan_Network_Pruning_via_Feature_Shift_Minimization_ACCV_2022_paper.html) | General | `S` | [PyTorch[A]](https://github.com/lscgx/FSM) | Structured pruning for hardware-friendly compression. |
| `ACCV` | [Lightweight Alpha Matting Network Using Distillation-Based Channel Pruning](https://openaccess.thecvf.com/content/ACCV2022/html/Yoon_Lightweight_Alpha_Matting_Network_Using_Distillation-Based_Channel_Pruning_ACCV_2022_paper.html) | Low-level Vision | `S` | [PyTorch[A]](https://github.com/DongGeun-Yoon/DCP) | Structured pruning for hardware-friendly compression. |
| `ACCV` | [Adaptive FSP : Adaptive Architecture Search with Filter Shape Pruning](https://openaccess.thecvf.com/content/ACCV2022/html/Kim_Adaptive_FSP__Adaptive_Architecture_Search_with_Filter_Shape_Pruning_ACCV_2022_paper.html) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| `ECCV` | [Soft Masking for Cost-Constrained Channel Pruning](https://link.springer.com/chapter/10.1007/978-3-031-20083-0_38) | CNN | `S` | [PyTorch[A]](https://github.com/NVlabs/SMCP) | Structured pruning for hardware-friendly compression. |
| `WACV` | [Hessian-Aware Pruning and Optimal Neural Implant](https://openaccess.thecvf.com/content/WACV2022/papers/Yu_Hessian-Aware_Pruning_and_Optimal_Neural_Implant_WACV_2022_paper.pdf) | General | `S` | [PyTorch[A]](https://github.com/yaozhewei/HAP) | Structured pruning for hardware-friendly compression. |
| `WACV` | [PPCD-GAN: Progressive Pruning and Class-Aware Distillation for Large-Scale Conditional GANs Compression](https://openaccess.thecvf.com/content/WACV2022/papers/Vo_PPCD-GAN_Progressive_Pruning_and_Class-Aware_Distillation_for_Large-Scale_Conditional_GANs_WACV_2022_paper.pdf) | GAN | `S` | - | Structured pruning for hardware-friendly compression. |
| `WACV` | [Channel Pruning via Lookahead Search Guided Reinforcement Learning](https://openaccess.thecvf.com/content/WACV2022/papers/Wang_Channel_Pruning_via_Lookahead_Search_Guided_Reinforcement_Learning_WACV_2022_paper.pdf) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| `WACV` | [EZCrop: Energy-Zoned Channels for Robust Output Pruning](https://openaccess.thecvf.com/content/WACV2022/papers/Lin_EZCrop_Energy-Zoned_Channels_for_Robust_Output_Pruning_WACV_2022_paper.pdf) | Deployment | `S` | [PyTorch[A]](https://github.com/rlin27/EZCrop) | Structured pruning for hardware-friendly compression. |
| `ICIP` | [One-Cycle Pruning: Pruning Convnets With Tight Training Budget](https://ieeexplore.ieee.org/document/9897980) | CNN | `U` | - | Unstructured sparsity / weight pruning method. |
| `ICIP` | [RAPID: A Single Stage Pruning Framework](https://ieeexplore.ieee.org/document/9898000) | General | `U` | - | Unstructured sparsity / weight pruning method. |
| `ICIP` | [The Rise of the Lottery Heroes: Why Zero-Shot Pruning is Hard](https://ieeexplore.ieee.org/document/9897223) | Sparsity Theory | `U` | - | Unstructured sparsity / weight pruning method. |
| `ICIP` | [Truncated Lottery Ticket for Deep Pruning](https://ieeexplore.ieee.org/document/9897767) | Sparsity Theory | `U` | - | Unstructured sparsity / weight pruning method. |
| `ICIP` | [Which Metrics For Network Pruning: Final Accuracy? or Accuracy Drop?](https://ieeexplore.ieee.org/document/9898051) | General | `S/U` | - | - |
| `ISMSI` | [Structured Pruning with Automatic Pruning Rate Derivation for Image Processing Neural Networks](https://dl.acm.org/doi/abs/10.1145/3533050.3533066) | Low-level Vision | `S` | - | Structured pruning for hardware-friendly compression. |

<h3 align="center">2021</h3>

| Venue | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| `ICLR` | [Neural Pruning via Growing Regularization](https://openreview.net/forum?id=o966_Is_nPA) | General | `S` | [PyTorch[A]](https://github.com/mingsun-tse/regularization-pruning) | Structured pruning for hardware-friendly compression. |
| `ICLR` | [Network Pruning That Matters: A Case Study on Retraining Variants](https://openreview.net/forum?id=Cb54AMqHQFP) | General | `S` | [PyTorch[A]](https://github.com/lehduong/NPTM) | Structured pruning for hardware-friendly compression. |
| `ICLR` | [Layer-adaptive Sparsity for the Magnitude-based Pruning](https://openreview.net/forum?id=H6ATjJ0TKdf) | Sparsity Theory | `U` | [PyTorch[A]](https://github.com/jaeho-lee/layer-adaptive-sparsity) | Unstructured sparsity / weight pruning method. |
| `NIPS` | [Only Train Once: A One-Shot Neural Network Training And Pruning Framework](https://openreview.net/forum?id=p5rMPjrcCZq) | General | `S` | [PyTorch[A]](https://github.com/tianyic/only_train_once) | Structured pruning for hardware-friendly compression. |
| `CVPR` | [NPAS: A Compiler-Aware Framework of Unified Network Pruning and Architecture Search for Beyond Real-Time Mobile Acceleration](https://openaccess.thecvf.com/content/CVPR2021/html/Li_NPAS_A_Compiler-Aware_Framework_of_Unified_Network_Pruning_and_Architecture_CVPR_2021_paper.html) | Deployment | `S` | - | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Network Pruning via Performance Maximization](https://openaccess.thecvf.com/content/CVPR2021/html/Gao_Network_Pruning_via_Performance_Maximization_CVPR_2021_paper.html) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Convolutional Neural Network Pruning With Structural Redundancy Reduction*](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Convolutional_Neural_Network_Pruning_With_Structural_Redundancy_Reduction_CVPR_2021_paper.html) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Manifold Regularized Dynamic Network Pruning](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Manifold_Regularized_Dynamic_Network_Pruning_CVPR_2021_paper.html) | General | `S` | [PyTorch[A]](https://github.com/yehuitang/Pruning/tree/master/ManiDP) | Input-adaptive or dynamic pruning strategy. |
| `CVPR` | [Joint-DetNAS: Upgrade Your Detector With NAS, Pruning and Dynamic Distillation](https://openaccess.thecvf.com/content/CVPR2021/html/Yao_Joint-DetNAS_Upgrade_Your_Detector_With_NAS_Pruning_and_Dynamic_Distillation_CVPR_2021_paper.html) | General | `S` | - | Input-adaptive or dynamic pruning strategy. |
| `ICCV` | [ResRep: Lossless CNN Pruning via Decoupling Remembering and Forgetting](https://openaccess.thecvf.com/content/ICCV2021/html/Ding_ResRep_Lossless_CNN_Pruning_via_Decoupling_Remembering_and_Forgetting_ICCV_2021_paper.html) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| `ICCV` | [Achieving On-Mobile Real-Time Super-Resolution With Neural Architecture and Pruning Search](https://openaccess.thecvf.com/content/ICCV2021/html/Zhan_Achieving_On-Mobile_Real-Time_Super-Resolution_With_Neural_Architecture_and_Pruning_Search_ICCV_2021_paper.html) | Low-level Vision | `S` | - | Structured pruning for hardware-friendly compression. |
| `ICCV` | [GDP: Stabilized Neural Network Pruning via Gates With Differentiable Polarization*](https://openaccess.thecvf.com/content/ICCV2021/html/Guo_GDP_Stabilized_Neural_Network_Pruning_via_Gates_With_Differentiable_Polarization_ICCV_2021_paper.html) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| `WACV` | [Holistic Filter Pruning for Efficient Deep Neural Networks](https://openaccess.thecvf.com/content/WACV2021/html/Enderich_Holistic_Filter_Pruning_for_Efficient_Deep_Neural_Networks_WACV_2021_paper.html) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| `ICML` | [Accelerate CNNs from Three Dimensions: A Comprehensive Pruning Framework](https://icml.cc/virtual/2021/poster/9081) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| `ICML` | [Group Fisher Pruning for Practical Network Compression](https://icml.cc/virtual/2021/poster/9875) | General | `S` | [PyTorch[A]](https://github.com/jshilong/FisherPruning) | Structured pruning for hardware-friendly compression. |

<h3 align="center">2020</h3>

| Venue | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| `CVPR` | [HRank: Filter Pruning using High-Rank Feature Map](https://openaccess.thecvf.com/content_CVPR_2020/html/Lin_HRank_Filter_Pruning_Using_High-Rank_Feature_Map_CVPR_2020_paper.html) | CNN | `S` | [PyTorch[A]](https://github.com/lmbxmu/HRank) | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Towards efficient model compression via learned global ranking](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chin_Towards_Efficient_Model_Compression_via_Learned_Global_Ranking_CVPR_2020_paper.pdf) | General | `S` | [PyTorch[A]](https://github.com/enyac-group/LeGR) | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Learning Filter Pruning Criteria for Deep Convolutional Neural Networks Acceleration](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Learning_Filter_Pruning_Criteria_for_Deep_Convolutional_Neural_Networks_Acceleration_CVPR_2020_paper.html) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Group_Sparsity_The_Hinge_Between_Filter_Pruning_and_Decomposition_for_CVPR_2020_paper.html) | Sparsity Theory | `S` | [PyTorch[A]](https://github.com/ofsoundof/group_sparsity) | Structured pruning for hardware-friendly compression. |
| `CVPR` | [APQ: Joint Search for Network Architecture, Pruning and Quantization Policy](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_APQ_Joint_Search_for_Network_Architecture_Pruning_and_Quantization_Policy_CVPR_2020_paper.html) | General | `S` | [PyTorch[A]](https://github.com/mit-han-lab/apq) | Combines pruning with quantization/compression. |
| `ICLR` | [Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints](https://openreview.net/forum?id=HyxLRTVKPH) | General | `U` | - | Unstructured sparsity / weight pruning method. |
| `MLSys` | [Shrinkbench: What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033) | General | - | [PyTorch[A]](https://github.com/JJGO/shrinkbench) | - |
| `BMBS` | [Similarity Based Filter Pruning for Efficient Super-Resolution Models](https://ieeexplore.ieee.org/abstract/document/9379712) | Low-level Vision | `S` | - | Structured pruning for hardware-friendly compression. |

<h3 align="center">2019</h3>

| Venue | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| `CVPR` | [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.html) | CNN | `S` | [PyTorch[A]](https://github.com/he-y/filter-pruning-geometric-median) | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Variational Convolutional Neural Network Pruning](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Variational_Convolutional_Neural_Network_Pruning_CVPR_2019_paper.html) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Towards Optimal Structured CNN Pruning via Generative Adversarial Learning](https://openaccess.thecvf.com/content_CVPR_2019/html/Lin_Towards_Optimal_Structured_CNN_Pruning_via_Generative_Adversarial_Learning_CVPR_2019_paper.html) | CNN | `S` | [PyTorch[A]](https://github.com/ShaohuiLin/GAL) | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Partial Order Pruning: For Best Speed/Accuracy Trade-Off in Neural Architecture Search](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Partial_Order_Pruning_For_Best_SpeedAccuracy_Trade-Off_in_Neural_Architecture_CVPR_2019_paper.html) | General | `S` | [PyTorch[A]](https://github.com/lixincn2015/Partial-Order-Pruning) | Structured pruning for hardware-friendly compression. |
| `CVPR` | [Importance Estimation for Neural Network Pruning](https://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html) | General | `S` | [PyTorch[A]](https://github.com/NVlabs/Taylor_pruning) | Structured pruning for hardware-friendly compression. |
| `ICLR` | [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://openreview.net/forum?id=rJl-b3RcF7) | Sparsity Theory | `U` | [PyTorch[A]](https://github.com/facebookresearch/open_lth) | Unstructured sparsity / weight pruning method. |
| `ICLR` | [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://openreview.net/forum?id=B1VZqjAcYX) | ViT/Transformer | `U` | [Tensorflow[A]](https://github.com/namhoonlee/snip-public) | Unstructured sparsity / weight pruning method. |
| `ICCV` | [MetaPruning: Meta-Learning for Automatic Neural Network Channel Pruning](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_MetaPruning_Meta_Learning_for_Automatic_Neural_Network_Channel_Pruning_ICCV_2019_paper.html) | CNN | `S` | [PyTorch[A]](https://github.com/liuzechun/MetaPruning) | Structured pruning for hardware-friendly compression. |
| `ICCV` | [Accelerate CNN via Recursive Bayesian Pruning](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Accelerate_CNN_via_Recursive_Bayesian_Pruning_ICCV_2019_paper.html) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |

<h3 align="center">2018</h3>

| Venue | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| `CVPR` | [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://openaccess.thecvf.com/content_cvpr_2018/html/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.html) | General | `S` | [PyTorch[A]](https://github.com/arunmallya/packnet) | Structured pruning for hardware-friendly compression. |
| `CVPR` | [NISP: Pruning Networks Using Neuron Importance Score Propagation](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_NISP_Pruning_Networks_CVPR_2018_paper.html) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| `ICIP` | [Online Filter Clustering and Pruning for Efficient Convnets](https://ieeexplore.ieee.org/abstract/document/8451123) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| `IJCAI` | [Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://www.ijcai.org/proceedings/2018/0309.pdf) | CNN | `S` | [PyTorch[A]](https://github.com/he-y/soft-filter-pruning) | Structured pruning for hardware-friendly compression. |

<h3 align="center">2017</h3>

| Venue | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| `CVPR` | [Designing Energy-Efficient Convolutional Neural Networks Using Energy-Aware Pruning](https://openaccess.thecvf.com/content_cvpr_2017/html/Yang_Designing_Energy-Efficient_Convolutional_CVPR_2017_paper.html) | Deployment | `S` | - | Structured pruning for hardware-friendly compression. |
| `ICLR` | [Pruning Filters for Efficient ConvNets](https://openreview.net/forum?id=rJqFGTslg) | CNN | `S` | [PyTorch[O]](doc/PFEC.md) | Structured pruning for hardware-friendly compression. |
| `ICCV` | [Channel Pruning for Accelerating Very Deep Neural Networks](https://openaccess.thecvf.com/content_iccv_2017/html/He_Channel_Pruning_for_ICCV_2017_paper.html) | CNN | `S` | [PyTorch[A]](https://github.com/yihui-he/channel-pruning) | Structured pruning for hardware-friendly compression. |
| `ICCV` | [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://openaccess.thecvf.com/content_iccv_2017/html/Luo_ThiNet_A_Filter_ICCV_2017_paper.html) | CNN | `S` | [Caffe[A]](https://github.com/Roll920/ThiNet) | Structured pruning for hardware-friendly compression. |
| `ICCV` | [Learning Efficient Convolutional Networks Through Network Slimming](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) | CNN | `S` | [PyTorch[A]](https://github.com/Eric-mingjie/network-slimming) | Structured pruning for hardware-friendly compression. |


## Journal Publications 

<h3 align="center">2026</h3>

| Journal | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| - | - | General | - | - | No peer-reviewed journal pruning papers were confidently added for 2026 yet; most 2026 additions are conference or preprint entries above. |

<h3 align="center">2025</h3>

| Journal | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| [Neural Networks](https://www.sciencedirect.com/journal/neural-networks) | [OOPS: Outlier-aware and Quadratic Programming Based Structured Pruning for Large Language Models](https://www.sciencedirect.com/science/article/abs/pii/S0893608025012134) | LLM | `S` | - | Structured LLM pruning with outlier-aware quadratic-programming formulation. |

<h3 align="center">2024</h3>

| Journal | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| [Neural Networks](https://www.sciencedirect.com/journal/neural-networks) | [Efficient tensor decomposition-based filter pruning](https://www.sciencedirect.com/science/article/abs/pii/S0893608024003174) | CNN | `S` | [PyTorch[A]](https://github.com/pvtien96/CORING) | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Enhanced Network Compression Through Tensor Decompositions and Pruning](https://ieeexplore.ieee.org/document/10463116) | General | `S` | [PyTorch[A]](https://github.com/pvtien96/NORTON) | Structured pruning for hardware-friendly compression. |
| [IEEE Transactions on Artificial Intelligence](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=76) | [Distilled Gradual Pruning with Pruned Fine-tuning](https://ieeexplore.ieee.org/document/10438214) | General | `U` | [PyTorch[A]](https://github.com/rom42pla/dg2pf) | Unstructured sparsity / weight pruning method. |

<h3 align="center">2023</h3>

| Journal | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| [IEEE Trans Circuits Syst Video Technol](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=76) | [DCFP: Distribution Calibrated Filter Pruning for Lightweight and Accurate Long-tail Semantic Segmentation](https://ieeexplore.ieee.org/abstract/document/10364745) | Detection/Segmentation | `S` | - | Structured pruning for hardware-friendly compression. |
| [IEEE Internet Things J.](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6488907) | [SNPF: Sensitiveness Based Network Pruning Framework for Efficient Edge Computing](https://ieeexplore.ieee.org/document/10250769) | Deployment | `S` | - | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Manipulating Identical Filter Redundancy for Efficient Pruning on Deep and Complicated CNN](https://ieeexplore.ieee.org/document/10283855) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Block-Wise Partner Learning for Model Compression](https://ieeexplore.ieee.org/document/10237122) | General | `S` | [PyTorch[A]](https://github.com/zhangxin-xd/BPL) | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Hierarchical Threshold Pruning Based on Uniform Response Criterion](https://ieeexplore.ieee.org/document/10103912) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [CATRO: Channel Pruning via Class-Aware Trace Ratio Optimization](https://ieeexplore.ieee.org/document/10094002) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Adaptive Filter Pruning via Sensitivity Feedback](https://ieeexplore.ieee.org/document/10064249) | ViT/Transformer | `S` | - | Structured pruning for hardware-friendly compression. |
| [Neurocomputing](https://www.sciencedirect.com/journal/neurocomputing) | [Filter pruning with uniqueness mechanism in the frequency domain for efficient neural networks](https://www.sciencedirect.com/science/article/pii/S0925231223001364) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | [Compact Neural Network via Stacking Hybrid Units](https://ieeexplore.ieee.org/abstract/document/10275036) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | [Performance-aware Approximation of Global Channel Pruning for Multitask CNNs](https://arxiv.org/abs/2303.11923) | CNN | `S` | [PyTorch[A]](https://github.com/HankYe/PAGCP/tree/main) | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | [Adaptive Search-and-Training for Robust and Efficient Network Pruning](https://ieeexplore.ieee.org/abstract/document/10052756) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| [Image Vis. Comput.](https://www.sciencedirect.com/journal/image-and-vision-computing) | [Loss-aware automatic selection of structured pruning criteria for deep neural network acceleration](https://www.sciencedirect.com/science/article/pii/S0262885623001191) | General | `S` | [PyTorch[A]](https://github.com/ghimiredhikura/laasp) | Structured pruning for hardware-friendly compression. |
| [Comput. Vis. Image Underst.](https://www.sciencedirect.com/journal/computer-vision-and-image-understanding) | [Feature independent Filter Pruning by Successive Layers analysis](https://www.sciencedirect.com/science/article/pii/S1077314223002084) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| [IEEE Access](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6287639) | [Differentiable Neural Architecture, Mixed Precision and Accelerator Co-Search](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10266308) | Deployment | `S` | - | Structured pruning for hardware-friendly compression. |

<h3 align="center">2022</h3>

| Journal | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| [IEEE Trans. Image Process.](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=83) | [Efficient Layer Compression Without Pruning](https://ieeexplore.ieee.org/abstract/document/10214522) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | [Learning to Explore Distillability and Sparsability: A Joint Framework for Model Compression](https://ieeexplore.ieee.org/abstract/document/9804342) | General | `S` | - | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | [1xN Pattern for Pruning Convolutional Neural Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9847369) | CNN | `S` | [PyTorch[A]](https://github.com/lmbxmu/1xN) | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Filter Pruning by Switching to Neighboring CNNs With Good Attribute](https://ieeexplore.ieee.org/document/9716788) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Model Pruning Enables Efficient Federated Learning on Edge Devices](https://ieeexplore.ieee.org/abstract/document/9762360) | Federated/Privacy | `S` | - | Pruning method for federated or distributed settings. |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [DAIS: Automatic Channel Pruning via Differentiable Annealing Indicator Search](https://ieeexplore.ieee.org/document/9749778) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Network Pruning Using Adaptive Exemplar Filters](https://ieeexplore.ieee.org/document/9448300) | CNN | `S` | [PyTorch[A]](https://github.com/lmbxmu/EPruner) | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Carrying Out CNN Channel Pruning in a White Box](https://ieeexplore.ieee.org/document/9712474) | CNN | `S` | [PyTorch[A]](https://github.com/zyxxmu/White-Box) | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Pruning Networks With Cross-Layer Ranking & k-Reciprocal Nearest Filters](https://ieeexplore.ieee.org/document/9737040) | CNN | `S` | [PyTorch[A]](https://github.com/lmbxmu/CLR-RNF) | Structured pruning for hardware-friendly compression. |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Filter Sketch for Network Pruning](https://ieeexplore.ieee.org/document/9454340) | CNN | `S` | [PyTorch[A]](https://github.com/lmbxmu/FilterSketch) | Structured pruning for hardware-friendly compression. |
| [Neurocomputing](https://www.sciencedirect.com/journal/neurocomputing) | [FPFS: Filter-level pruning via distance weight measuring filter similarity](https://www.sciencedirect.com/science/article/pii/S092523122201164X) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| [Neurocomputing](https://www.sciencedirect.com/journal/neurocomputing) | [RUFP: Reinitializing unimportant filters for soft pruning](https://www.sciencedirect.com/science/article/pii/S0925231222001667) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| [Neural Netw](https://www.sciencedirect.com/journal/neural-networks) | [HRel: Filter pruning based on High Relevance between activation maps and class labels](https://www.sciencedirect.com/science/article/pii/S0893608021004962) | CNN | `S` | [PyTorch[A]*](https://github.com/sarvanichinthapalli/HRel) | Structured pruning for hardware-friendly compression. |
| [Comput. Intell. Neurosci.](https://www.hindawi.com/journals/cin/) | [Differentiable Network Pruning via Polarization of Probabilistic Channelwise Soft Masks](https://www.hindawi.com/journals/cin/2022/7775419/) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| [J. Syst. Archit.](https://www.sciencedirect.com/journal/journal-of-systems-architecture) | [Optimizing deep neural networks on intelligent edge accelerators via flexible-rate filter pruning](https://www.sciencedirect.com/science/article/pii/S1383762122000303) | Deployment | `S` | - | Structured pruning for hardware-friendly compression. |
| [Appl. Sci.](https://www.mdpi.com/journal/applsci) | [Magnitude and Similarity Based Variable Rate Filter Pruning for Efficient Convolution Neural Networks](https://www.mdpi.com/2076-3417/13/1/316) | Sparsity Theory | `S` | [PyTorch[A]](https://github.com/ghimiredhikura/MSVFP-FilterPruning) | Structured pruning for hardware-friendly compression. |
| [Sensors](https://www.mdpi.com/journal/sensors) | [Filter Pruning via Measuring Feature Map Information](https://www.mdpi.com/1424-8220/21/19/6601) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| [IEEE Access](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6287639) | [Automated Filter Pruning Based on High-Dimensional Bayesian Optimization](https://ieeexplore.ieee.org/document/9718082) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| [IEEE Signal Process. Lett.](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=97) | [A Low-Complexity Modified ThiNet Algorithm for Pruning Convolutional Neural Networks](https://ieeexplore.ieee.org/document/9748003) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |

<h3 align="center">2021</h3>

| Journal | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | [Discrimination-Aware Network Pruning for Deep Model Compression](https://ieeexplore.ieee.org/document/9384353) | General | `S` | [PyTorch[A]~](https://github.com/SCUT-AILab/DCP) | Structured pruning for hardware-friendly compression. |

<h3 align="center">2020</h3>  

| Journal | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [EDP: An Efficient Decomposition and Pruning Scheme for Convolutional Neural Network Compression](https://ieeexplore.ieee.org/document/9246734) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| [IEEE Access](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6287639) | [Filter Pruning Without Damaging Networks Capacity](https://ieeexplore.ieee.org/document/9091183) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |
| [Electronics](https://www.mdpi.com/journal/electronics) | [Pruning Convolutional Neural Networks with an Attention Mechanism for Remote Sensing Image Classification](https://www.mdpi.com/2079-9292/9/8/1209) | CNN | `S` | - | Structured pruning for hardware-friendly compression. |


## Survey Articles

| Year | Venue | Title | Area | Type | Code | Notes |
|:--:|:--|:--|:--|:--:|:--:|:--|
| `2026` | `arXiv` | [Towards Efficient Multimodal Large Language Models: A Survey on Token Compression](https://github.com/yaolinli/MLLM-Token-Compression) | LLM | - | - | Survey / overview resource. |
| `2024` | `IEEE TPAMI` | [A Survey on Deep Neural Network Pruning: Taxonomy, Comparison, Analysis, and Recommendations](https://arxiv.org/abs/2308.06767) | Survey | - | - | Survey / overview resource. |
| `2024` | `arXiv` | [A Survey on Model Compression for Large Language Models](https://arxiv.org/abs/2308.07633) | LLM | - | - | Survey / overview resource. |
| `2023` | [`Artif. Intell. Rev.`](https://www.springer.com/journal/10462) | [Deep neural network pruning method based on sensitive layers and reinforcement learning](https://link.springer.com/article/10.1007/s10462-023-10566-5) | General | - | - | Survey / overview resource. |
| `2023` | `arVix` | [A Survey on Deep Neural Network Pruning: Taxonomy, Comparison, Analysis, and Recommendations](https://arxiv.org/abs/2308.06767) | Survey | - | - | Survey / overview resource. |
| `2023` | `arVix` | [Structured Pruning for Deep Convolutional Neural Networks: A survey](https://arxiv.org/abs/2303.00566) | CNN | - | - | Survey / overview resource. |
| `2022` | [`Electronics`](https://www.mdpi.com/journal/electronics) | [A Survey on Efficient Convolutional Neural Networks and Hardware Acceleration](https://www.mdpi.com/2079-9292/11/6/945) | Deployment | - | - | Survey / overview resource. |
| `2022` | [`I-SMAC`](https://i-smac.org/ismac2022/) | [A Survey on Filter Pruning Techniques for Optimization of Deep Neural Networks](https://ieeexplore.ieee.org/abstract/document/9987264/) | CNN | - | - | Survey / overview resource. |
| `2021` | [`JMLR`](https://jmlr.csail.mit.edu/) | [Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks](https://www.jmlr.org/papers/volume22/21-0366/21-0366.pdf) | Sparsity Theory | - | - | Survey / overview resource. |
| `2021` | [`Neurocomputing`](https://www.sciencedirect.com/journal/neurocomputing) | [Pruning and quantization for deep neural network acceleration: A survey](https://www.sciencedirect.com/science/article/pii/S0925231221010894) | Survey | - | - | Survey / overview resource. |
| `2020` | [`IEEE Access`](https://ieeexplore.ieee.org/xpl/aboutJournal.jsp?punumber=6287639) | [Methods for Pruning Deep Neural Networks](https://ieeexplore.ieee.org/document/9795013/) | General | - | - | Survey / overview resource. |

## Other Publications
| Venue | Title | Area | Type | Code | Notes |
|:--|:--|:--|:--:|:--:|:--|
| `2023` | `arVix` | [Why is the State of Neural Network Pruning so Confusing? On the Fairness, Comparison Setup, and Trainability in Network Pruning](https://arxiv.org/abs/2301.05219) | General | - | [PyTorch[A]](https://github.com/mingsun-tse/why-the-state-of-pruning-so-confusing)| - |
| `2023` | `arVix` | [Ten Lessons We Have Learned in the New "Sparseland": A Short Handbook for Sparse Neural Network Researchers](https://arxiv.org/abs/2302.02596) | Sparsity Theory | - | - | - |
| `2022` | `ICML` | **Tutorial** -- [Sparsity in Deep Learning: Pruning and growth for efficient inference and training](https://icml.cc/virtual/2021/tutorial/10845) | Sparsity Theory | - | - | - |


## Pruning Software and Toolbox

| Year | Title | Area | Type | Code | Notes |
|:--:|:--|:--|:--:|:--:|:--|
| `2026` | [LLM Compressor / vLLM compression tools](https://docs.vllm.ai/projects/llm-compressor/en/latest/) | LLM | `U/S/Q/SS` | [Python[A]](https://github.com/vllm-project/llm-compressor) | Recent method for pruning or sparsifying large language models. |
| `2025` | [GETA](https://github.com/microsoft/geta) | General | `S/Q` | [PyTorch[A]](https://github.com/microsoft/geta) | - |
| `2025` | [PACT](https://github.com/orailix/PACT) | General | `T` | [PyTorch[A]](https://github.com/orailix/PACT) | - |
| `2025` | [DivPrune](https://github.com/vbdi/divprune) | General | `T` | [PyTorch[A]](https://github.com/vbdi/divprune) | - |
| `2025` | [Token Cropr](https://github.com/benbergner/cropr) | ViT/Transformer | `T` | [PyTorch[A]](https://github.com/benbergner/cropr) | Token/patch reduction for transformer acceleration. |
| `2025` | [2SSP](https://github.com/FabrizioSandri/2SSP) | General | `S` | [PyTorch[A]](https://github.com/FabrizioSandri/2SSP) | Structured pruning for hardware-friendly compression. |
| `2024` | [LLM-Shearing / Sheared LLaMA](https://github.com/princeton-nlp/LLM-Shearing) | LLM | `S` | [PyTorch[A]](https://github.com/princeton-nlp/LLM-Shearing) | Recent method for pruning or sparsifying large language models. |
| `2024` | [LoRAPrune](https://github.com/aim-uofa/LoRAPrune) | General | `S` | [PyTorch[A]](https://github.com/aim-uofa/LoRAPrune) | Structured pruning for hardware-friendly compression. |
| `2023` | [SparseGPT](https://github.com/IST-DASLab/sparsegpt) | LLM | `U/SS` | [PyTorch[A]](https://github.com/IST-DASLab/sparsegpt) | Recent method for pruning or sparsifying large language models. |
| `2023` | [Wanda](https://github.com/locuslab/wanda) | General | `U/SS` | [PyTorch[A]](https://github.com/locuslab/wanda) | - |
| `2023` | [UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers](https://proceedings.mlr.press/v202/shi23e.html) | VLM/MLLM | `S` | [PyTorch[A]](https://github.com/sdc17/UPop) | Recent method for pruning or sparsifying large language models. |
| `2023` | [DepGraph: Towards Any Structural Pruning](https://openaccess.thecvf.com/content/CVPR2023/papers/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.pdf) | General | `S` | [PyTorch[A]](https://github.com/VainF/Torch-Pruning) | Structured pruning for hardware-friendly compression. |
| `2023` | **[Torch-Pruning](https://arxiv.org/abs/2301.12900)** | Toolbox | `S` | [PyTorch[A]](https://github.com/VainF/Torch-Pruning) | Software resource for pruning/compression workflows. |
| `2023` | [JaxPruner: JaxPruner: A concise library for sparsity research](https://arxiv.org/abs/2304.14082) | Sparsity Theory | `U/S` | [PyTorch[A]](https://github.com/google-research/jaxpruner) | - |
| `2022` | [FasterAI: Prune and Distill your models with FastAI and PyTorch](https://nathanhubens.github.io/fasterai/) | General | `U` | [PyTorch[A]](https://github.com/nathanhubens/fasterai) | Unstructured sparsity / weight pruning method. |
| `2022` | [Simplify: A Python library for optimizing pruned neural networks](https://www.sciencedirect.com/science/article/pii/S2352711021001576) | Toolbox | - | [PyTorch[A]](https://github.com/EIDOSlab/simplify) | Software resource for pruning/compression workflows. |
| `2021` | PyTorchViz [A small package to create visualizations of PyTorch execution graphs] | General | - | [PyTorch[A]](https://github.com/szagoruyko/pytorchviz) | - |
| `2020` | [What is the State of Neural Network Pruning?](https://proceedings.mlsys.org/paper/2020/file/d2ddea18f00665ce8623e36bd4e3c7c5-Paper.pdf) | General | `S/U` | [PyTorch[A]](https://github.com/jjgo/shrinkbench) | - |
| `2019` | [Official PyTorch Pruning Tool](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) | Toolbox | `S/U` | [PyTorch[A]](https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/prune.py) | Software resource for pruning/compression workflows. |

## Citation

If this repository helps your research, please cite it as:

```bibtex
@misc{awesome_pruning,
  title  = {Awesome Pruning},
  author = {Deepak Ghimire and others},
  year   = {2026},
  note   = {Curated list of pruning papers, surveys, software, and code across CNNs, ViTs, LLMs, VLMs/MLLMs, diffusion, and deployment-aware compression}
}
```
