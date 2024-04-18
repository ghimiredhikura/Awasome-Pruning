# Awesome Pruning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Awesome resources in deep neural network pruning. This collection is inspired by [he-y/Awesome-Pruning](https://github.com/he-y/Awesome-Pruning). 

> [Note: You are welcome to create pool requests and add more interesting papers.]

| Section | Year of Publication|
|:---------| :--- |
| **[Conference Publications](#conference-publications)** | [`2024`](#2024)  [`2023`](#2023)  [`2022`](#2022) [`2021`](#2021)  [`2020`](#2020) [`2019`](#2019) [`2018`](#2018) [`2017`](#2017)|
| **[Journal Publications](#journal-publications)** | [`2024`](#2024-1) [`2023`](#2023-1)  [`2022`](#2022-1) [`2021`](#2021-1)  [`2020`](#2020-1) |
| **[Survey Articles](#survey-articles)** | [`2020~2023`](#survey-articles) |
| **[Other Publications](#other-publications)** | [`2022~2023`](#other-publications) |
| **[Pruning Software and Toolbox](#pruning-software-and-toolbox)** | [`2019~2023`](#pruning-software-and-toolbox) |

| Symbol | Meaning |
|:---------:|:-----------|
| `U` |Unstructured or Weight Pruning| 
| `S` |Structured or Filter or Channel Pruning| 
| `A` |Official or Author Implementation| 
| `O` |Unofficial or 3rd Party Implementation| 

## Conference Publications

**<h3 align="center">2024</h3>**

| Venue | Title | Type | Code |
|:-----|:------|:-----:|:----:|
| `ICLR`  | [Towards Meta-Pruning via Optimal Transport](https://openreview.net/forum?id=sMoifbuxjB) | `S` | [PyTorch[A]](https://github.com/alexandertheus/Intra-Fusion) <span id="stars-repo2">Loading...</span> |
| `ICLR`  | [Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework](https://openreview.net/forum?id=eoSeaK4QJo) | `U` | [PyTorch[A]](https://github.com/xyshi2000/Unstructured-Pruning) |
| `ICLR`  | [Masks, Signs, And Learning Rate Rewinding](https://openreview.net/forum?id=qODvxQ8TXW) | `S` | [PyTorch[A]](https://github.com/xyshi2000/Unstructured-Pruning) |
| `ICLR`  | [Scaling Laws for Sparsely-Connected Foundation Models](https://openreview.net/forum?id=i9K2ZWkYIP) | `S` | [PyTorch[A]](https://github.com/google-research/jaxpruner/tree/main/jaxpruner/projects/bigsparse) |
| `ICLR`  | [Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging](https://openreview.net/forum?id=xx0ITyHp3u) | `S` |  |
| `ICLR`  | [Adaptive Sharpness-Aware Pruning for Robust Sparse Networks](https://openreview.net/forum?id=QFYVVwiAM8) | `S` |  |
| `ICLR`  | [What Makes a Good Prune? Maximal Unstructured Pruning for Maximal Cosine Similarity](https://openreview.net/forum?id=jsvvPVVzwf) | `U` | [PyTorch[A]](https://github.com/gmw99/what_makes_a_good_prune) |
| `ICLR`  | [In defense of parameter sharing for model-compression](https://openreview.net/forum?id=ypAT2ixD4X) | `S/U` |  |
| `ICLR`  | [ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models](https://openreview.net/forum?id=iIT02bAKzv) | `U` | |
| `ICLR`  | [Data-independent Module-aware Pruning for Hierarchical Vision Transformers](https://openreview.net/forum?id=7Ol6foUi1G) | `S` | [PyTorch[A]](https://github.com/he-y/Data-independent-Module-Aware-Pruning) |
| `ICLR`  | [SWAP: Sparse Entropic Wasserstein Regression for Robust Network Pruning](https://openreview.net/forum?id=LJWizuuBUy) | `S` |  |
| `ICLR`  | [Sparse Weight Averaging with Multiple Particles for Iterative Magnitude Pruning](https://openreview.net/forum?id=Y9t7MqZtCR) | `U` | |
| `ICLR`  | [Synergistic Patch Pruning for Vision Transformer: Unifying Intra- & Inter-Layer Patch Importance](https://openreview.net/forum?id=COO51g41Q4) | `S` |  |
| `ICLR`  | [FedP3: Federated Personalized and Privacy-friendly Network Pruning under Model Heterogeneity](https://openreview.net/forum?id=hbHwZYqk9T) | `S` |  |
| `ICLR`  | [The Need for Speed: Pruning Transformers with One Recipe](https://openreview.net/forum?id=MVmT6uQ3cQ) | `S` | [PyTorch[A]](https://github.com/Skhaki18/optin-transformer-pruning) |
| `ICLR`  | [SAS: Structured Activation Sparsification](https://openreview.net/forum?id=vZfi5to2Xl) | `S` | [PyTorch[A]](https://github.com/DensoITLab/sas_) |
| `CVPR`  | [OrthCaps: An Orthogonal CapsNet with Sparse Attention Routing and Pruning](https://arxiv.org/abs/2403.13351) | `S` | [PyTorch[A]](https://github.com/ornamentt/OrthCap) |
| `CVPR`  | [Zero-TPrune: Zero-Shot Token Pruning through Leveraging of the Attention Graph in Pre-Trained Transformers](https://arxiv.org/abs/2305.17328) | `S` | [PyTorch[A]](https://jha-lab.github.io/zerotprune/) |
| `CVPR`  | [Finding Lottery Tickets in Vision Models via Data-driven Spectral Foresight Pruning](https://github.com/iurada/px-ntk-pruning) | `S` | [PyTorch[A]](https://github.com/iurada/px-ntk-pruning) |
| `CVPR`  | BilevelPruning: Unified Dynamic and Static Channel Pruning for Convolutional Neural Networks | `S` | |
| `CVPR`  | [FedMef: Towards Memory-efficient Federated Dynamic Pruning](https://arxiv.org/pdf/2403.14737.pdf) | `S` | |
| `CVPR`  | Resource-Efficient Transformer Pruning for Finetuning of Large Models | `S` | |
| `CVPR`  | Device-Wise Federated Network Pruning | `S` | |
| `CVPR`  | [Auto-Train-Once: Controller Network Guided Automatic Network Pruning from Scratch](https://arxiv.org/abs/2403.14729) | `S` | |
| `CVPR`  | [Jointly Training and Pruning CNNs via Learnable Agent Guidance and Alignment](https://arxiv.org/abs/2403.14729) | `S` | |
| `CVPR`  | [Diversity-aware Channel Pruning for StyleGAN Compression](https://arxiv.org/abs/2403.13548) | `S` | [PyTorch[A]](https://jiwoogit.github.io/DCP-GAN_site/) |
| `CVPR`  | [MADTP: Multimodal Alignment-Guided Dynamic Token Pruning for Accelerating Vision-Language Transformer](https://arxiv.org/abs/2403.02991) | `S` | [PyTorch[A]](https://github.com/double125/MADTP) |
| `AAAI`  | [Dynamic Feature Pruning and Consolidation for Occluded Person Re-Identification](https://arxiv.org/abs/2211.14742) | `S` | |
| `AAAI`  | [REPrune: Channel Pruning via Kernel Representative Selection](https://arxiv.org/abs/2211.14742) | `S` | |
| `AAAI`  | [Revisiting Gradient Pruning: A Dual Realization for Defending against Gradient Attacks](https://arxiv.org/abs/2401.16687) | `S` | |
| `AAAI`  | [IRPruneDet: Efficient Infrared Small Target Detection via Wavelet Structure-Regularized Soft Channel Pruning](https://ojs.aaai.org/index.php/AAAI/article/view/28551) | `S` | |
| `AAAI`  | [EPSD: Early Pruning with Self-Distillation for Efficient Model Compression](https://arxiv.org/abs/2402.00084) | `S` | |
| `WACV`  | [Pruning from Scratch via Shared Pruning Module and Nuclear norm-based Regularization](https://openaccess.thecvf.com/content/WACV2024/papers/Lee_Pruning_From_Scratch_via_Shared_Pruning_Module_and_Nuclear_Norm-Based_WACV_2024_paper.pdf) | `S` | [PyTorch[A]](https://github.com/jsleeg98/NuSPM) |
| `WACV`  | [Towards Better Structured Pruning Saliency by Reorganizing Convolution](https://openaccess.thecvf.com/content/WACV2024/papers/Sun_Towards_Better_Structured_Pruning_Saliency_by_Reorganizing_Convolution_WACV_2024_paper.pdf) | `S` | [PyTorch[A]](https://github.com/AlexSunNik/SPSRC) |
| `WACV`  | [Torque based Structured Pruning for Deep Neural Network](https://openaccess.thecvf.com/content/WACV2024/papers/Gupta_Torque_Based_Structured_Pruning_for_Deep_Neural_Network_WACV_2024_paper.pdf) | `S` |  |
| `WACV`  | [Revisiting Token Pruning for Object Detection and Instance Segmentation](https://openaccess.thecvf.com/content/WACV2024/html/Liu_Revisiting_Token_Pruning_for_Object_Detection_and_Instance_Segmentation_WACV_2024_paper.html) | `S` | [PyTorch[A]](https://github.com/uzh-rpg/svit/) |
| `WACV`  | [Token Fusion: Bridging the Gap Between Token Pruning and Token Merging](https://openaccess.thecvf.com/content/WACV2024/html/Kim_Token_Fusion_Bridging_the_Gap_Between_Token_Pruning_and_Token_WACV_2024_paper.html) | `S` |  |
| `WACV`  | [PATROL: Privacy-Oriented Pruning for Collaborative Inference Against Model Inversion Attacks](https://openaccess.thecvf.com/content/WACV2024/html/Ding_PATROL_Privacy-Oriented_Pruning_for_Collaborative_Inference_Against_Model_Inversion_Attacks_WACV_2024_paper.html) | `S` |  |

**<h3 align="center">2023</h3>**

| Venue | Title | Type | Code |
|:-----|:------|:-----:|:----:|
| `NIPS`  | [Diff-Pruning: Structural Pruning for Diffusion Models](https://arxiv.org/abs/2305.10924) | `S` | [PyTorch[A]](https://github.com/VainF/Diff-Pruning) |
| `NIPS`  | [LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627) | `S` | [PyTorch[A]](https://github.com/horseee/LLM-Pruner) |
| `ICCV`  | [Automatic Network Pruning via Hilbert-Schmidt Independence Criterion Lasso under Information Bottleneck Principle](https://openaccess.thecvf.com/content/ICCV2023/html/Guo_Automatic_Network_Pruning_via_Hilbert-Schmidt_Independence_Criterion_Lasso_under_Information_ICCV_2023_paper.html) | `S` | [PyTorch[A]](https://github.com/sunggo/APIB) |
| `ICCV`  | [Unified Data-Free Compression: Pruning and Quantization without Fine-Tuning](https://openaccess.thecvf.com/content/ICCV2023/html/Bai_Unified_Data-Free_Compression_Pruning_and_Quantization_without_Fine-Tuning_ICCV_2023_paper.html) | `S` | [PyTorch[A]](https://github.com/Dtudy/UDFC) |
| `ICCV`  | [Structural Alignment for Network Pruning through Partial Regularization](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_Structural_Alignment_for_Network_Pruning_through_Partial_Regularization_ICCV_2023_paper.html) | `S` | |
| `ICCV`  | [Differentiable Transportation Pruning](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Differentiable_Transportation_Pruning_ICCV_2023_paper.html) | `S` | |
| `ICCV`  | [Dynamic Token Pruning in Plain Vision Transformers for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2023/html/Tang_Dynamic_Token_Pruning_in_Plain_Vision_Transformers_for_Semantic_Segmentation_ICCV_2023_paper.html) | `S` | [PyTorch[A]](https://github.com/zbwxp/Dynamic-Token-Pruning) |
| `ICCV`  | [Towards Fairness-aware Adversarial Network Pruning](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Towards_Fairness-aware_Adversarial_Network_Pruning_ICCV_2023_paper.html) | `S` | |
| `ICCV`  | [Efficient Joint Optimization of Layer-Adaptive Weight Pruning in Deep Neural Networks](https://openaccess.thecvf.com/content/ICCV2023/html/Xu_Efficient_Joint_Optimization_of_Layer-Adaptive_Weight_Pruning_in_Deep_Neural_ICCV_2023_paper.html) | `S` | [PyTorch[A]](https://github.com/Akimoto-Cris/RD_VIT_PRUNE) |
| `CVPR`  | [DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900) | `S` | [PyTorch[A]](https://github.com/VainF/Torch-Pruning) |
| `CVPR`  | [X-Pruner: eXplainable Pruning for Vision Transformers](https://arxiv.org/abs/2303.04935) | `U/S` |  |
| `CVPR`  | [Joint Token Pruning and Squeezing Towards More Aggressive Compression of Vision Transformers](https://arxiv.org/abs/2304.10716) | `S` | [PyTorch[A]](https://github.com/megvii-research/TPS-CVPR2023) |
| `CVPR`  | [Global Vision Transformer Pruning with Hessian-Aware Saliency](https://arxiv.org/abs/2110.04869) | `S` |  |
| `CVPR`  | [CP3: Channel Pruning Plug-in for Point-based Networks](https://arxiv.org/abs/2303.13097) | `S` |  |
| `CVPR`  | [Training Debiased Subnetworks With Contrastive Weight Pruning](https://openaccess.thecvf.com/content/CVPR2023/html/Park_Training_Debiased_Subnetworks_With_Contrastive_Weight_Pruning_CVPR_2023_paper.html) | `U` | |
| `CVPR`  | [Pruning Parameterization With Bi-Level Optimization for Efficient Semantic Segmentation on the Edge](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Pruning_Parameterization_With_Bi-Level_Optimization_for_Efficient_Semantic_Segmentation_on_CVPR_2023_paper.html) | `S` | |
| `ICLR`  | [JaxPruner: A concise library for sparsity research](https://arxiv.org/abs/2304.14082) | `U/S` | [PyTorch[A]](https://github.com/google-research/jaxpruner) |
| `ICLR`  | [OTOv2: Automatic, Generic, User-Friendly](https://openreview.net/forum?id=7ynoX1ojPMt) | `S` | [PyTorch[A]](https://github.com/tianyic/only_train_once) |
| `ICLR`  | [How I Learned to Stop Worrying and Love Retraining](https://openreview.net/forum?id=_nF5imFKQI) | `U` | [PyTorch[A]](https://github.com/ZIB-IOL/BIMP) |
| `ICLR`  | [Token Merging: Your ViT But Faster ](https://openreview.net/forum?id=JroZRaRw7Eu) | `U/S` | [PyTorch[A]](https://github.com/facebookresearch/ToMe) |
| `ICLR`  | [Revisiting Pruning at Initialization Through the Lens of Ramanujan Graphs](https://openreview.net/forum?id=uVcDssQff_) | `U` | [PyTorch[A]](https://github.com/VITA-Group/ramanujan-on-pai) (soon...) |
| `ICLR`  | [Unmasking the Lottery Ticket Hypothesis: What's Encoded in a Winning Ticket's Mask?](https://openreview.net/forum?id=xSsW2Am-ukZ) | `U` |  |
| `ICLR`  | [NTK-SAP: Improving neural network pruning by aligning training dynamics](https://openreview.net/forum?id=-5EWhW_4qWP) | `U` |  |
| `ICLR`  | [DFPC: Data flow driven pruning of coupled channels without data](https://openreview.net/forum?id=mhnHqRqcjYU) | `S` | [PyTorch[A]](https://github.com/TanayNarshana/DFPC-Pruning) |
| `ICLR`  | [TVSPrune - Pruning Non-discriminative filters via Total Variation separability of intermediate representations without fine tuning](https://openreview.net/forum?id=sZI1Oj9KBKy) | `S` | [PyTorch[A]](https://github.com/chaimurti/TVSPrune) |
| `ICLR`  | [Pruning Deep Neural Networks from a Sparsity Perspective](https://openreview.net/forum?id=i-DleYh34BM) | `U` | [PyTorch[A]](https://openreview.net/attachment?id=i-DleYh34BM&name=supplementary_material) |
| `ICLR`  | [A Unified Framework of Soft Threshold Pruning](https://openreview.net/forum?id=cCFqcrq0d8) | `U` | [PyTorch[A]](https://openreview.net/attachment?id=cCFqcrq0d8&name=supplementary_material) |
| `WACV`  | [Calibrating Deep Neural Networks Using Explicit Regularisation and Dynamic Data Pruning](https://openaccess.thecvf.com/content/WACV2023/html/Patra_Calibrating_Deep_Neural_Networks_Using_Explicit_Regularisation_and_Dynamic_Data_WACV_2023_paper.html) | `S` | |
| `WACV`  | [Attend Who Is Weak: Pruning-Assisted Medical Image Localization Under Sophisticated and Implicit Imbalances](https://openaccess.thecvf.com/content/WACV2023/html/Jaiswal_Attend_Who_Is_Weak_Pruning-Assisted_Medical_Image_Localization_Under_Sophisticated_WACV_2023_paper.html) | `S` | |
| [`ICASSP`](https://2023.ieeeicassp.org/important-dates/)  | [WHC: Weighted Hybrid Criterion for Filter Pruning on Convolutional Neural Networks](https://arxiv.org/abs/2302.08185) | `S` | [PyTorch[A]](https://github.com/ShaowuChen/WHC) |

<h3 align="center">2022</h3>

| Venue | Title | Type | Code |
|:-----|:------|:-----:|:----:|
| `CVPR`  | [Interspace Pruning: Using Adaptive Filter Representations To Improve Training of Sparse CNNs](https://openaccess.thecvf.com/content/CVPR2022/html/Wimmer_Interspace_Pruning_Using_Adaptive_Filter_Representations_To_Improve_Training_of_CVPR_2022_paper.html) | `U` | |
| `CVPR`  | [Revisiting Random Channel Pruning for Neural Network Compression](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Revisiting_Random_Channel_Pruning_for_Neural_Network_Compression_CVPR_2022_paper.html) | `S` | [PyTorch[A]](https://github.com/ofsoundof/random_channel_pruning) (soon...) |
| `CVPR`  | [Fire Together Wire Together: A Dynamic Pruning Approach With Self-Supervised Mask Prediction](https://openaccess.thecvf.com/content/CVPR2022/html/Elkerdawy_Fire_Together_Wire_Together_A_Dynamic_Pruning_Approach_With_Self-Supervised_CVPR_2022_paper.html) | `S` | [PyTorch[A]](https://github.com/selkerdawy/FTWT) |
| `CVPR`  | [When to Prune? A Policy towards Early Structural Pruning](https://openaccess.thecvf.com/content/CVPR2022/html/Shen_When_To_Prune_A_Policy_Towards_Early_Structural_Pruning_CVPR_2022_paper.html) | `S` | |
| `CVPR`  | [Dreaming to Prune Image Deraining Networks](https://openaccess.thecvf.com/content/CVPR2022/papers/Zou_Dreaming_To_Prune_Image_Deraining_Networks_CVPR_2022_paper.pdf) | `S` | |
| `ICLR`  | [SOSP: Efficiently Capturing Global Correlations by Second-Order Structured Pruning](https://openreview.net/forum?id=t5EmXZ3ZLR) | `S` | |
| `ICLR`  | [Learning Pruning-Friendly Networks via Frank-Wolfe: One-Shot, Any-Sparsity, And No Retraining](https://openreview.net/forum?id=O1DEtITim__) | `U` | [PyTorch[A]](https://github.com/VITA-Group/SFW-Once-for-All-Pruning) |
| `ICLR`  | [Revisit Kernel Pruning with Lottery Regulated Grouped Convolutions](https://openreview.net/forum?id=LdEhiMG9WLO) | `S` | [PyTorch[A]](https://github.com/choH/lottery_regulated_grouped_kernel_pruning) | 
| `ICLR`  | [Dual Lottery Ticket Hypothesis](https://openreview.net/forum?id=fOsN52jn25l) | `U` |[PyTorch[A]](https://github.com/yueb17/DLTH) |
| `NIPS`  | [SAViT: Structure-Aware Vision Transformer Pruning via Collaborative Optimization](https://openreview.net/forum?id=w5DacXWzQ-Q) | `S` |[PyTorch[A]](https://github.com/hikvision-research/SAViT)(soon...) |
| `NIPS`  | [Structural Pruning via Latency-Saliency Knapsack](https://openreview.net/forum?id=cUOR-_VsavA) | `S` |[PyTorch[A]](https://github.com/NVlabs/HALP)|
| `ACCV`  | [Filter Pruning via Automatic Pruning Rate Search⋆](https://openaccess.thecvf.com/content/ACCV2022/html/Sun_Filter_Pruning_via_Automatic_Pruning_Rate_Search_ACCV_2022_paper.html) | `S` | |
| `ACCV`  | [Network Pruning via Feature Shift Minimization](https://openaccess.thecvf.com/content/ACCV2022/html/Duan_Network_Pruning_via_Feature_Shift_Minimization_ACCV_2022_paper.html) | `S` | [PyTorch[A]](https://github.com/lscgx/FSM) |
| `ACCV`  | [Lightweight Alpha Matting Network Using Distillation-Based Channel Pruning](https://openaccess.thecvf.com/content/ACCV2022/html/Yoon_Lightweight_Alpha_Matting_Network_Using_Distillation-Based_Channel_Pruning_ACCV_2022_paper.html) | `S` | [PyTorch[A]](https://github.com/DongGeun-Yoon/DCP) |
| `ACCV`  | [Adaptive FSP : Adaptive Architecture Search with Filter Shape Pruning](https://openaccess.thecvf.com/content/ACCV2022/html/Kim_Adaptive_FSP__Adaptive_Architecture_Search_with_Filter_Shape_Pruning_ACCV_2022_paper.html) | `S` | |
| `ECCV`  | [Soft Masking for Cost-Constrained Channel Pruning](https://link.springer.com/chapter/10.1007/978-3-031-20083-0_38) | `S` | [PyTorch[A]](https://github.com/NVlabs/SMCP) |
| `WACV`  | [Hessian-Aware Pruning and Optimal Neural Implant](https://openaccess.thecvf.com/content/WACV2022/papers/Yu_Hessian-Aware_Pruning_and_Optimal_Neural_Implant_WACV_2022_paper.pdf) | `S` | [PyTorch[A]](https://github.com/yaozhewei/HAP) |
| `WACV`  | [PPCD-GAN: Progressive Pruning and Class-Aware Distillation for Large-Scale Conditional GANs Compression](https://openaccess.thecvf.com/content/WACV2022/papers/Vo_PPCD-GAN_Progressive_Pruning_and_Class-Aware_Distillation_for_Large-Scale_Conditional_GANs_WACV_2022_paper.pdf) | `S` |  |
| `WACV`  | [Channel Pruning via Lookahead Search Guided Reinforcement Learning](https://openaccess.thecvf.com/content/WACV2022/papers/Wang_Channel_Pruning_via_Lookahead_Search_Guided_Reinforcement_Learning_WACV_2022_paper.pdf) | `S` |  |
| `WACV`  | [EZCrop: Energy-Zoned Channels for Robust Output Pruning](https://openaccess.thecvf.com/content/WACV2022/papers/Lin_EZCrop_Energy-Zoned_Channels_for_Robust_Output_Pruning_WACV_2022_paper.pdf) | `S` | [PyTorch[A]](https://github.com/rlin27/EZCrop) |
| `ICIP`  | [One-Cycle Pruning: Pruning Convnets With Tight Training Budget](https://ieeexplore.ieee.org/document/9897980) | `U` | |
| `ICIP`  | [RAPID: A Single Stage Pruning Framework](https://ieeexplore.ieee.org/document/9898000) | `U` | |
| `ICIP`  | [The Rise of the Lottery Heroes: Why Zero-Shot Pruning is Hard](https://ieeexplore.ieee.org/document/9897223) | `U` | |
| `ICIP`  | [Truncated Lottery Ticket for Deep Pruning](https://ieeexplore.ieee.org/document/9897767) | `U` | |
| `ICIP`  | [Which Metrics For Network Pruning: Final Accuracy? or Accuracy Drop?](https://ieeexplore.ieee.org/document/9898051) | `S/U` | |
| `ISMSI` | [Structured Pruning with Automatic Pruning Rate Derivation for Image Processing Neural Networks](https://dl.acm.org/doi/abs/10.1145/3533050.3533066) | `S` | |

<h3 align="center">2021</h3>

| Venue | Title | Type | Code |
|:-----|:------|:-----:|:----:|
| `ICLR`  | [Neural Pruning via Growing Regularization](https://openreview.net/forum?id=o966_Is_nPA) | `S` | [PyTorch[A]](https://github.com/mingsun-tse/regularization-pruning) |
| `ICLR`  | [Network Pruning That Matters: A Case Study on Retraining Variants](https://openreview.net/forum?id=Cb54AMqHQFP) | `S` | [PyTorch[A]](https://github.com/lehduong/NPTM) |
| `ICLR`  | [Layer-adaptive Sparsity for the Magnitude-based Pruning](https://openreview.net/forum?id=H6ATjJ0TKdf) | `U` | [PyTorch[A]](https://github.com/jaeho-lee/layer-adaptive-sparsity) |
| `NIPS`  | [Only Train Once: A One-Shot Neural Network Training And Pruning Framework](https://openreview.net/forum?id=p5rMPjrcCZq) | `S` | [PyTorch[A]](https://github.com/tianyic/only_train_once) |
| `CVPR`  | [NPAS: A Compiler-Aware Framework of Unified Network Pruning and Architecture Search for Beyond Real-Time Mobile Acceleration](https://openaccess.thecvf.com/content/CVPR2021/html/Li_NPAS_A_Compiler-Aware_Framework_of_Unified_Network_Pruning_and_Architecture_CVPR_2021_paper.html) | `S` | |
| `CVPR`  | [Network Pruning via Performance Maximization](https://openaccess.thecvf.com/content/CVPR2021/html/Gao_Network_Pruning_via_Performance_Maximization_CVPR_2021_paper.html) | `S` | |
| `CVPR`  | [Convolutional Neural Network Pruning With Structural Redundancy Reduction*](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Convolutional_Neural_Network_Pruning_With_Structural_Redundancy_Reduction_CVPR_2021_paper.html) | `S` | |
| `CVPR`  | [Manifold Regularized Dynamic Network Pruning](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Manifold_Regularized_Dynamic_Network_Pruning_CVPR_2021_paper.html) | `S` | [PyTorch[A]](https://github.com/yehuitang/Pruning/tree/master/ManiDP) |
| `CVPR`  | [Joint-DetNAS: Upgrade Your Detector With NAS, Pruning and Dynamic Distillation](https://openaccess.thecvf.com/content/CVPR2021/html/Yao_Joint-DetNAS_Upgrade_Your_Detector_With_NAS_Pruning_and_Dynamic_Distillation_CVPR_2021_paper.html) | `S` | |
| `ICCV`  | [ResRep: Lossless CNN Pruning via Decoupling Remembering and Forgetting](https://openaccess.thecvf.com/content/ICCV2021/html/Ding_ResRep_Lossless_CNN_Pruning_via_Decoupling_Remembering_and_Forgetting_ICCV_2021_paper.html) | `S` | |
| `ICCV`  | [Achieving On-Mobile Real-Time Super-Resolution With Neural Architecture and Pruning Search](https://openaccess.thecvf.com/content/ICCV2021/html/Zhan_Achieving_On-Mobile_Real-Time_Super-Resolution_With_Neural_Architecture_and_Pruning_Search_ICCV_2021_paper.html) | `S` | |
| `ICCV`  | [GDP: Stabilized Neural Network Pruning via Gates With Differentiable Polarization*](https://openaccess.thecvf.com/content/ICCV2021/html/Guo_GDP_Stabilized_Neural_Network_Pruning_via_Gates_With_Differentiable_Polarization_ICCV_2021_paper.html) | `S` | |
| `WACV`  | [Holistic Filter Pruning for Efficient Deep Neural Networks](https://openaccess.thecvf.com/content/WACV2021/html/Enderich_Holistic_Filter_Pruning_for_Efficient_Deep_Neural_Networks_WACV_2021_paper.html) | `S` | |
| `ICML`  | [Accelerate CNNs from Three Dimensions: A Comprehensive Pruning Framework](https://icml.cc/virtual/2021/poster/9081) | `S` | |
| `ICML`  | [Group Fisher Pruning for Practical Network Compression](https://icml.cc/virtual/2021/poster/9875) | `S` | [PyTorch[A]](https://github.com/jshilong/FisherPruning) |

<h3 align="center">2020</h3>

| Venue | Title | Type | Code |
|:-----|:------|:-----:|:----:|
| `CVPR`  | [HRank: Filter Pruning using High-Rank Feature Map](https://openaccess.thecvf.com/content_CVPR_2020/html/Lin_HRank_Filter_Pruning_Using_High-Rank_Feature_Map_CVPR_2020_paper.html) | `S` | [PyTorch[A]](https://github.com/lmbxmu/HRank) |
| `CVPR`  | [Towards efficient model compression via learned global ranking](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chin_Towards_Efficient_Model_Compression_via_Learned_Global_Ranking_CVPR_2020_paper.pdf) | `S` | [PyTorch[A]](https://github.com/enyac-group/LeGR) |
| `CVPR`  | [Learning Filter Pruning Criteria for Deep Convolutional Neural Networks Acceleration](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Learning_Filter_Pruning_Criteria_for_Deep_Convolutional_Neural_Networks_Acceleration_CVPR_2020_paper.html) | `S` | |
| `CVPR`  | [Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Group_Sparsity_The_Hinge_Between_Filter_Pruning_and_Decomposition_for_CVPR_2020_paper.html) | `S` | [PyTorch[A]](https://github.com/ofsoundof/group_sparsity) |
| `CVPR`  | [APQ: Joint Search for Network Architecture, Pruning and Quantization Policy](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_APQ_Joint_Search_for_Network_Architecture_Pruning_and_Quantization_Policy_CVPR_2020_paper.html) | `S` | [PyTorch[A]](https://github.com/mit-han-lab/apq) |
| `ICLR`  | [Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints](https://openreview.net/forum?id=HyxLRTVKPH) | `U` | |
| `MLSys` | [Shrinkbench: What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033) | | [PyTorch[A]](https://github.com/JJGO/shrinkbench) |
| `BMBS`  | [Similarity Based Filter Pruning for Efficient Super-Resolution Models](https://ieeexplore.ieee.org/abstract/document/9379712) | `S` |  |

<h3 align="center">2019</h3>

| Venue | Title | Type | Code |
|:-----|:------|:-----:|:----:|
| `CVPR`  | [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.html) | `S` |[PyTorch[A]](https://github.com/he-y/filter-pruning-geometric-median) |
| `CVPR`  | [Variational Convolutional Neural Network Pruning](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Variational_Convolutional_Neural_Network_Pruning_CVPR_2019_paper.html) | `S` | |
| `CVPR`  | [Towards Optimal Structured CNN Pruning via Generative Adversarial Learning](https://openaccess.thecvf.com/content_CVPR_2019/html/Lin_Towards_Optimal_Structured_CNN_Pruning_via_Generative_Adversarial_Learning_CVPR_2019_paper.html) | `S` | [PyTorch[A]](https://github.com/ShaohuiLin/GAL) |
| `CVPR`  | [Partial Order Pruning: For Best Speed/Accuracy Trade-Off in Neural Architecture Search](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Partial_Order_Pruning_For_Best_SpeedAccuracy_Trade-Off_in_Neural_Architecture_CVPR_2019_paper.html) | `S` | [PyTorch[A]](https://github.com/lixincn2015/Partial-Order-Pruning) |
| `CVPR`  | [Importance Estimation for Neural Network Pruning](https://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html) | `S` | [PyTorch[A]](https://github.com/NVlabs/Taylor_pruning) |
| `ICLR`  | [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://openreview.net/forum?id=rJl-b3RcF7) | `U` | [PyTorch[A]](https://github.com/facebookresearch/open_lth) |
| `ICLR`  | [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://openreview.net/forum?id=B1VZqjAcYX) | `U` | [Tensorflow[A]](https://github.com/namhoonlee/snip-public) |
| `ICCV`  | [MetaPruning: Meta-Learning for Automatic Neural Network Channel Pruning](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_MetaPruning_Meta_Learning_for_Automatic_Neural_Network_Channel_Pruning_ICCV_2019_paper.html) | `S` | [PyTorch[A]](https://github.com/liuzechun/MetaPruning) |
| `ICCV`  | [Accelerate CNN via Recursive Bayesian Pruning](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Accelerate_CNN_via_Recursive_Bayesian_Pruning_ICCV_2019_paper.html) | `S` |  |

<h3 align="center">2018</h3>

| Venue | Title | Type | Code |
|:-----|:------|:-----:|:----:|
| `CVPR`  | [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://openaccess.thecvf.com/content_cvpr_2018/html/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.html) | `S` | [PyTorch[A]](https://github.com/arunmallya/packnet) |
| `CVPR`  | [NISP: Pruning Networks Using Neuron Importance Score Propagation](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_NISP_Pruning_Networks_CVPR_2018_paper.html) | `S` | |
| `ICIP`  | [Online Filter Clustering and Pruning for Efficient Convnets](https://ieeexplore.ieee.org/abstract/document/8451123) | `S` |  |
| `IJCAI` | [Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://www.ijcai.org/proceedings/2018/0309.pdf) | `S` | [PyTorch[A]](https://github.com/he-y/soft-filter-pruning) | 

<h3 align="center">2017</h3>

| Venue | Title | Type | Code |
|:-----|:------|:-----:|:----:|
| `CVPR`  | [Designing Energy-Efficient Convolutional Neural Networks Using Energy-Aware Pruning](https://openaccess.thecvf.com/content_cvpr_2017/html/Yang_Designing_Energy-Efficient_Convolutional_CVPR_2017_paper.html) | `S` | |
| `ICLR`  | [Pruning Filters for Efficient ConvNets](https://openreview.net/forum?id=rJqFGTslg) | `S` | [PyTorch[O]](doc/PFEC.md) |
| `ICCV`  | [Channel Pruning for Accelerating Very Deep Neural Networks](https://openaccess.thecvf.com/content_iccv_2017/html/He_Channel_Pruning_for_ICCV_2017_paper.html) | `S` | [PyTorch[A]](https://github.com/yihui-he/channel-pruning) |
| `ICCV`  | [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://openaccess.thecvf.com/content_iccv_2017/html/Luo_ThiNet_A_Filter_ICCV_2017_paper.html) | `S` | [Caffe[A]](https://github.com/Roll920/ThiNet) |
| `ICCV`  | [Learning Efficient Convolutional Networks Through Network Slimming](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) | `S` | [PyTorch[A]](https://github.com/Eric-mingjie/network-slimming) |


## Journal Publications 

<h3 align="center">2024</h3>

| Journal | Title | Type | Code |
|:-----|:------|:----:|:----:|
| [IEEE Transactions on Artificial Intelligence](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=76) | [Distilled Gradual Pruning with Pruned Fine-tuning](https://ieeexplore.ieee.org/document/10438214) | `U` | [PyTorch[A]](https://github.com/rom42pla/dg2pf) | 

<h3 align="center">2023</h3>

| Journal | Title | Type | Code |
|:-----|:------|:----:|:----:|
| [IEEE Trans Circuits Syst Video Technol](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=76) | [DCFP: Distribution Calibrated Filter Pruning for Lightweight and Accurate Long-tail Semantic Segmentation](https://ieeexplore.ieee.org/abstract/document/10364745) | `S` | | 
| [IEEE Internet Things J.](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6488907) | [SNPF: Sensitiveness Based Network Pruning Framework for Efficient Edge Computing](https://ieeexplore.ieee.org/document/10250769) | `S` | | 
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Manipulating Identical Filter Redundancy for Efficient Pruning on Deep and Complicated CNN](https://ieeexplore.ieee.org/document/10283855) | `S` | | 
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Block-Wise Partner Learning for Model Compression](https://ieeexplore.ieee.org/document/10237122) | `S` | [PyTorch[A]](https://github.com/zhangxin-xd/BPL)| 
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Hierarchical Threshold Pruning Based on Uniform Response Criterion](https://ieeexplore.ieee.org/document/10103912) | `S` | | 
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [CATRO: Channel Pruning via Class-Aware Trace Ratio Optimization](https://ieeexplore.ieee.org/document/10094002) | `S` | | 
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Adaptive Filter Pruning via Sensitivity Feedback](https://ieeexplore.ieee.org/document/10064249) | `S` | | 
| [Neurocomputing](https://www.sciencedirect.com/journal/neurocomputing) | [Filter pruning with uniqueness mechanism in the frequency domain for efficient neural networks](https://www.sciencedirect.com/science/article/pii/S0925231223001364) | `S` | |
| [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | [Compact Neural Network via Stacking Hybrid Units](https://ieeexplore.ieee.org/abstract/document/10275036) | `S` | |
| [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | [Performance-aware Approximation of Global Channel Pruning for Multitask CNNs](https://arxiv.org/abs/2303.11923) | `S` | [PyTorch[A]](https://github.com/HankYe/PAGCP/tree/main) |
| [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | [Adaptive Search-and-Training for Robust and Efficient Network Pruning](https://ieeexplore.ieee.org/abstract/document/10052756) | `S` | |
| [Image Vis. Comput.](https://www.sciencedirect.com/journal/image-and-vision-computing) | [Loss-aware automatic selection of structured pruning criteria for deep neural network acceleration](https://www.sciencedirect.com/science/article/pii/S0262885623001191) | `S` | [PyTorch[A]](https://github.com/ghimiredhikura/laasp) |
| [Comput. Vis. Image Underst.](https://www.sciencedirect.com/journal/computer-vision-and-image-understanding) | [Feature independent Filter Pruning by Successive Layers analysis](https://www.sciencedirect.com/science/article/pii/S1077314223002084) | `S` |  |
| [IEEE Access](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6287639) | [Differentiable Neural Architecture, Mixed Precision and Accelerator Co-Search](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10266308) | `S` |  |

<h3 align="center">2022</h3>

| Journal | Title | Type | Code |
|:-----|:------|:----:|:----:|
| [IEEE Trans. Image Process.](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=83) | [Efficient Layer Compression Without Pruning](https://ieeexplore.ieee.org/abstract/document/10214522) | `S` | |
| [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | [Learning to Explore Distillability and Sparsability: A Joint Framework for Model Compression](https://ieeexplore.ieee.org/abstract/document/9804342) | `S` | |
| [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | [1xN Pattern for Pruning Convolutional Neural Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9847369) | `S` | [PyTorch[A]](https://github.com/lmbxmu/1xN) |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Filter Pruning by Switching to Neighboring CNNs With Good Attribute](https://ieeexplore.ieee.org/document/9716788) | `S` |  |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Model Pruning Enables Efficient Federated Learning on Edge Devices](https://ieeexplore.ieee.org/abstract/document/9762360) | `S` |  |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [DAIS: Automatic Channel Pruning via Differentiable Annealing Indicator Search](https://ieeexplore.ieee.org/document/9749778) | `S` |  |
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Network Pruning Using Adaptive Exemplar Filters](https://ieeexplore.ieee.org/document/9448300) | `S` | [PyTorch[A]](https://github.com/lmbxmu/EPruner) | 
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Carrying Out CNN Channel Pruning in a White Box](https://ieeexplore.ieee.org/document/9712474) | `S` | [PyTorch[A]](https://github.com/zyxxmu/White-Box) | 
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Pruning Networks With Cross-Layer Ranking & k-Reciprocal Nearest Filters](https://ieeexplore.ieee.org/document/9737040) | `S` | [PyTorch[A]](https://github.com/lmbxmu/CLR-RNF) | 
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Filter Sketch for Network Pruning](https://ieeexplore.ieee.org/document/9454340) | `S` | [PyTorch[A]](https://github.com/lmbxmu/FilterSketch) |
| [Neurocomputing](https://www.sciencedirect.com/journal/neurocomputing) | [FPFS: Filter-level pruning via distance weight measuring filter similarity](https://www.sciencedirect.com/science/article/pii/S092523122201164X) | `S` | |
| [Neurocomputing](https://www.sciencedirect.com/journal/neurocomputing) | [RUFP: Reinitializing unimportant filters for soft pruning](https://www.sciencedirect.com/science/article/pii/S0925231222001667) | `S` |  |
| [Neural Netw](https://www.sciencedirect.com/journal/neural-networks) | [HRel: Filter pruning based on High Relevance between activation maps and class labels](https://www.sciencedirect.com/science/article/pii/S0893608021004962) | `S` | [PyTorch[A]*](https://github.com/sarvanichinthapalli/HRel) |
| [Comput. Intell. Neurosci.](https://www.hindawi.com/journals/cin/) | [Differentiable Network Pruning via Polarization of Probabilistic Channelwise Soft Masks](https://www.hindawi.com/journals/cin/2022/7775419/) | `S` | | 
| [J. Syst. Archit.](https://www.sciencedirect.com/journal/journal-of-systems-architecture) | [Optimizing deep neural networks on intelligent edge accelerators via flexible-rate filter pruning](https://www.sciencedirect.com/science/article/pii/S1383762122000303) | `S` | | 
| [Appl. Sci.](https://www.mdpi.com/journal/applsci) | [Magnitude and Similarity Based Variable Rate Filter Pruning for Efficient Convolution Neural Networks](https://www.mdpi.com/2076-3417/13/1/316) | `S` | [PyTorch[A]](https://github.com/ghimiredhikura/MSVFP-FilterPruning) | 
| [Sensors](https://www.mdpi.com/journal/sensors) | [Filter Pruning via Measuring Feature Map Information](https://www.mdpi.com/1424-8220/21/19/6601) | `S` |  |
| [IEEE Access](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6287639) | [Automated Filter Pruning Based on High-Dimensional Bayesian Optimization](https://ieeexplore.ieee.org/document/9718082) | `S` |  | 
| [IEEE Signal Process. Lett.](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=97)  | [A Low-Complexity Modified ThiNet Algorithm for Pruning Convolutional Neural Networks](https://ieeexplore.ieee.org/document/9748003) | `S` | |

<h3 align="center">2021</h3>

| Journal | Title | Type | Code |
|:-----|:------|:----:|:----:|
| [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34)  | [Discrimination-Aware Network Pruning for Deep Model Compression](https://ieeexplore.ieee.org/document/9384353) | `S` | [PyTorch[A]~](https://github.com/SCUT-AILab/DCP)|

<h3 align="center">2020</h3>  

| Journal | Title | Type | Code |
|:-----|:------|:----:|:----:|
| [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [EDP: An Efficient Decomposition and Pruning Scheme for Convolutional Neural Network Compression](https://ieeexplore.ieee.org/document/9246734) | `S` | | 
| [IEEE Access](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6287639) | [Filter Pruning Without Damaging Networks Capacity](https://ieeexplore.ieee.org/document/9091183) | `S` |  | 
| [Electronics](https://www.mdpi.com/journal/electronics) | [Pruning Convolutional Neural Networks with an Attention Mechanism for Remote Sensing Image Classification](https://www.mdpi.com/2079-9292/9/8/1209) | `S` |  | 

## Survey Articles
| Year  | Venue | Title |
|:------:|:-----:|:-----|
| `2023` | [`Artif. Intell. Rev.`](https://www.springer.com/journal/10462) | [Deep neural network pruning method based on sensitive layers and reinforcement learning](https://link.springer.com/article/10.1007/s10462-023-10566-5) | 
| `2023` | `arVix` | [A Survey on Deep Neural Network Pruning: Taxonomy, Comparison, Analysis, and Recommendations](https://arxiv.org/abs/2308.06767) | 
| `2023` | `arVix` | [Structured Pruning for Deep Convolutional Neural Networks: A survey](https://arxiv.org/abs/2303.00566) | 
| `2022` | [`Electronics`](https://www.mdpi.com/journal/electronics) | [A Survey on Efficient Convolutional Neural Networks and Hardware Acceleration](https://www.mdpi.com/2079-9292/11/6/945) | 
| `2022` | [`I-SMAC`](https://i-smac.org/ismac2022/) | [A Survey on Filter Pruning Techniques for Optimization of Deep Neural Networks](https://ieeexplore.ieee.org/abstract/document/9987264/) |
| `2021` | [`JMLR`](https://jmlr.csail.mit.edu/) | [Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks](https://www.jmlr.org/papers/volume22/21-0366/21-0366.pdf) |
| `2021` | [`Neurocomputing`](https://www.sciencedirect.com/journal/neurocomputing) | [Pruning and quantization for deep neural network acceleration: A survey](https://www.sciencedirect.com/science/article/pii/S0925231221010894) |
| `2020` | [`IEEE Access`](https://ieeexplore.ieee.org/xpl/aboutJournal.jsp?punumber=6287639) | [Methods for Pruning Deep Neural Networks](https://ieeexplore.ieee.org/document/9795013/) | 

## Other Publications
| Year  | Venue | Title | Code |
|:------:|:-----:|:-----|:------:|
| `2023` | `arVix` | [Why is the State of Neural Network Pruning so Confusing? On the Fairness, Comparison Setup, and Trainability in Network Pruning](https://arxiv.org/abs/2301.05219) | [PyTorch[A]](https://github.com/mingsun-tse/why-the-state-of-pruning-so-confusing)(soon...) | 
| `2023` | `arVix` |[Ten Lessons We Have Learned in the New "Sparseland": A Short Handbook for Sparse Neural Network Researchers](https://arxiv.org/abs/2302.02596) |  | 
| `2022` | `ICML` | **Tutorial** -- [Sparsity in Deep Learning: Pruning and growth for efficient inference and training](https://icml.cc/virtual/2021/tutorial/10845) |  |  

## Pruning Software and Toolbox
|Year | Title | Type | Code |
|:----:|:------|:----:|:----:|
|`2023` | **[Torch-Pruning](https://arxiv.org/abs/2301.12900)** | `S` | [PyTorch[A]](https://github.com/VainF/Torch-Pruning) |
|`2023` | [JaxPruner: JaxPruner: A concise library for sparsity research](https://arxiv.org/abs/2304.14082) | `U/S` | [PyTorch[A]](https://github.com/google-research/jaxpruner) |
|`2022` | [FasterAI: Prune and Distill your models with FastAI and PyTorch](https://nathanhubens.github.io/fasterai/) | `U` | [PyTorch[A]](https://github.com/nathanhubens/fasterai) |
|`2022` | [Simplify: A Python library for optimizing pruned neural networks](https://www.sciencedirect.com/science/article/pii/S2352711021001576) | | [PyTorch[A]](https://github.com/EIDOSlab/simplify) |
|`2021` | [DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900) | `S` | [PyTorch[A]](https://github.com/VainF/Torch-Pruning) |
|`2021` | PyTorchViz [A small package to create visualizations of PyTorch execution graphs] | | [PyTorch[A]](https://github.com/szagoruyko/pytorchviz) |
|`2020` | [What is the State of Neural Network Pruning?](https://proceedings.mlsys.org/paper/2020/file/d2ddea18f00665ce8623e36bd4e3c7c5-Paper.pdf) | `S/U` | [PyTorch[A]](https://github.com/jjgo/shrinkbench) |
|`2019` | [Official PyTorch Pruning Tool](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) | `S/U` | [PyTorch[A]](https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/prune.py) |
