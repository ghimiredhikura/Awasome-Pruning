# Awasome Pruning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Awasome Resources in Deep Neural Network Pruning. This collection is prepared inspired by [he-y/Awesome-Pruning](https://github.com/he-y/Awesome-Pruning).

| Symbol | `U` | `S` | `A`| `O`| 
|:---------:|:-----------:|:-----------:|:-----------:|:-----------:|
| **Details** |Unstructured or Weight Pruning|Structured or Filter or Channel Pruning|Official or Author Implementation|Unofficial or 3rd Party Implementation|  

## Conference

| Year   | Venue | Title | Type | Code |
|:------:|:-----|:------|:-----:|:----:|
| `2023`   | `ICLR`  | [OTOv2: Automatic, Generic, User-Friendly](https://openreview.net/forum?id=7ynoX1ojPMt) | `S` | [PyTorch[A]](https://github.com/tianyic/only_train_once) |
| `2023`   | `ICLR`  | [How I Learned to Stop Worrying and Love Retraining](https://openreview.net/forum?id=_nF5imFKQI) | `U` | [PyTorch[A]](https://github.com/ZIB-IOL/BIMP) |
| `2023`   | `ICLR`  | [Token Merging: Your ViT But Faster ](https://openreview.net/forum?id=JroZRaRw7Eu) | `U/S` | [PyTorch[A]](https://github.com/facebookresearch/ToMe) |
| `2023`   | `ICLR`  | [Revisiting Pruning at Initialization Through the Lens of Ramanujan Graphs](https://openreview.net/forum?id=uVcDssQff_) | `U` | [PyTorch[A]](https://github.com/VITA-Group/ramanujan-on-pai) (soon...) |
| `2023`   | `ICLR`  | [Unmasking the Lottery Ticket Hypothesis: What's Encoded in a Winning Ticket's Mask?](https://openreview.net/forum?id=xSsW2Am-ukZ) | `U` |  |
| `2023`   | `ICLR`  | [NTK-SAP: Improving neural network pruning by aligning training dynamics](https://openreview.net/forum?id=-5EWhW_4qWP) | `U` |  |
| `2023`   | `ICLR`  | [DFPC: Data flow driven pruning of coupled channels without data](https://openreview.net/forum?id=mhnHqRqcjYU) | `S` | [PyTorch[A]](https://drive.google.com/drive/folders/18eRYzWnB_6Qq0cYiSzvyOgicqn50g3-m) |
| `2023`   | `ICLR`  | [Pruning Deep Neural Networks from a Sparsity Perspective](https://openreview.net/forum?id=i-DleYh34BM) | `U` | [PyTorch[A]](https://openreview.net/attachment?id=i-DleYh34BM&name=supplementary_material) |
| `2023`   | `ICLR`  | [A Unified Framework of Soft Threshold Pruning](https://openreview.net/forum?id=cCFqcrq0d8) | `U` | [PyTorch[A]](https://openreview.net/attachment?id=cCFqcrq0d8&name=supplementary_material) |
| `2023`   | `WACV`  | [Calibrating Deep Neural Networks Using Explicit Regularisation and Dynamic Data Pruning](https://openaccess.thecvf.com/content/WACV2023/html/Patra_Calibrating_Deep_Neural_Networks_Using_Explicit_Regularisation_and_Dynamic_Data_WACV_2023_paper.html) | `S` | |
| `2023`   | `WACV`  | [Attend Who Is Weak: Pruning-Assisted Medical Image Localization Under Sophisticated and Implicit Imbalances](https://openaccess.thecvf.com/content/WACV2023/html/Jaiswal_Attend_Who_Is_Weak_Pruning-Assisted_Medical_Image_Localization_Under_Sophisticated_WACV_2023_paper.html) | `S` | |
| `2023`   | [`ICASSP`](https://2023.ieeeicassp.org/important-dates/)  | [WHC: Weighted Hybrid Criterion for Filter Pruning on Convolutional Neural Networks](https://arxiv.org/abs/2302.08185) | `S` | [PyTorch[A]](https://github.com/ShaowuChen/WHC) |
| `2022`   | `CVPR`  | [Interspace Pruning: Using Adaptive Filter Representations To Improve Training of Sparse CNNs](https://openaccess.thecvf.com/content/CVPR2022/html/Wimmer_Interspace_Pruning_Using_Adaptive_Filter_Representations_To_Improve_Training_of_CVPR_2022_paper.html) | `U` | |
| `2022`   | `CVPR`  | [Revisiting Random Channel Pruning for Neural Network Compression](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Revisiting_Random_Channel_Pruning_for_Neural_Network_Compression_CVPR_2022_paper.html) | `S` | [PyTorch[A]](https://github.com/ofsoundof/random_channel_pruning) (soon...) |
| `2022`   | `CVPR`  | [Fire Together Wire Together: A Dynamic Pruning Approach With Self-Supervised Mask Prediction](https://openaccess.thecvf.com/content/CVPR2022/html/Elkerdawy_Fire_Together_Wire_Together_A_Dynamic_Pruning_Approach_With_Self-Supervised_CVPR_2022_paper.html) | `S` | [PyTorch[A]](https://github.com/selkerdawy/FTWT) |
| `2022`   | `CVPR`  | [When to Prune? A Policy towards Early Structural Pruning](https://openaccess.thecvf.com/content/CVPR2022/html/Shen_When_To_Prune_A_Policy_Towards_Early_Structural_Pruning_CVPR_2022_paper.html) | `S` | |
| `2022`   | `CVPR`  | [Dreaming to Prune Image Deraining Networks](https://openaccess.thecvf.com/content/CVPR2022/papers/Zou_Dreaming_To_Prune_Image_Deraining_Networks_CVPR_2022_paper.pdf) | `S` | |
| `2022`   | `ICLR`  | [SOSP: Efficiently Capturing Global Correlations by Second-Order Structured Pruning](https://openreview.net/forum?id=t5EmXZ3ZLR) | `S` | |
| `2022`   | `ICLR`  | [Learning Pruning-Friendly Networks via Frank-Wolfe: One-Shot, Any-Sparsity, And No Retraining](https://openreview.net/forum?id=O1DEtITim__) | `U` | [PyTorch[A]](https://github.com/VITA-Group/SFW-Once-for-All-Pruning) |
| `2022`   | `ICLR`  | [Revisit Kernel Pruning with Lottery Regulated Grouped Convolutions](https://openreview.net/forum?id=LdEhiMG9WLO) | `S` | [PyTorch[A]](https://github.com/choH/lottery_regulated_grouped_kernel_pruning) | 
| `2022`   | `ICLR`  | [Dual Lottery Ticket Hypothesis](https://openreview.net/forum?id=fOsN52jn25l) | `U` |[PyTorch[A]](https://github.com/yueb17/DLTH) |
| `2022`   | `NIPS`  | [SAViT: Structure-Aware Vision Transformer Pruning via Collaborative Optimization](https://openreview.net/forum?id=w5DacXWzQ-Q) | `S` |[PyTorch[A]](https://github.com/hikvision-research/SAViT)(soon...) |
| `2022`   | `NIPS`  | [Structural Pruning via Latency-Saliency Knapsack](https://openreview.net/forum?id=cUOR-_VsavA) | `S` |[PyTorch[A]](https://github.com/NVlabs/HALP)|
| `2022`   | `ACCV`  | [Filter Pruning via Automatic Pruning Rate Search⋆](https://openaccess.thecvf.com/content/ACCV2022/html/Sun_Filter_Pruning_via_Automatic_Pruning_Rate_Search_ACCV_2022_paper.html) | `S` | |
| `2022`   | `ACCV`  | [Network Pruning via Feature Shift Minimization](https://openaccess.thecvf.com/content/ACCV2022/html/Duan_Network_Pruning_via_Feature_Shift_Minimization_ACCV_2022_paper.html) | `S` | [PyTorch[A]](https://github.com/lscgx/FSM) |
| `2022`   | `ACCV`  | [Lightweight Alpha Matting Network Using Distillation-Based Channel Pruning](https://openaccess.thecvf.com/content/ACCV2022/html/Yoon_Lightweight_Alpha_Matting_Network_Using_Distillation-Based_Channel_Pruning_ACCV_2022_paper.html) | `S` | [PyTorch[A]](https://github.com/DongGeun-Yoon/DCP) |
| `2022`   | `ACCV`  | [Adaptive FSP : Adaptive Architecture Search with Filter Shape Pruning](https://openaccess.thecvf.com/content/ACCV2022/html/Kim_Adaptive_FSP__Adaptive_Architecture_Search_with_Filter_Shape_Pruning_ACCV_2022_paper.html) | `S` | |
| `2022`   | `ECCV`  | [Soft Masking for Cost-Constrained Channel Pruning](https://link.springer.com/chapter/10.1007/978-3-031-20083-0_38) | `S` | [PyTorch[A]](https://github.com/NVlabs/SMCP) |
| `2022`   | `WACV`  | [Hessian-Aware Pruning and Optimal Neural Implant](https://openaccess.thecvf.com/content/WACV2022/papers/Yu_Hessian-Aware_Pruning_and_Optimal_Neural_Implant_WACV_2022_paper.pdf) | `S` | [PyTorch[A]](https://github.com/yaozhewei/HAP) |
| `2022`   | `WACV`  | [PPCD-GAN: Progressive Pruning and Class-Aware Distillation for Large-Scale Conditional GANs Compression](https://openaccess.thecvf.com/content/WACV2022/papers/Vo_PPCD-GAN_Progressive_Pruning_and_Class-Aware_Distillation_for_Large-Scale_Conditional_GANs_WACV_2022_paper.pdf) | `S` |  |
| `2022`   | `WACV`  | [Channel Pruning via Lookahead Search Guided Reinforcement Learning](https://openaccess.thecvf.com/content/WACV2022/papers/Wang_Channel_Pruning_via_Lookahead_Search_Guided_Reinforcement_Learning_WACV_2022_paper.pdf) | `S` |  |
| `2022`   | `WACV`  | [EZCrop: Energy-Zoned Channels for Robust Output Pruning](https://openaccess.thecvf.com/content/WACV2022/papers/Lin_EZCrop_Energy-Zoned_Channels_for_Robust_Output_Pruning_WACV_2022_paper.pdf) | `S` | [PyTorch[A]](https://github.com/rlin27/EZCrop) |
| `2022`   | `ICIP`  | [One-Cycle Pruning: Pruning Convnets With Tight Training Budget](https://ieeexplore.ieee.org/document/9897980) | `U` | |
| `2022`   | `ICIP`  | [RAPID: A Single Stage Pruning Framework](https://ieeexplore.ieee.org/document/9898000) | `U` | |
| `2022`   | `ICIP`  | [The Rise of the Lottery Heroes: Why Zero-Shot Pruning is Hard](https://ieeexplore.ieee.org/document/9897223) | `U` | |
| `2022`   | `ICIP`  | [Truncated Lottery Ticket for Deep Pruning](https://ieeexplore.ieee.org/document/9897767) | `U` | |
| `2022`   | `ICIP`  | [Which Metrics For Network Pruning: Final Accuracy? or Accuracy Drop?](https://ieeexplore.ieee.org/document/9898051) | `S/U` | |
| `2022`   | `ISMSI` | [Structured Pruning with Automatic Pruning Rate Derivation for Image Processing Neural Networks](https://dl.acm.org/doi/abs/10.1145/3533050.3533066) | `S` | |
| `2021`   | `ICLR`  | [Neural Pruning via Growing Regularization](https://openreview.net/forum?id=o966_Is_nPA) | `S` | [PyTorch[A]](https://github.com/mingsun-tse/regularization-pruning) |
| `2021`   | `ICLR`  | [Network Pruning That Matters: A Case Study on Retraining Variants](https://openreview.net/forum?id=Cb54AMqHQFP) | `S` | [PyTorch[A]](https://github.com/lehduong/NPTM) |
| `2021`   | `ICLR`  | [Layer-adaptive Sparsity for the Magnitude-based Pruning](https://openreview.net/forum?id=H6ATjJ0TKdf) | `U` | [PyTorch[A]](https://github.com/jaeho-lee/layer-adaptive-sparsity) |
| `2021`   | `NIPS`  | [Only Train Once: A One-Shot Neural Network Training And Pruning Framework](https://openreview.net/forum?id=p5rMPjrcCZq) | `S` | [PyTorch[A]](https://github.com/tianyic/only_train_once) |
| `2021`   | `CVPR`  | [NPAS: A Compiler-Aware Framework of Unified Network Pruning and Architecture Search for Beyond Real-Time Mobile Acceleration](https://openaccess.thecvf.com/content/CVPR2021/html/Li_NPAS_A_Compiler-Aware_Framework_of_Unified_Network_Pruning_and_Architecture_CVPR_2021_paper.html) | `S` | |
| `2021`   | `CVPR`  | [Network Pruning via Performance Maximization](https://openaccess.thecvf.com/content/CVPR2021/html/Gao_Network_Pruning_via_Performance_Maximization_CVPR_2021_paper.html) | `S` | |
| `2021`   | `CVPR`  | [Convolutional Neural Network Pruning With Structural Redundancy Reduction*](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Convolutional_Neural_Network_Pruning_With_Structural_Redundancy_Reduction_CVPR_2021_paper.html) | `S` | |
| `2021`   | `CVPR`  | [Manifold Regularized Dynamic Network Pruning](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Manifold_Regularized_Dynamic_Network_Pruning_CVPR_2021_paper.html) | `S` | [PyTorch[A]](https://github.com/yehuitang/Pruning/tree/master/ManiDP) |
| `2021`   | `CVPR`  | [Joint-DetNAS: Upgrade Your Detector With NAS, Pruning and Dynamic Distillation](https://openaccess.thecvf.com/content/CVPR2021/html/Yao_Joint-DetNAS_Upgrade_Your_Detector_With_NAS_Pruning_and_Dynamic_Distillation_CVPR_2021_paper.html) | `S` | |
| `2021`   | `ICCV`  | [ResRep: Lossless CNN Pruning via Decoupling Remembering and Forgetting](https://openaccess.thecvf.com/content/ICCV2021/html/Ding_ResRep_Lossless_CNN_Pruning_via_Decoupling_Remembering_and_Forgetting_ICCV_2021_paper.html) | `S` | |
| `2021`   | `ICCV`  | [Achieving On-Mobile Real-Time Super-Resolution With Neural Architecture and Pruning Search](https://openaccess.thecvf.com/content/ICCV2021/html/Zhan_Achieving_On-Mobile_Real-Time_Super-Resolution_With_Neural_Architecture_and_Pruning_Search_ICCV_2021_paper.html) | `S` | |
| `2021`   | `ICCV`  | [GDP: Stabilized Neural Network Pruning via Gates With Differentiable Polarization*](https://openaccess.thecvf.com/content/ICCV2021/html/Guo_GDP_Stabilized_Neural_Network_Pruning_via_Gates_With_Differentiable_Polarization_ICCV_2021_paper.html) | `S` | |
| `2021`   | `WACV`  | [Holistic Filter Pruning for Efficient Deep Neural Networks](https://openaccess.thecvf.com/content/WACV2021/html/Enderich_Holistic_Filter_Pruning_for_Efficient_Deep_Neural_Networks_WACV_2021_paper.html) | `S` | |
| `2021`   | `ICML`  | [Accelerate CNNs from Three Dimensions: A Comprehensive Pruning Framework](https://icml.cc/virtual/2021/poster/9081) | `S` | |
| `2021`   | `ICML`  | [Group Fisher Pruning for Practical Network Compression](https://icml.cc/virtual/2021/poster/9875) | `S` | [PyTorch[A]](https://github.com/jshilong/FisherPruning) |
| `2020`   | `CVPR`  | [HRank: Filter Pruning using High-Rank Feature Map](https://openaccess.thecvf.com/content_CVPR_2020/html/Lin_HRank_Filter_Pruning_Using_High-Rank_Feature_Map_CVPR_2020_paper.html) | `S` | [PyTorch[A]](https://github.com/lmbxmu/HRank) |
| `2020`   | `CVPR`  | [Towards efficient model compression via learned global ranking](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chin_Towards_Efficient_Model_Compression_via_Learned_Global_Ranking_CVPR_2020_paper.pdf) | `S` | [PyTorch[A]](https://github.com/enyac-group/LeGR) |
| `2020`   | `CVPR`  | [Learning Filter Pruning Criteria for Deep Convolutional Neural Networks Acceleration](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Learning_Filter_Pruning_Criteria_for_Deep_Convolutional_Neural_Networks_Acceleration_CVPR_2020_paper.html) | `S` | |
| `2020`   | `CVPR`  | [Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Group_Sparsity_The_Hinge_Between_Filter_Pruning_and_Decomposition_for_CVPR_2020_paper.html) | `S` | [PyTorch[A]](https://github.com/ofsoundof/group_sparsity) |
| `2020`   | `CVPR`  | [APQ: Joint Search for Network Architecture, Pruning and Quantization Policy](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_APQ_Joint_Search_for_Network_Architecture_Pruning_and_Quantization_Policy_CVPR_2020_paper.html) | `S` | [PyTorch[A]](https://github.com/mit-han-lab/apq) |
| `2020`   | `ICLR`  | [Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints](https://openreview.net/forum?id=HyxLRTVKPH) | `U` | |
| `2020`   | `MLSys` | [Shrinkbench: What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033) | | [PyTorch[A]](https://github.com/JJGO/shrinkbench) |
| `2020`   | `BMBS`  | [Similarity Based Filter Pruning for Efficient Super-Resolution Models](https://ieeexplore.ieee.org/abstract/document/9379712) | `S` |  |
| `2019`   | `CVPR`  | [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.html) | `S` |[PyTorch[A]](https://github.com/he-y/filter-pruning-geometric-median) |
| `2019`   | `CVPR`  | [Variational Convolutional Neural Network Pruning](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Variational_Convolutional_Neural_Network_Pruning_CVPR_2019_paper.html) | `S` | |
| `2019`   | `CVPR`  | [Towards Optimal Structured CNN Pruning via Generative Adversarial Learning](https://openaccess.thecvf.com/content_CVPR_2019/html/Lin_Towards_Optimal_Structured_CNN_Pruning_via_Generative_Adversarial_Learning_CVPR_2019_paper.html) | `S` | [PyTorch[A]](https://github.com/ShaohuiLin/GAL) |
| `2019`   | `CVPR`  | [Partial Order Pruning: For Best Speed/Accuracy Trade-Off in Neural Architecture Search](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Partial_Order_Pruning_For_Best_SpeedAccuracy_Trade-Off_in_Neural_Architecture_CVPR_2019_paper.html) | `S` | [PyTorch[A]](https://github.com/lixincn2015/Partial-Order-Pruning) |
| `2019`   | `CVPR`  | [Importance Estimation for Neural Network Pruning](https://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html) | `S` | [PyTorch[A]](https://github.com/NVlabs/Taylor_pruning) |
| `2019`   | `ICLR`  | [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://openreview.net/forum?id=rJl-b3RcF7) | `U` | [PyTorch[A]](https://github.com/facebookresearch/open_lth) |
| `2019`   | `ICLR`  | [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://openreview.net/forum?id=B1VZqjAcYX) | `U` | [Tensorflow[A]](https://github.com/namhoonlee/snip-public) |
| `2019`   | `ICCV`  | [MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_MetaPruning_Meta_Learning_for_Automatic_Neural_Network_Channel_Pruning_ICCV_2019_paper.html) | `S` | [PyTorch[A]](https://github.com/liuzechun/MetaPruning) |
| `2019`   | `ICCV`  | [Accelerate CNN via Recursive Bayesian Pruning](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Accelerate_CNN_via_Recursive_Bayesian_Pruning_ICCV_2019_paper.html) | `S` |  |
| `2018`   | `CVPR`  | [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://openaccess.thecvf.com/content_cvpr_2018/html/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.html) | `S` | [PyTorch[A]](https://github.com/arunmallya/packnet) |
| `2018`   | `CVPR`  | [NISP: Pruning Networks Using Neuron Importance Score Propagation](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_NISP_Pruning_Networks_CVPR_2018_paper.html) | `S` | |
| `2018`   | `ICIP`  | [Online Filter Clustering and Pruning for Efficient Convnets](https://ieeexplore.ieee.org/abstract/document/8451123) | `S` |  |
| `2018`   | `IJCAI` | [Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://www.ijcai.org/proceedings/2018/0309.pdf) | `S` | [PyTorch[A]](https://github.com/he-y/soft-filter-pruning) | 
| `2017`   | `CVPR`  | [Designing Energy-Efficient Convolutional Neural Networks Using Energy-Aware Pruning](https://openaccess.thecvf.com/content_cvpr_2017/html/Yang_Designing_Energy-Efficient_Convolutional_CVPR_2017_paper.html) | `S` | |
| `2017`   | `ICLR`  | [Pruning Filters for Efficient ConvNets](https://openreview.net/forum?id=rJqFGTslg) | `S` | [PyTorch[O]](doc/PFEC.md) |
| `2017`   | `ICCV`  | [Channel Pruning for Accelerating Very Deep Neural Networks](https://openaccess.thecvf.com/content_iccv_2017/html/He_Channel_Pruning_for_ICCV_2017_paper.html) | `S` | [PyTorch[A]](https://github.com/yihui-he/channel-pruning) |
| `2017`   | `ICCV`  | [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://openaccess.thecvf.com/content_iccv_2017/html/Luo_ThiNet_A_Filter_ICCV_2017_paper.html) | `S` | [Caffe[A]](https://github.com/Roll920/ThiNet) |
| `2017`   | `ICCV`  | [Learning Efficient Convolutional Networks Through Network Slimming](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) | `S` | [PyTorch[A]](https://github.com/Eric-mingjie/network-slimming) |


## Journal 

| Year  | Journal | Title | Type | Code |
|:------:|:-----|:------|:----:|:----:|
| `2023` | [Neurocomputing](https://www.sciencedirect.com/journal/neurocomputing) | [Filter pruning with uniqueness mechanism in the frequency domain for efficient neural networks](https://www.sciencedirect.com/science/article/pii/S0925231223001364) | `S` | |
| `2023` | [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | [Performance-aware Approximation of Global Channel Pruning for Multitask CNNs](https://arxiv.org/abs/2303.11923) | `S` | [PyTorch[A]](https://github.com/HankYe/PAGCP/tree/main) |
| `2023` | [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | [Adaptive Search-and-Training for Robust and Efficient Network Pruning](https://ieeexplore.ieee.org/abstract/document/10052756) | `S` | |
| `2022` | [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | [Learning to Explore Distillability and Sparsability: A Joint Framework for Model Compression](https://ieeexplore.ieee.org/abstract/document/9804342) | `S` | |
| `2022` | [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | [1xN Pattern for Pruning Convolutional Neural Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9847369) | `S` | [PyTorch[A]](https://github.com/lmbxmu/1xN) |
| `2022` | [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Filter Pruning by Switching to Neighboring CNNs With Good Attribute](https://ieeexplore.ieee.org/document/9716788) | `S` |  |
| `2022` | [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Model Pruning Enables Efficient Federated Learning on Edge Devices](https://ieeexplore.ieee.org/abstract/document/9762360) | `S` |  |
| `2022`   | [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Network Pruning Using Adaptive Exemplar Filters](https://ieeexplore.ieee.org/document/9448300) | `S` | [PyTorch[A]](https://github.com/lmbxmu/EPruner) | 
| `2022`   | [IEEE Trans. NNLS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | [Filter Sketch for Network Pruning](https://ieeexplore.ieee.org/document/9454340) | `S` | [PyTorch[A]](https://github.com/lmbxmu/FilterSketch) |
| `2022` | [Neurocomputing](https://www.sciencedirect.com/journal/neurocomputing) | [FPFS: Filter-level pruning via distance weight measuring filter similarity](https://www.sciencedirect.com/science/article/pii/S092523122201164X) | `S` | |
| `2022` | [Neurocomputing](https://www.sciencedirect.com/journal/neurocomputing) | [RUFP: Reinitializing unimportant filters for soft pruning](https://www.sciencedirect.com/science/article/pii/S0925231222001667) | `S` |  |
| `2022`   | [Neural Netw](https://www.sciencedirect.com/journal/neural-networks) | [HRel: Filter pruning based on High Relevance between activation maps and class labels](https://www.sciencedirect.com/science/article/pii/S0893608021004962) | `S` | [PyTorch[A]*](https://github.com/sarvanichinthapalli/HRel) |
| `2022` | [Comput. Intell. Neurosci.](https://www.hindawi.com/journals/cin/) | [Differentiable Network Pruning via Polarization of Probabilistic Channelwise Soft Masks](https://www.hindawi.com/journals/cin/2022/7775419/) | `S` | | 
| `2022`   | [J. Syst. Archit.](https://www.sciencedirect.com/journal/journal-of-systems-architecture) | [Optimizing deep neural networks on intelligent edge accelerators via flexible-rate filter pruning](https://www.sciencedirect.com/science/article/pii/S1383762122000303) | `S` | | 
| `2022`   | [Appl. Sci.](https://www.mdpi.com/journal/applsci) | [Magnitude and Similarity Based Variable Rate Filter Pruning for Efficient Convolution Neural Networks](https://www.mdpi.com/2076-3417/13/1/316) | `S` | [PyTorch[A]](https://github.com/ghimiredhikura/MSVFP-FilterPruning) | 
| `2022`   | [Sensors](https://www.mdpi.com/journal/sensors) | [Filter Pruning via Measuring Feature Map Information](https://www.mdpi.com/1424-8220/21/19/6601) | `S` |  |
| `2022`   | [IEEE Access](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6287639) | [Automated Filter Pruning Based on High-Dimensional Bayesian Optimization](https://ieeexplore.ieee.org/document/9718082) | `S` |  | 
| `2022`   | [IEEE Signal Process. Lett.](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=97)  | [A Low-Complexity Modified ThiNet Algorithm for Pruning Convolutional Neural Networks](https://ieeexplore.ieee.org/document/9748003) | `S` | |
| `2021`   | [IEEE Trans. PAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34)  | [Discrimination-Aware Network Pruning for Deep Model Compression](https://ieeexplore.ieee.org/document/9384353) | `S` | [PyTorch[A]~](https://github.com/SCUT-AILab/DCP)|
| `2020`   | [IEEE Access](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6287639) | [Filter Pruning Without Damaging Networks Capacity](https://ieeexplore.ieee.org/document/9091183) | `S` |  | 
| `2020`   | [Electronics](https://www.mdpi.com/journal/electronics) | [Pruning Convolutional Neural Networks with an Attention Mechanism for Remote Sensing Image Classification](https://www.mdpi.com/2079-9292/9/8/1209) | `S` |  | 

## Survery Papers 
| Year  | Venue | Title |
|:------:|:-----:|:-----|
| `2023` | `arVix` | [Structured Pruning for Deep Convolutional Neural Networks: A survey](https://arxiv.org/abs/2303.00566) | 
| `2022` | `Electronics` | [A Survey on Efficient Convolutional Neural Networks and Hardware Acceleration](https://www.mdpi.com/2079-9292/11/6/945) | 
| `2022` | [`I-SMAC`](https://i-smac.org/ismac2022/) | [A Survey on Filter Pruning Techniques for Optimization of Deep Neural Networks](https://ieeexplore.ieee.org/abstract/document/9987264/) |
| `2021` | [`JMLR`](https://jmlr.csail.mit.edu/) | [Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks](https://www.jmlr.org/papers/volume22/21-0366/21-0366.pdf) |
| `2021` | `Neurocomputing`| [Pruning and quantization for deep neural network acceleration: A survey](https://www.sciencedirect.com/science/article/pii/S0925231221010894) |
| `2020` | `IEEE Access` | [Methods for Pruning Deep Neural Networks](https://ieeexplore.ieee.org/document/9795013/) | 

## Other 
| Year  | Venue | Title | Code |
|:------:|:-----:|:-----|:------:|
| `2023` | `arVix` | [Why is the State of Neural Network Pruning so Confusing? On the Fairness, Comparison Setup, and Trainability in Network Pruning](https://arxiv.org/abs/2301.05219) | [PyTorch[A]](https://github.com/mingsun-tse/why-the-state-of-pruning-so-confusing)(soon...) | 
| `2023` | `arVix` |[Ten Lessons We Have Learned in the New "Sparseland": A Short Handbook for Sparse Neural Network Researchers](https://arxiv.org/abs/2302.02596) |  | 
| `2022` | `ICML` | **Tutorial** -- [Sparsity in Deep Learning: Pruning and growth for efficient inference and training](https://icml.cc/virtual/2021/tutorial/10845) |  |  


## Pruning Software/Toolbox
|Year | Title | Type | Code |
|:----:|:------|:----:|:----:|
|`2022` | [FasterAI: Prune and Distill your models with FastAI and PyTorch](https://nathanhubens.github.io/fasterai/) | `U` | [PyTorch[A]](https://github.com/nathanhubens/fasterai) |
|`2022` | [Simplify: A Python library for optimizing pruned neural networks](https://www.sciencedirect.com/science/article/pii/S2352711021001576) | | [PyTorch[A]](https://github.com/EIDOSlab/simplify) |
|`2021` | [DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900) | `S` | [PyTorch[A]](https://github.com/VainF/Torch-Pruning) |
|`2021` | PyTorchViz [A small package to create visualizations of PyTorch execution graphs] | | [PyTorch[A]](https://github.com/szagoruyko/pytorchviz) |
|`2020` | [What is the State of Neural Network Pruning?](https://proceedings.mlsys.org/paper/2020/file/d2ddea18f00665ce8623e36bd4e3c7c5-Paper.pdf) | `S/U` | [PyTorch[A]](https://github.com/jjgo/shrinkbench) |
|`2019` | [Official PyTorch Pruning Tool](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) | `S/U` | [PyTorch[A]](https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/prune.py) |
