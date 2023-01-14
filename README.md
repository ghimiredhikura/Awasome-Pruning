# Awasome-CNNPruning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Awasome Research Papers and other Resources in Convolution Neural Network Pruning. This collection is prepared inspired by [he-y/Awesome-Pruning](https://github.com/he-y/Awesome-Pruning).

| Notation  | Explanation | 
|:---------:|:----------- |
|   `U`     |*Unstructured, Weight Pruning* |
|   `S`     |*Structured, Filter/Channel Pruning* |
|   `A`     |*Official/Author Implementation* |  
|   `O`     |*Unofficial/3rd Party Implementation* |  

## Conference

| Year   | Venue | Title | Type | Code |
|:------:|:-----|:------|:----:|:----:|
| `2023`   | `ICLR`  | [How I Learned to Stop Worrying and Love Retraining](https://openreview.net/forum?id=_nF5imFKQI) | `U` | [PyTorch[A]](https://openreview.net/attachment?id=_nF5imFKQI&name=supplementary_material) |
| `2023`   | `ICLR`  | [Revisiting Pruning at Initialization Through the Lens of Ramanujan Graphs](https://openreview.net/forum?id=uVcDssQff_) | `U` |  |
| `2022`   | `CVPR`  | [Interspace Pruning: Using Adaptive Filter Representations To Improve Training of Sparse CNNs](https://openaccess.thecvf.com/content/CVPR2022/html/Wimmer_Interspace_Pruning_Using_Adaptive_Filter_Representations_To_Improve_Training_of_CVPR_2022_paper.html) | `U` | |
| `2022`   | `CVPR`  | [Revisiting Random Channel Pruning for Neural Network Compression](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Revisiting_Random_Channel_Pruning_for_Neural_Network_Compression_CVPR_2022_paper.html) | `S` | [PyTorch[A]](https://github.com/ofsoundof/random_channel_pruning) |
| `2022`   | `CVPR`  | [Fire Together Wire Together: A Dynamic Pruning Approach With Self-Supervised Mask Prediction](https://openaccess.thecvf.com/content/CVPR2022/html/Elkerdawy_Fire_Together_Wire_Together_A_Dynamic_Pruning_Approach_With_Self-Supervised_CVPR_2022_paper.html) | `S` | |
| `2022`   | `CVPR`  | [When to Prune? A Policy towards Early Structural Pruning](https://openaccess.thecvf.com/content/CVPR2022/papers/Shen_When_To_Prune_A_Policy_Towards_Early_Structural_Pruning_CVPR_2022_paper.pdf) | `S` | |
| `2022`   | `CVPR`  | [Dreaming to Prune Image Deraining Networks](https://openaccess.thecvf.com/content/CVPR2022/papers/Zou_Dreaming_To_Prune_Image_Deraining_Networks_CVPR_2022_paper.pdf) | `S` | |
| `2022`   | `ICLR`  | [SOSP: Efficiently Capturing Global Correlations by Second-Order Structured Pruning](https://openreview.net/forum?id=t5EmXZ3ZLR) | `S` | |
| `2022`   | `ICLR`  | [Learning Pruning-Friendly Networks via Frank-Wolfe: One-Shot, Any-Sparsity, And No Retraining](https://openreview.net/forum?id=O1DEtITim__) | `U` | [PyTorch[A]](https://github.com/VITA-Group/SFW-Once-for-All-Pruning) |
| `2022`   | `ICLR`  | [Revisit Kernel Pruning with Lottery Regulated Grouped Convolutions](https://openreview.net/forum?id=LdEhiMG9WLO) | `S` | [PyTorch[A]](https://github.com/choH/lottery_regulated_grouped_kernel_pruning) | 
| `2022`   | `ICLR`  | [Dual Lottery Ticket Hypothesis](https://openreview.net/forum?id=fOsN52jn25l) | `U` |[PyTorch[A]](https://github.com/yueb17/DLTH) |
| `2022`   | `ACCV`  | [Filter Pruning via Automatic Pruning Rate Search⋆](https://openaccess.thecvf.com/content/ACCV2022/html/Sun_Filter_Pruning_via_Automatic_Pruning_Rate_Search_ACCV_2022_paper.html) | `S` | |
| `2022`   | `ACCV`  | [Network Pruning via Feature Shift Minimization](https://openaccess.thecvf.com/content/ACCV2022/html/Duan_Network_Pruning_via_Feature_Shift_Minimization_ACCV_2022_paper.html) | `S` | [PyTorch[A]](https://github.com/lscgx/FSM) |
| `2022`   | `ACCV`  | [Lightweight Alpha Matting Network Using Distillation-Based Channel Pruning](https://openaccess.thecvf.com/content/ACCV2022/html/Yoon_Lightweight_Alpha_Matting_Network_Using_Distillation-Based_Channel_Pruning_ACCV_2022_paper.html) | `S` | [PyTorch[A]](https://github.com/DongGeun-Yoon/DCP) |
| `2022`   | `ACCV`  | [Adaptive FSP : Adaptive Architecture Search with Filter Shape Pruning](https://openaccess.thecvf.com/content/ACCV2022/html/Kim_Adaptive_FSP__Adaptive_Architecture_Search_with_Filter_Shape_Pruning_ACCV_2022_paper.html) | `S` | |
| `2022`   | `WACV`  | [Hessian-Aware Pruning and Optimal Neural Implant](https://openaccess.thecvf.com/content/WACV2022/papers/Yu_Hessian-Aware_Pruning_and_Optimal_Neural_Implant_WACV_2022_paper.pdf) | `S` | [PyTorch[A]](https://github.com/yaozhewei/HAP) |
| `2022`   | `WACV`  | [PPCD-GAN: Progressive Pruning and Class-Aware Distillation for Large-Scale Conditional GANs Compression](https://openaccess.thecvf.com/content/WACV2022/papers/Vo_PPCD-GAN_Progressive_Pruning_and_Class-Aware_Distillation_for_Large-Scale_Conditional_GANs_WACV_2022_paper.pdf) | `S` |  |
| `2022`   | `WACV`  | [Channel Pruning via Lookahead Search Guided Reinforcement Learning](https://openaccess.thecvf.com/content/WACV2022/papers/Wang_Channel_Pruning_via_Lookahead_Search_Guided_Reinforcement_Learning_WACV_2022_paper.pdf) | `S` |  |
| `2022`   | `WACV`  | [EZCrop: Energy-Zoned Channels for Robust Output Pruning](https://openaccess.thecvf.com/content/WACV2022/papers/Lin_EZCrop_Energy-Zoned_Channels_for_Robust_Output_Pruning_WACV_2022_paper.pdf) | `S` | [PyTorch[a]](https://github.com/rlin27/EZCrop) |
| `2022`   | `ICIP`  | [A Low-Complexity Modified ThiNet Algorithm for Pruning Convolutional Neural Networks](https://ieeexplore.ieee.org/document/9748003) | `S` | |
| `2022`   | `ICIP`  | [One-Cycle Pruning: Pruning Convnets With Tight Training Budget](https://ieeexplore.ieee.org/document/9897980) | `U` | |
| `2022`   | `ICIP`  | [RAPID: A Single Stage Pruning Framework](https://ieeexplore.ieee.org/document/9898000) | `U` | |
| `2022`   | `ICIP`  | [The Rise of the Lottery Heroes: Why Zero-Shot Pruning is Hard](https://ieeexplore.ieee.org/document/9897223) | `U` | |
| `2022`   | `ICIP`  | [Truncated Lottery Ticket for Deep Pruning](https://ieeexplore.ieee.org/document/9897767) | `U` | |
| `2022`   | `ICIP`  | [Which Metrics For Network Pruning: Final Accuracy? or Accuracy Drop?](https://ieeexplore.ieee.org/document/9898051) | `S/U` | |
| `2022`   | `ISMSI`  | [Structured Pruning with Automatic Pruning Rate Derivation for Image Processing Neural Networks](https://dl.acm.org/doi/abs/10.1145/3533050.3533066) | `S` | |
| `2021`   | `NIPS` | [Only Train Once: A One-Shot Neural Network Training And Pruning Framework](https://openreview.net/forum?id=p5rMPjrcCZq) | `S` | [PyTorch[A]](https://github.com/tianyic/only_train_once) |
| `2020`   | `CVPR`  | [HRank: Filter Pruning using High-Rank Feature Map](https://openaccess.thecvf.com/content_CVPR_2020/html/Lin_HRank_Filter_Pruning_Using_High-Rank_Feature_Map_CVPR_2020_paper.html) | `S` | [PyTorch[A]](https://github.com/lmbxmu/HRank) |
| `2020`   | `CVPR`  | [Towards efficient model compression via learned global ranking](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chin_Towards_Efficient_Model_Compression_via_Learned_Global_Ranking_CVPR_2020_paper.pdf) | `S` | [PyTorch[A]](https://github.com/enyac-group/LeGR) |
| `2020`   | `ICLR` | [Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints](https://openreview.net/forum?id=HyxLRTVKPH) | `U` | |
| `2020`   | `MLSys` | [Shrinkbench: What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033) | | [PyTorch[A]](https://github.com/JJGO/shrinkbench) |
| `2020`   | `BMBS`  | [Similarity Based Filter Pruning for Efficient Super-Resolution Models](https://ieeexplore.ieee.org/abstract/document/9379712) | `S` |  |
| `2019`   | `CVPR` | [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.html) | `S` |[PyTorch[A]](https://github.com/he-y/filter-pruning-geometric-median) |
| `2019`   | `ICLR` | [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://openreview.net/forum?id=rJl-b3RcF7) | `U` | [PyTorch[A]](https://github.com/facebookresearch/open_lth) |
| `2019`   | `ICLR` | [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://openreview.net/forum?id=B1VZqjAcYX) | `U` | [Tensorflow[A]](https://github.com/namhoonlee/snip-public) |
| `2018`   | `ICIP` | [Online Filter Clustering and Pruning for Efficient Convnets](https://ieeexplore.ieee.org/abstract/document/8451123) | `S` |  |
| `2018`   | `IJCAI` | [Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://www.ijcai.org/proceedings/2018/0309.pdf) | `S` | [PyTorch[A]](https://github.com/he-y/soft-filter-pruning) | 
| `2017`   | `ICLR` | [Pruning Filters for Efficient ConvNets](https://openreview.net/forum?id=rJqFGTslg) | `S` | [PyTorch[O]](doc/PFEC.md) |


## Journal 

| Year  | Journal | Title | Type | Code |
|:------:|:-----|:------|:----:|:----:|
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
| `2020`   | [IEEE Access](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6287639) | [Filter Pruning Without Damaging Networks Capacity](https://ieeexplore.ieee.org/document/9091183) | `S` |  | 
| `2020`   | [Electronics](https://www.mdpi.com/journal/electronics) | [Pruning Convolutional Neural Networks with an Attention Mechanism for Remote Sensing Image Classification](https://www.mdpi.com/2079-9292/9/8/1209) | `S` |  | 

### Pruning Software/Toolbox

|Year | Venue | Title | Type | Code |
|:----:|:---|:------|:----:|:----:|
|`2022` | [FasterAI](https://nathanhubens.github.io/fasterai/) | [FasterAI: Prune and Distill your models with FastAI and PyTorch](https://nathanhubens.github.io/fasterai/) | `U` | [PyTorch[A]](https://github.com/nathanhubens/fasterai) |
|`2022` | [SoftwareX](https://www.sciencedirect.com/journal/softwarex) | [Simplify: A Python library for optimizing pruned neural networks](https://www.sciencedirect.com/science/article/pii/S2352711021001576) | | [PyTorch[A]](https://github.com/EIDOSlab/simplify) |
|`2021` | [github.com](https://github.com/VainF/Torch-Pruning) | Toolbox: Structural Pruning for Model Acceleration | `S` | [PyTorch[A]](https://github.com/VainF/Torch-Pruning) |
|`2021` | [github.com](https://github.com/szagoruyko/pytorchviz) | Toolbox: PyTorchViz [A small package to create visualizations of PyTorch execution graphs] | | [PyTorch[A]](https://github.com/szagoruyko/pytorchviz) |
|`2020` | `MLSys` | [What is the State of Neural Network Pruning?](https://proceedings.mlsys.org/paper/2020/file/d2ddea18f00665ce8623e36bd4e3c7c5-Paper.pdf) | `S/U` | [PyTorch[A]](https://github.com/jjgo/shrinkbench) |
|`2019` | [PyTorch](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) | [Official PyTorch Pruning Tool](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) | `S/U` | [PyTorch[A]](https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/prune.py) |