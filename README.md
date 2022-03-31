# Awasome-Pruning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

### Awasome Research Papers and other Resources in Neural Network Pruning. This collection is prepared inspired by [he-y/Awesome-Pruning](https://github.com/he-y/Awesome-Pruning).

| Notation  | Explanation | 
|:----------|:----------- |
|   `U`       |*Unstructured, Weight Pruning* |
|   `S`       |*Structured, Filter/Channel Pruning* |
|   `A`       |*Official/Author Implementation* |  
|   `3rd Party`|*Unofficial/3rd Party Implementation* |  

| Year   | Venue | Title | Type | Code |
|:------:|:-----:|:------|:----:|:----:|
| `2022`   | `ICLR`  | [Revisit Kernel Pruning with Lottery Regulated Grouped Convolutions](https://openreview.net/forum?id=LdEhiMG9WLO) | `S` | [PyTorch-A ](https://github.com/choH/lottery_regulated_grouped_kernel_pruning) | 
| `2022`   | `ICLR`  | [Dual Lottery Ticket Hypothesis](https://openreview.net/forum?id=fOsN52jn25l) | `U` |[PyTorch ](https://github.com/yueb17/DLTH) | 
| `2020`   | `MLSys`  | [Shrinkbench: What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033) | | [PyTorch ](https://github.com/JJGO/shrinkbench) |
| `2019`   | `ICLR` | [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://openreview.net/forum?id=rJl-b3RcF7) | `U` | [PyTorch](https://github.com/facebookresearch/open_lth) |
| `2019`   | `CVPR` | [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.html) | `S` |[PyTorch ](https://github.com/he-y/filter-pruning-geometric-median) | 
| `2018`   | `IJCAI` | [Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://www.ijcai.org/proceedings/2018/0309.pdf) | `S` | [PyTorch ](https://github.com/he-y/soft-filter-pruning) | 
| `2018`   | `ICIP` | [Online Filter Clustering and Pruning for Efficient Convnets](https://ieeexplore.ieee.org/abstract/document/8451123) | `S` |  |
| `2017`   | `ICLR` | [Pruning Filters for Efficient ConvNets](https://openreview.net/forum?id=rJqFGTslg) | `S` | [PyTorch](doc/PFEC.md) |


### Other

| Title | Type | Code |
|:------ |:------: |:-----:|
| Toolbox: Pruning channels for model acceleration | `S/U` | [PyTorch](https://github.com/VainF/Torch-Pruning) |
