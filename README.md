# TSMamba: Multi-drone feature interaction via temporal-spatial mamba networks for aerial object tracking (Information Fusion 2026)

## Abstract
Object tracking in multi-drone videos has attracted increasing attention for its ability to provide richer
target information, particularly in scenarios involving occlusion or background interference. Most existing trackers adopt a template sharing-based framework, using cross-drone target templates for similarity estimation with intra-drone search regions to predict target responses. However, such framework relies solely on templates to spread cross-drone information, neglecting current temporal and
background information. Furthermore, similarity-based multi-drone interaction is highly sensitive
to cross-view variations. To address above issues, this paper proposes TSMamba, a novel tracking
framework that utilizes consistency and complementary information among cross-drone templates
and search features to enhance the responses of challenging samples. Specially, it incorporates a temporal consistency enhance mamba (TCEM) to exploit temporal consistency between target templates
and search regions to improve target distinguishability in complex scenarios. Meanwhile, a spatial
complementary fusion mamba (SCFM) is integrated to capture spatial complementary from different
visible components observed under varying viewpoints to build comprehensive target representations.
Additionally, a localization-aware response map correction (LRMC) is introduced to refine target response maps adaptively, improving robustness under dynamic conditions compared to existing predefined methods. To our knowledge, this is the first work explicitly enhance search features using
cross-drone information and the first to employ vision mamba for multi-drone feature interaction. Experiments on the MDOT and MDMT benchmarks demonstrate that TSMamba outperforms previous
state-of-the-art trackers by 5.0% and 4.0% in accuracy and precision. Moreover, TSMamba exhibits
competitive computational efficiency and strong generalization in real-world applications. Codes and
experimental data will be released at https://github.com/HanJin0326/TSMamba.

## Citation
Please consider citing this paper if you find this work helpful in your research:

```
@article{WU2026104279,
title = {TSMamba: Multi-drone feature interaction via temporal-spatial mamba networks for aerial object tracking},
journal = {Information Fusion},
volume = {133},
pages = {104279},
year = {2026},
doi = {https://doi.org/10.1016/j.inffus.2026.104279},
url = {https://www.sciencedirect.com/science/article/pii/S1566253526001582},
author = {Han Wu and Hao Sun and Kui Liu and Kefeng Ji and Gangyao Kuang}
}
```

If you have any question, please do not hesitate to contact me at wuhan0326@163.com or hhjj20190326 (微信).