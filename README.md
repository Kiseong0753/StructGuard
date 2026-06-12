## Stake the Points: Structure-Faithful Instance Unlearning [CVPR 2026]

**[Stake the Points: Structure-Faithful Instance Unlearning](https://openaccess.thecvf.com/content/CVPR2026/papers/Hong_Stake_the_Points_Structure-Faithful_Instance_Unlearning_CVPR_2026_paper.pdf)**  
*Kiseong Hong, JungKyoo Shin, Eunwoo Kim*  
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026


![StructGuard Overview](./Overview.png)


## Abstract
Machine unlearning (MU) addresses privacy risks in pretrained models. The main goal of MU is to remove the influence of designated data while preserving the utility of retained knowledge. Achieving this goal requires preserving semantic relations among retained instances, which existing studies often overlook. We observe that without such preservation, models suffer from progressive structural collapse, undermining both the deletion–retention balance. In this work, we propose a novel structure-faithful framework that introduces stakes, i.e., semantic anchors that serve as reference points to maintain the knowledge structure. By leveraging these anchors, our framework captures and stabilizes the semantic organization of knowledge. Specifically, we instantiate the anchors from language-driven attribute descriptions encoded by a semantic encoder (e.g., CLIP). We enforce preservation of the knowledge structure via structure-aware alignment and regularization: the former aligns the organization of retained knowledge before and after unlearning around anchors, while the latter regulates updates to structure-critical parameters. Results from image classification, retrieval, and face recognition show average gains of 32.9%, 22.5%, and 19.3% in performance, balancing the deletion–retention trade-off and enhancing generalization.


---
