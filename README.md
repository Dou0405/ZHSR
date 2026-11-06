# ZHSR

The code for **Zero-shot Blind Hyperspectral Image Super-resolution via UnMixing-inspired Deep Quasi-Newton Network**.

## 1. Introduction

This work introduces a Zero-shot Blind Hyperspectral Image Super-Resolution (ZHSR) framework. An UnMixing-inspired Super-Resolution (UMSR) model represents spectral mixing through shared endmembers and spatial abundances. Its L-BFGS solution is unfolded into DeepQN-Net, which learns from a single observed low-resolution HSI without pretrained weights or a known blur kernel.

* **Interpretability:** Quasi-Newton updates, physical operators and abundance priors form the unfolding stages.
* **Zero-shot learning:** Internal training pairs are constructed from the target observation.
* **Unified processing:** Every observation uses the same model and training-pair generation, without scenario-dependent branches.

**Data:** [ZHSR-datasets-CAVE-Butterfly-PaviaC-Chikusei-Hyperdrive-Matiwan-LN01](https://pan.baidu.com/s/11H5sV_xWvl1RBCyAvosUVA) (Baidu Netdisk). Please contact the author at [yongxuan@buaa.edu.cn](mailto:yongxuan@buaa.edu.cn) to obtain the extraction code. 

## 2. File Structure

* **Core Files:**
  * `models.py`: DeepQN-Net architecture.
  * `modules.py`: UMSR operators, QNS, AUM, GS, curvature buffer, scattering predictor and reconstruction module.
  * `loss.py`: Rehearsal, lookahead and adaptation objectives.
* **Utilities:**
  * `utils/data.py`: Explicit cube loading and internal training pairs.
  * `utils/metrics.py`: PSNR, SSIM, SAM (degrees) and ERGAS.
  * `utils/initialization.py`: SPA from the observed LR only.
* **Scripts:**
  * `learn.py`: Learning, evaluation and result saving.


## 3. How to Reproduce

Install Python 3.9+ and PyTorch for your CUDA environment, then:

```bash
pip install -r requirements.txt
```
* **Defaults:** 60 rehearsal epochs, learning rate `0.001`, seed `101`, rank `5`, three BS stages, and memory size `3`. Each epoch uses 12 observed crops and 12 synthetic pairs. Batch size is 1.
* **Outputs:** Full-precision epoch logs, configuration, initial/final weights and reconstruction MAT files.
* **Support:** For any issue, feel free to open an issue or contact yongxuan@buaa.edu.cn.

## 4. Citation

Bibliographic details will be updated upon publication.

```bibtex
@misc{dou_zhsr,
  title={Zero-shot Blind Hyperspectral Image Super-resolution via UnMixing-inspired Deep Quasi-Newton Network},
  author={Dou, Yongxuan and Xu, Zilu and Deng, Xin and Xu, Mai},
  year={2026}
}
```
