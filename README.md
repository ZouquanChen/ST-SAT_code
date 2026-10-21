# ST-SAT

This repository contains the official PyTorch implementation of:

**Decoupling Label Shift and Surrogate Gradient Errors for Robust Federated Spiking Neural Networks**

## Python Dependencies

The following package versions were installed in the server environment:

```text
torch==2.0.1+cu118
torchvision==0.15.2+cu118
numpy==1.23.5
pandas==1.5.3
spikingjelly==0.0.0.0.14
```

## Dynamic B Distillation

ST-SAT directly uses dynamic B. No distillation-mode selector or KD
coefficient arguments are required.

```text
existing_gate = stop_gradient(clamp(cos(grad(CE), grad(ExistingKL)), 0, 1))
reference = CE + MissingKL + existing_gate * ExistingKL
feature_gate = stop_gradient(clamp(cos(grad(reference), grad(FeatureMSE)), 0, 1))
loss = CE + Reg + MissingKL + existing_gate * ExistingKL + feature_gate * FeatureMSE
```

The gates use all trainable parameters, including the classifier. The feature
reference excludes regularization and feature MSE; the composite cosine is
used directly. The core strategy follows the original B implementation in
`B_KD_4GPU_linux_20261017`, with alternative modes removed.

All available distillation terms are included. CE and missing-class KL have
coefficient 1; existing-class KL and feature MSE are weighted only by their
detached dynamic gates. An empty class subset or absent feature contributes
zero. KL uses the full-class softmax, feature MSE uses firing rates, and the
teacher stays in evaluation mode. Neuron regularization coefficients remain
separate from distillation.

```bash
python main.py -data_dir PATH_TO_CIFAR10
```

Logs go under `log_dir/dynamic_b/`. Checkpoint, CSV and metrics names include
`_dynamic_b`, and metrics record `distillation_mode`. Each local epoch logs
mean `existing_gate`, `feature_gate` and `effective_distill`.

Pre-existing checkpoints and accuracy results are not distributed with this
repository. Training still saves new outputs; `results/` and the default
`saved/` directory are ignored by Git.

Run the checks with `python -m unittest test_dynamic_distillation -v`.

## Acknowledgements

This work is built upon the following papers:

```bibtex
@inproceedings{yu2025exploiting,
  title     = {Exploiting Label Skewness for Spiking Neural Networks in Federated Learning},
  author    = {Di, Yu and Changze, Lv and Xin, Du and Linshan, Jiang and Wentao, Tong and Zhenyu, Liao and Shuiguang, Deng},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  year      = {2025}
}

@article{fu2025adaptation,
  title     = {Adaptation and learning of spatio-temporal thresholds in spiking neural networks},
  author    = {Fu, Jiahui and Gou, Shuiping and Wang, Peizhao and Jiao, Licheng and Guo, Zhang and Li, Jisheng and Liu, Rong},
  journal   = {Neurocomputing},
  pages     = {130423},
  year      = {2025},
  publisher = {Elsevier}
}
```
