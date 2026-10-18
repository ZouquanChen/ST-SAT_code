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

## Fixed-Weight Distillation

The default `-st_distill_mode fixed` uses fixed class and feature distillation
coefficients. With the default settings, the loss is:

```text
CE + Reg + 0.1 * MissingKD + 0.03 * ExistingKD + 0.1 * FeatureMSE
```

Class KL terms use the full-class softmax and are weighted before summation.
Feature MSE is multiplied by `st_lambda_feature` only.

## Optional Dynamic B Distillation

Select the original B strategy with `-st_distill_mode dynamic_b` (or
`--st-distill-mode dynamic_b`). The original `models/kd_weighting.py` is copied
unchanged from the `B_KD_4GPU_linux_20261017` source snapshot.

```text
existing_gate = stop_gradient(clamp(cos(grad(CE), grad(ExistingKL)), 0, 1))
reference = CE + MissingKL + existing_gate * ExistingKL
feature_gate = stop_gradient(clamp(cos(grad(reference), grad(FeatureMSE)), 0, 1))
loss = CE + Reg + MissingKL + existing_gate * ExistingKL + feature_gate * FeatureMSE
```

The gradients cover all trainable parameters, including the classifier. The
feature reference excludes regularization and feature MSE; its cosine is used
directly, without an extra CE minimum. In this mode `lamda`,
`st_no_exist_label_weight`, `st_exist_label_weight` and `st_lambda_feature` are
enable switches: zero disables the corresponding term; positive values enable
it without an extra static multiplier. `lamda` controls both class KL terms;
feature KD is enabled independently. Missing KL therefore has coefficient 1
when enabled. Both modes retain full-class softmax KL, feature firing-rate MSE,
teacher evaluation mode, neuron settings and aggregation settings.

```bash
# Existing fixed-weight training (also the default when the mode is omitted)
python main.py -data_dir PATH_TO_CIFAR10 -st_distill_mode fixed

# Original dynamic B training
python main.py -data_dir PATH_TO_CIFAR10 -st_distill_mode dynamic_b
```

Dynamic B logs go under `log_dir/dynamic_b/`; checkpoint, CSV and metrics names
include `_dynamic_b`, so they do not overwrite fixed-mode outputs. Metrics
record `distillation_mode`. Each dynamic local epoch logs mean `existing_gate`,
`feature_gate` and `effective_distill`. The dynamic `Distill` value is the
effective KD loss; fixed mode retains its original unscaled diagnostic value.

Run both test suites with
`python -m unittest test_distillation test_dynamic_distillation -v`.

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
