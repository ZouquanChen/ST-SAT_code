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

## KD Gradient Weighting

`-st_kd_grad_weighting` (also `--st-kd-grad-weighting`) selects `none` or `cosine`.
The default is `none`, which preserves the original fixed-weight loss and its
addition order. Enable cosine weighting with:

```powershell
python main.py -data_dir PATH_TO_CIFAR10 -st_kd_grad_weighting cosine
```

With the default KD coefficients, the losses are:

```text
none:   CE + Reg + 0.1 * MissingKD + 0.03 * ExistingKD + 0.1 * FeatureMSE
cosine: CE + Reg + 0.1 * MissingKD + 0.03 * existing_gate * ExistingKD
                + 0.1 * feature_gate * FeatureMSE
```

The two gates are computed independently for each minibatch as the detached
cosine between pure CE gradients and the corresponding scaled KD gradients,
clipped to `[0, 1]`. All trainable parameters, including the final classifier,
are included; unused gradients count as zero. Conflicting or zero gradients
receive a zero gate. Disabled KD terms skip their gradient probes.
Missing-class KD and regularization retain their original coefficients.
Feature KD is scaled by `st_lambda_feature` only, without an extra `lamda`.

The constraint guarantees a nonnegative full-model gradient inner product with
CE for each gated term. It is not a per-layer guarantee or a guarantee about
the actual Adam/momentum update.

Each local epoch logs `mode`, `existing_gate`, `feature_gate`, and
`effective_distill` (the minibatch mean of KD after all coefficients and gates).
The existing `Distill` field retains its original unscaled meaning.
BN behavior, neuron settings, and aggregation momenta retain their existing
defaults. Changing this option takes effect when a new training process starts.

Run the checks with `python -m unittest test_kd_weighting -v`.

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
