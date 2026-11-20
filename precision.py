import os
import torch


def supports_precision_api():
    try:
        torch.backends.cuda.matmul.fp32_precision
    except (AttributeError, AssertionError):
        # Older PyTorch raises AssertionError, rather than AttributeError.
        return False
    return True


def configure_fp32():
    # Set before creating CUDA/cuBLAS handles; never inherit a forced TF32 opt-in.
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
    torch.set_default_dtype(torch.float32)
    if supports_precision_api():
        # PyTorch 2.9+: do not mix this API with the legacy allow_tf32 flags.
        torch.backends.fp32_precision = 'ieee'
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.fp32_precision = 'ieee'
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
        torch.backends.cudnn.rnn.fp32_precision = 'ieee'
        controls = {
            'api': 'fp32_precision',
            'matmul': torch.backends.cuda.matmul.fp32_precision,
            'convolution': torch.backends.cudnn.conv.fp32_precision,
        }
        assert controls['matmul'] == controls['convolution'] == 'ieee'
    else:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        controls = {
            'api': 'allow_tf32',
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
        }
        assert not controls['matmul_allow_tf32'] and not controls['cudnn_allow_tf32']
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return controls
