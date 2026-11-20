from config import parse_args

def main():
    args = parse_args()
    import os
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
    from trainer import run
    run(args)
if __name__ == '__main__':
    main()
