from config import parse_args

def main():
    args = parse_args()
    from trainer import run
    run(args)
if __name__ == '__main__':
    main()
