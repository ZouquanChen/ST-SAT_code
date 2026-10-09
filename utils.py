import os
import datetime
import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import functional
INPUT_SIZE = {'cifar10': (3, 32, 32)}
CLASS_NUM = {'cifar10': 10}

def load_data(args):
    train = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True)
    test = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True)
    return (train.data, np.array(train.targets), test.data, np.array(test.targets))

class CIFAR10Dataset(Dataset):

    def __init__(self, images, labels, indices=None):
        self.data = images if indices is None else images[indices]
        self.target = labels if indices is None else labels[indices]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.transform(self.data[index]), self.target[index])

def get_dataloader(Xs, ys, train_batch_size, test_batch_size=32, dataidxs=None):
    train_ds = CIFAR10Dataset(Xs[0], ys[0], dataidxs)
    test_ds = CIFAR10Dataset(Xs[1], ys[1])
    train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, drop_last=False)
    return (train_dl, test_dl, train_ds, test_ds)

def ensure_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur

class Logger:

    def __init__(self, args, desc='fl'):
        log_root = args.log_dir
        if getattr(args, 'simple_log', False):
            ensure_dir(log_root)
            index = 1
            while True:
                name = 'record.log' if index == 1 else f'record_{index}.log'
                logfilepath = os.path.join(log_root, name)
                try:
                    f = open(logfilepath, 'x', encoding='utf-8')
                except FileExistsError:
                    index += 1
                    continue
                break
        else:
            act = 'snn' if args.snn else 'ann'
            out_dir = os.path.join(log_root, f'{args.dataset}_{args.strategy}_{args.model}_{act}_T{args.T}_b{args.batch_size}_lr{args.lr}_{args.global_epochs}GE_{args.local_epochs}LE_{args.n_parties}party_frac{args.frac}')
            ensure_dir(out_dir)
            logfilename = f'{desc}_{args.partition}_alpha{args.alpha:g}_seed{args.init_seed}_record.log'
            logfilepath = os.path.join(out_dir, logfilename)
            f = open(logfilepath, 'w', encoding='utf-8')
        self.filename = logfilepath
        with f:
            f.write(str(args) + '\n')

    def info(self, s=None):
        print(s)
        f = open(self.filename, 'a', encoding='utf-8')
        f.write(f'[{get_local_time()}] - {s}\n')
        f.flush()
        f.close()

    def debug(self, s=None):
        self.info(f'[DEBUG] {s}')

    def warning(self, s=None):
        self.info(f'[WARNING] {s}')

def compute_accuracy(model, dataloader, device='cpu', repeat_num=4, logger=None, label='Evaluation', log_every=50):
    device = torch.device(device)
    if logger is not None and device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    if logger is not None:
        logger.info(f'[Eval] {label}: start, samples={len(dataloader.dataset)}, batches={len(dataloader)}')
    was_training = model.training
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    model.eval()
    try:
        functional.reset_net(model)
        with torch.no_grad():
            for (batch_idx, (x, target)) in enumerate(dataloader, 1):
                (x, target) = (x.to(device), target.to(device))
                x = x.unsqueeze(0).repeat(repeat_num, 1, 1, 1, 1)
                out = model(x)
                correct += (out.argmax(dim=1) == target).sum()
                total += target.shape[0]
                functional.reset_net(model)
                if logger is not None and log_every > 0 and (batch_idx % log_every == 0):
                    logger.info(f'[Eval] {label}: batches={batch_idx}/{len(dataloader)}, samples={total}')
        if total == 0:
            raise ValueError('Cannot evaluate an empty dataloader')
        accuracy = correct.item() / float(total)
    finally:
        functional.reset_net(model)
        model.train(was_training)
    if logger is not None:
        logger.info(f'[Eval] {label}: acc={accuracy:.4f}, samples={total}, time={time.perf_counter() - started:.2f}s')
    return accuracy
