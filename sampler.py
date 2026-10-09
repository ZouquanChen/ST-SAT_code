import random
import numpy as np

def partition_data(args, y_train):
    n_train = y_train.shape[0]
    if args.partition in ('noniid-c-dir', 'iid-diff-quantity') and n_train < 10 * args.n_parties:
        raise ValueError('This partition requires at least 10 training samples per client')
    partition = args.partition
    n_parties = args.n_parties
    alpha = args.alpha
    K = args.num_classes
    if partition == 'iid':
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
    elif partition == 'noniid-c-dir':
        min_size = 0
        min_require_size = 10
        N = n_train
        net_dataidx_map = {}
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for (p, idx_j) in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for (idx_j, idx) in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    elif partition > 'noniid-cnum-0' and partition <= 'noniid-cnum-9':
        num = int(partition.split('-')[2])
        times = [0 for _ in range(K)]
        contain = []
        for i in range(n_parties):
            current = [i % K]
            times[i % K] += 1
            j = 1
            while j < num:
                ind = random.randint(0, K - 1)
                if ind not in current:
                    j = j + 1
                    current.append(ind)
                    times[ind] += 1
            contain.append(current)
        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i in range(K):
            idx_k = np.where(y_train == i)[0]
            np.random.shuffle(idx_k)
            if times[i] == 0:
                continue
            split = np.array_split(idx_k, times[i])
            ids = 0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                    ids += 1
        for i in range(n_parties):
            net_dataidx_map[i] = net_dataidx_map[i].tolist()
    elif partition == 'iid-diff-quantity':
        print('quantity skew: clients with different size of iid samples')
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * len(idxs))
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs, proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
    else:
        raise NotImplementedError('Unknown imbalance type')
    return net_dataidx_map
