# from scipy import stats
from collections import deque
import json
import os
from pprint import pprint  
import numpy as np
from scipy.stats import f_oneway, ttest_ind, kruskal
from torch import scalar_tensor
from torch.distributed.distributed_c10d import group
from torch.functional import unique
from sklearn.cluster import KMeans

def check_balancing(probs, labels, n_labels, method='anova'):
    # log_probs = np.log(probs)
    unique_labels = np.unique(labels)
    input_args = [] # #labels arrays, each of num_samples * dims
    pvalues = []
    if len(unique_labels) == 2:
        if n_labels == 2:
            linear_p = np.array([np.log(prob[0] / prob[1]) for prob in probs])
            for i in range(2): 
                input_args.append(linear_p[labels == i])
        else:
            for label in unique_labels:
                input_args.append(probs[labels == label])
        t_statistic, pvalues = ttest_ind(*input_args)
        return t_statistic, np.array(pvalues)
    else:
        # probs = np.log(probs)
        if method == 'krustal':
            pvalues = []
            for i in range(n_labels - 1):
                input_args = []
                for label in unique_labels:
                    if (labels == label).sum() < 100:
                        continue
                    input_args.append(probs[labels == label][:,i])
                if len(input_args) < 2:
                    return 0, np.ones(2)
                statistic, pvalue = kruskal(*input_args)
                pvalues.append(pvalue)
        if method == 'anova':
            for label in unique_labels:
                if (labels == label).sum() < 500:
                    continue
                probs_label = probs[labels == label]
                # input_args.append(np.log(np.divide(probs_label[:,:-1], probs_label[:,-1].reshape(-1, 1))))
                input_args.append(probs_label[:,:-1])
            if len(input_args) < 2:
                return 0, np.ones(2)
            statistic, pvalues = f_oneway(*input_args)
        return statistic, np.array(pvalues)

def blocking_multi(indexes, probs, labels, thr, use_clustering=True, method='anova'):
    queue = deque([indexes])
    blocks = []
    split_count = 0
    # old_split_idx = 0
    n_labels = len(np.unique(labels))
    if n_labels == 2:
        use_clustering = False
    ulabels = np.arange(n_labels).astype(int)
    while queue:
        # print("-------")
        # print(len(queue))
        cur_indexes = queue.popleft().astype(int)
        cur_probs = probs[cur_indexes]
        cur_labels = labels[cur_indexes]
        if len(np.unique(cur_labels)) < 2:
            blocks.append((cur_probs, 1., cur_indexes))
            continue
        f_stats, pvalues = check_balancing(cur_probs, cur_labels, n_labels, method)
#         print("{:.3f}".format(float(check[0])), len(cur_indexes), len(np.unique(cur_labels)), cur_probs)
        # if np.any(pvalues < thr) and split_count < 100: # change to the thr for p_values
        if f_stats > thr: # try the original setting
            # for label in ulabels[pvalues < thr]:
            # label = np.random.choice(ulabels[pvalues < thr])
            if use_clustering:
                cur_ulabels = np.unique(cur_labels)
                cluster_vecs = cur_probs
                # n_clusters = len(cur_ulabels)
                n_clusters = 2
                cluster_func = KMeans(n_clusters, init='k-means++', max_iter=10000).fit(cluster_vecs)
                cluster_ids = cluster_func.labels_
                for i in range(n_clusters):
                    queue.append(cur_indexes[cluster_ids==i])

            else:
                # old_split_idx = (old_split_idx + 1) % len(ulabels)
                # label = ulabels[pvalues < thr][old_split_idx]
                label = ulabels[pvalues < thr][0][0]
                split_dim = int(label)
                split_probs = np.array([prob[split_dim] for prob in cur_probs])
                median = np.median(split_probs)
                sub1_indexes = cur_indexes[split_probs <= median]
                sub2_indexes = cur_indexes[split_probs > median]
                queue.append(sub1_indexes)
                queue.append(sub2_indexes)
            split_count += 1
        else:
            blocks.append((cur_probs, pvalues, cur_indexes))
    return blocks


def blocking_and_assign_new_probs(indexes, probs, labels, thr, method): # thr: in (0, 1). For p value.
    probs = np.array(probs)
    # print(probs[:,:-1].shape)
    # exit()
    labels = np.array(labels)
    blocks = blocking_multi(indexes, probs, labels, thr, method=method)
    blocks = sorted(blocks, key=lambda x: probs[x[-1]].mean())
    block_indexes = []
    block_relax_probs = []
    block_labels = []
    # print('Count\tRelax\tMean\tStd\tSta\tP-value')
    print('Count\tRelax\tP-value')
    for sta, pvalue, cur_indexes in blocks:
        cur_labels = labels[cur_indexes]
        block_indexes.append(cur_indexes)
        relax_prob = np.zeros(probs.shape[1])
        for i in range(probs.shape[1]):
            relax_prob[i] = float((cur_labels==i).sum())/len(cur_labels)
        block_relax_probs.append(relax_prob.tolist())
        # block_relax_probs.append((cur_labels == dim_index).mean())
        block_labels.append(cur_labels)
        print('{}\t{}\t{}'.format(len(cur_indexes), relax_prob, pvalue))
#         print('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(len(cur_indexes), block_relax_probs[-1], probs[cur_indexes].mean(), probs[cur_indexes].std(), sta, pvalue[0]))
    return block_indexes, block_relax_probs, block_labels

def blocking_envs(train_bias_probs, train_y, thr, train_ixs=None, save_folder=None, val_bias_probs=None, val_y=None, method='anova'):
    block_indexes, block_relax_scores, block_labels = blocking_and_assign_new_probs(np.arange(0, len(train_y)), train_bias_probs, train_y, thr, method)
    print('Block Count', len(block_relax_scores))
    group_ix = np.zeros(len(train_y))
    weights = np.ones(len(train_y))

    for i, b_index in enumerate(block_indexes):
        for idx in b_index:
            group_ix[idx] = i
            prop = block_relax_scores[i][int(train_y[idx])]
            if prop > 0.01 and prop < 0.99:
                weights[idx] = 1. / prop
    print("Examples of weights: ", weights[:10])

    d = {
        'group_ix': group_ix.tolist(),
        'weight': weights.tolist(),
        'y': train_y,
        'counts': len(block_relax_scores)
    }

    d_group_only = {
        'group_ix': group_ix.tolist(),
        'y': train_y,
        'counts': len(block_relax_scores)
    }

    if val_bias_probs is not None:
        # get block centers
        block_centers = []
        val_group_ix = []
        val_weights = []
        val_plain_tev_weights = []
        for i, b_index in enumerate(block_indexes):
            int_b_index = np.array(b_index).astype(int)
            block_centers.append(np.array(train_bias_probs)[int_b_index].mean(axis=0))
        block_centers = np.array(block_centers)
        print("Shape of block_centers: ", block_centers.shape)
    
        for idx, probs in enumerate(val_bias_probs):
            gix = int(np.argmin(np.square(np.array(probs).reshape(1, -1) - block_centers).sum(axis=1)))
            val_group_ix.append(gix)
            prop = block_relax_scores[gix][int(val_y[idx])]
            val_plain_tev_weights.append(len(train_y) / len(block_indexes[gix]) / len(block_indexes))
            if prop > 0.01 and prop < 0.99:
                weight = 1. / prop
            else:
                weight = 1.
            val_weights.append(weight)
        d['val_group_ix'] = val_group_ix
        d['val_weight'] = val_weights
        d_group_only['val_group_ix'] = val_group_ix
        d_group_only['val_weight'] = val_plain_tev_weights
    
    if train_ixs is not None:
        d['train_ixs'] = train_ixs
    file_name = os.path.join(save_folder, f'blocking-{method}-{len(block_relax_scores)}.json')
    with open(file_name, 'w') as f:
        json.dump(d, f)
    print("Results are saved to ", file_name)
    group_only_file = os.path.join(save_folder, f'unweight-blocking-{len(block_relax_scores)}.json')
    with open(group_only_file, 'w') as f:
        json.dump(d_group_only, f)
    print("Unweighted groups are saved to ", group_only_file)
    return d, file_name

def blocking_envs_hans(train_bias_org, train_bias_probs, train_bias_y, thr, save_folder, save=False, method='anova'):
    block_indexes, block_relax_scores, block_labels = blocking_and_assign_new_probs(np.arange(0, len(train_bias_y)), train_bias_probs, train_bias_y, thr, method)
    print('Block Count', len(block_relax_scores))
    flatten_block_probs = [None for _ in range(len(train_bias_y))]
    group_idxs = np.zeros(len(train_bias_y))
    gid = 0
    for indexes, relax_score in zip(block_indexes, block_relax_scores):
        for i, idx in enumerate(indexes):
#             prev_prob = train_bias_probs[idx]
            flatten_block_probs[idx] = [relax_score[0], relax_score[1]*relax_score[0], (1.-relax_score[1])*relax_score[0]]
            group_idxs[idx] = gid
        gid += 1
# #             raise Exception()
#             flatten_block_probs[idx] = [relax_score, prev_prob[1], prev_prob[2]]
    flatten_block_probs = np.array(flatten_block_probs)
#     train_bias_probs = flatten_block_probs
#     flatten_block_probs = [None for _ in range(len(train_bias_y))]
    # all_block_count = 0
    
    # for bidx in range(len(block_indexes)):
    #     cur_block_indexes = block_indexes[bidx]
    #     print('----------- {} ------------'.format(bidx))
    #     sub_block_indexes, sub_block_relax_scores, _ = blocking_and_assign_new_probs(cur_block_indexes, train_bias_probs, train_bias_y, thr, 1)
    #     print('Block Count', len(sub_block_indexes))
    #     all_block_count += len(sub_block_indexes)
    #     for indexes, relax_score in zip(sub_block_indexes, sub_block_relax_scores):
    #         for i, idx in enumerate(indexes):
    #             thr = 1-train_bias_probs[idx, 0]
    #             flatten_block_probs[idx] = [train_bias_probs[idx, 0], (relax_score)*thr, (1-relax_score)*thr]
    # flatten_block_probs = np.array(flatten_block_probs)   
    # print(all_block_count)
    
    
#     print(set(tuple(b) for b in flatten_block_probs))
    
    # flatten_block_logprobs = np.log(flatten_block_probs).tolist()
    blocked_probs = flatten_block_probs.tolist()
    group_idxs = group_idxs.astype(int).tolist()
    
    new_bias = {'logits': blocked_probs, 'group': group_idxs, 'id': train_bias_org['id'], 'index': train_bias_org['index'], 'index2token': train_bias_org['index2token'], 'y': train_bias_org['y']}
    # print(new_bias['id'][:10])
    file_name = os.path.join(save_folder, f'blocking-{thr}-{len(block_relax_scores)}.json')
    if not save: return
    with open(file_name, 'w') as f:
        json.dump(new_bias, f)

