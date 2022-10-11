from hashlib import new
import torch
import json
import os
from sklearn.cluster import KMeans, MeanShift, DBSCAN, SpectralClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
import numpy as np
from collections import Counter
from scipy.spatial import distance
# from datasets.multienv.eiil import eiil
from eiil import eiil
from collections import Counter
import argparse
from random import randint
from blocking import blocking_envs

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--env_type", type=str, required=True, help="eiil, or kmeans")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--n_clusters', nargs='+', type=int, default=2)
    parser.add_argument('--auto_select', dest='auto_select', action='store_true')
    parser.add_argument('--dataset_name', type=str, required=True, help="Examples:cmnist, bmnist")
    parser.set_defaults(auto_select=False)
    return parser

class eiil_config(object):
    def __init__(self, lr, steps, save_dir, probs_file, is_val) -> None:
        self.lr = lr
        self.n_steps = steps
        self.save_file = os.path.join(save_dir, "train_eiil_0_{}_{}.json".format(lr, steps))
        self.is_val = is_val
        self.pretrained_model_prediction_file = probs_file

def prepare_for_infer_envs(preds_file, probs_file_dir):
    erm_preds = torch.load(preds_file)
    prob_1 = (1. / (1. + np.exp(-erm_preds['logits']))).reshape(-1, 1)
    probs = np.concatenate((1. - prob_1, prob_1), 1)
    with open(probs_file_dir, "w") as f:
        new_data = {
            "y": erm_preds['y'].tolist(),
            "logits_1": erm_preds['logits'].tolist(),
            "log_probs": probs.tolist() # infact, probs
        }
        json.dump(new_data, f)
    print("Prepared as json.")
    return probs_file_dir

def cluster(train_bias, cluster_func, use_log=True, print_centers=False, show_separation=True):
    log_probs = train_bias['log_probs']
    vector = np.array(log_probs) if use_log else np.exp(log_probs)
    cluster_func = cluster_func.fit(vector)
    cluster_ids = cluster_func.labels_
    if hasattr(cluster_func, 'cluster_centers_'):
        if print_centers:
            if use_log:
                print(sorted(np.exp(cluster_func.cluster_centers_), key=lambda x:x[0]))
            else:
                print(sorted(cluster_func.cluster_centers_, key=lambda x:x[0]))
        if show_separation:
            centers = cluster_func.cluster_centers_
            separation = 0
            for c in centers:
                separation += np.square(c - np.array(centers)).sum()
            separation = separation/(len(centers) ** 2)
            print("separation:", separation)
    counter = Counter(cluster_ids)
    print(counter)
    # print("silhouette_score: ", silhouette_score(vector, cluster_ids)) # lower is better
    db_score = davies_bouldin_score(vector, cluster_ids)
    print("calinski_harabasz_score:", calinski_harabasz_score(vector, cluster_ids)) # higher is better
    print("davies_bouldin_score:", db_score) # lower is better
    return cluster_ids, cluster_func, db_score

def get_kmeans_env_file(probs_file, val_probs_file, n_clusters, use_log=True, save_dir=None, auto_select=False):
    with open(probs_file, "r") as f:
        data = json.load(f)
    save_num = n_clusters
    if auto_select:
        db_score_list = []
        for num_clusters in n_clusters:
            print('------')
            print("number of clusters:", num_clusters)
            cluster_ids, cluster_func, db_score = cluster(data, KMeans(n_clusters=num_clusters, init='k-means++', max_iter=10000), use_log=use_log)
            # print("Centers: ", cluster_func.cluster_centers_)
            db_score_list.append(db_score)
        max_num_idx = np.argmin(np.array(db_score_list))
        max_num = n_clusters[max_num_idx]
        print("Optimal number of clusters: ", max_num)
        save_num = max_num
    if save_dir is not None:
        cluster_ids, cluster_func, _ = cluster(data, KMeans(n_clusters=save_num, init='k-means++', max_iter=10000), use_log=use_log)
        group_ids = [int(i) for i in cluster_ids]
        centers = cluster_func.cluster_centers_
        val_ids = []
        data_val = json.load(open(val_probs_file))
        for i, log_prob in enumerate(data_val['log_probs']):
            group_id = int(np.argmin(np.square(log_prob - centers).sum(axis=1)))
            val_ids.append(group_id)
        gids = {'group_ix': group_ids, 'y': data['y'], 
                'val_group_ix': val_ids, 'val_y': data_val['y']}
        with open(os.path.join(save_dir, "kmeans_{}.json".format(save_num)), 'w') as f:
            json.dump(gids, f)
        print("Kmeans results are saved.")
        return save_num
    else:
        for num_clusters in n_clusters:
            print('------')
            print("number of clusters:", num_clusters)
            cluster_ids, cluster_func, _ = cluster(data, KMeans(n_clusters=num_clusters, init='k-means++', max_iter=10000), use_log=use_log)
            # print("Centers: ", cluster_func.cluster_centers_)

#
# Insert weights
#
# def idx2weight_generation(y, group_ix, data):
def insert_weights(env_file_name, save_dir, save=True):
    with open(os.path.join(save_dir, f'{env_file_name}.json')) as f:
        env_data = json.load(f)
    y = env_data['y']
    group_ix = env_data['group_ix']
    gid_with_y = [(group_ix[i], y[i]) for i in range(len(group_ix))]
    gid_y_counter = Counter(gid_with_y)
    print('gid_y_counter:', gid_y_counter)
    # group_counter = Counter(group_ix)
    group = set(group_ix)
    # counts_group = [group_counter[i] for i in range(len(group))]
    counts_y_group = np.zeros((len(group), len(set(y))))
    for key in gid_y_counter.keys():
        counts_y_group[key[0], int(key[1])] = gid_y_counter[key]
    propensity_group = counts_y_group / np.sum(counts_y_group, axis=1).reshape(len(group),1)
    print('propensity_group:', propensity_group)
    # for train
    idx2weight = np.ones(len(y))
    for i in range(len(y)):
        inv_weight = propensity_group[group_ix[i], int(y[i])]
        if inv_weight == 0:
            print("anomaly sample: ", i)
            # if i in train_ixs:
            #     print("Strange. Warning.")
            continue
        elif inv_weight < 0.01 or inv_weight > 0.99 : continue
        idx2weight[i] = 1. / inv_weight
    d = {
            "group_ix": group_ix,
            "weight": idx2weight.tolist()
        }
    # for val
    if 'val_y' in env_data.keys():
        val_group_ix = env_data['val_group_ix']
        val_weights = np.ones(len(val_group_ix))
        for i, gix in enumerate(val_group_ix):
            inv_weight = propensity_group[gix, int(env_data['val_y'][i])]
            if inv_weight == 0:
                print("anomaly sample: ", i)
                continue
            elif inv_weight < 0.01 or inv_weight > 0.99 : continue
            val_weights[i] = 1. / inv_weight
        d["val_group_ix"] = val_group_ix
        d["val_weight"] = val_weights.tolist()
    if save:
        file_name = os.path.join(save_dir, env_file_name)
        with open(f"{file_name}_weight.json", 'w') as f:
            json.dump(d, f)
        print("Saved file as ", f"{file_name}_weight.json")
    return idx2weight

def insert_blocking_results(trainval_file_name, save_dir, group_weights, method):
    group_counts = group_weights['counts']
    file_name = os.path.join(save_dir, f'{trainval_file_name}-blocking-{method}-{group_counts}.json')
    trainval_data = json.load(open(os.path.join(save_dir, f'{trainval_file_name}.json')))
    for i in range(len(trainval_data)):
        trainval_data[i]['group_id'] = randint(0, 1)
        trainval_data[i]['weight'] = 1.
    for i, data_ix in enumerate(group_weights['train_ixs']):
        trainval_data[data_ix]['group_id'] = group_weights['group_ix'][i]
        trainval_data[data_ix]['weight'] = group_weights['weight'][i]
    with open(file_name, 'w') as f:
        json.dump(trainval_data, f)
    print("Saved file as ", file_name)

# -----------------
# The main function
#
def get_env_weights(config):
    save_dir = config.save_dir
    if 'cmnist' in config.dataset_name:
        train_preds_file = os.path.join(save_dir, "preds_Train_Main_epoch_100.pt")
        val_preds_file = os.path.join(save_dir, "preds_Val_Main_epoch_100.pt")
        train_probs_file = os.path.join(save_dir, "erm_preds_100.json")
        val_probs_file = os.path.join(save_dir, "erm_preds_val_100.json")
        # kmeans_env_file = os.path.join(save_dir, f"kmeans_{n_clusters}.json")
        # eiil_env_file = os.path.join(save_dir, f"train_eiil_0_{lr}_{steps}.json")
        prepare_for_infer_envs(train_preds_file, train_probs_file)
        prepare_for_infer_envs(val_preds_file, val_probs_file)
    else:
        train_probs_file = "/sharefs/thuhc-data/c2d949ee/backup/psp03/folder/data/bias-mitigators/data/biased_mnist/full_v7_0.7/onehot_lr_train.json"
        val_probs_file = "/sharefs/thuhc-data/c2d949ee/backup/psp03/folder/data/bias-mitigators/data/biased_mnist/full_v7_0.7/onehot_lr_val.json"

    ei_config = eiil_config(config.lr, config.n_steps, save_dir, train_probs_file, is_val=False)
    if config.env_type == 'eiil':
        ei_val_config = eiil_config(config.lr, config.n_steps, save_dir, val_probs_file, is_val=True)
        print('Start eiil.')
        eiil(ei_config)
        eiil(ei_val_config)
        
    elif config.env_type == 'kmeans':
        if config.dataset_name == 'cmnist':
            use_log = True
        else:
            use_log = False
        if not config.auto_select:
            get_kmeans_env_file(train_probs_file, val_probs_file, config.n_clusters, auto_select=config.auto_select, use_log=use_log)
        else:
            save_num = get_kmeans_env_file(train_probs_file, val_probs_file, config.n_clusters, save_dir=save_dir, auto_select=config.auto_select, use_log=use_log)
            insert_weights(f"kmeans_{save_num}", save_dir, save=config.auto_select)
    elif config.env_type == 'blocking':
        train_data = json.load(open(train_probs_file))
        val_data = json.load(open(val_probs_file))
        method = 'anova'
        if 'train_ixs' in train_data.keys():
            group_weights, file_name = blocking_envs(train_data['log_probs'], train_data['y'], 15, train_data['train_ixs'], save_dir, method=method)
        else:
            group_weights, file_name = blocking_envs(train_data['log_probs'], train_data['y'], 15, val_bias_probs=val_data['log_probs'], val_y=val_data['y'], save_folder=save_dir, method=method)
        if config.dataset_name == 'bmnist':
            insert_blocking_results("trainval", save_dir, group_weights, method)
        else:
            print('Blocking env_file is: ', file_name)


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_args()
    get_env_weights(config)