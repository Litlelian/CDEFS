import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import r_regression, SelectKBest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV

from utils import *
from feature import *
from model import *

def get_5fold_test_probs_fast(all_embs, all_emb_ways, labels, save_path, if_save_pred = False, task_name = '', seed_bias = 0, model_est = -1):
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok = True)
    each_embs_probs_5fold_list = []
    all_test_performance = {'Acc':[], 'Mcc':[], 'Sens':[], 'Spec':[], 'AUC':[]}
    if seed_bias != 0:
        save_path_name = save_path.split('/')[-1] + f'_10time_5fold_seed_bias{seed_bias}.txt'
    else:
        save_path_name = save_path.split('/')[-1] + f'_10time_5fold.txt'
    if save_path_name not in os.listdir('/'.join(save_path.split('/')[:-1])):
        for time in range(10):
            kf = KFold(n_splits=5, shuffle=True, random_state=time+seed_bias)
            for k, (tr_idx, tt_idx) in enumerate(kf.split(labels)):
                tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
                print(f'Time {time} Fold {k}')
                each_embs_probs = []
                for ei, embs in enumerate(all_embs):
                    tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
                    if model_est == -1: clf = get_model('RF', tr_embs.shape[-1])
                    else: clf = get_RF_diff_hyper(model_est)
                    clf.fit(tr_embs, tr_labels)
                    each_embs_probs.append(clf.predict_proba(tt_embs)[:, 1])
                each_embs_probs = np.stack(each_embs_probs)
                each_embs_probs_5fold_list.append(each_embs_probs)
        if seed_bias != 0:
            save_txt(f'{save_path}_10time_5fold_seed_bias{seed_bias}.txt', each_embs_probs_5fold_list)
        else:
            save_txt(f'{save_path}_10time_5fold.txt', each_embs_probs_5fold_list)
    else:
        print('Has inferenced !!!')
        if seed_bias != 0:
            each_embs_probs_5fold_list = load_txt(f'{save_path}_10time_5fold_seed_bias{seed_bias}.txt')
        else:
            each_embs_probs_5fold_list = load_txt(f'{save_path}_10time_5fold.txt')
    
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time+seed_bias)
        for foldk, (tr_idx, tt_idx) in enumerate(kf.split(labels)):
            tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
            each_embs_probs = each_embs_probs_5fold_list[foldk + 5 * time]
            pred_probs = np.array(each_embs_probs).mean(0)
            preds = [1 if i > 0.5 else 0 for i in pred_probs]
            acc, mcc, sen, spe, auc = performance(preds, pred_probs, tt_labels)
            test_performance = {'Acc':acc, 'Mcc':mcc, 'Sens':sen, 'Spec':spe, 'AUC':auc}
            for k, v in test_performance.items():
                all_test_performance[k].append(v)
    if not if_save_pred:
        return each_embs_probs_5fold_list
    else:
        record = {'Embed way': [' '.join(all_emb_ways)]}
        for k in test_performance.keys():
            print(f'{k} : {np.mean(all_test_performance[k]):.4f} ({np.std(all_test_performance[k]):.4f})')
            record[k + ' mean'] = [np.mean(all_test_performance[k])]
            record[k + ' std'] = [np.std(all_test_performance[k])]
        record = pd.DataFrame(record)
        record.to_csv(f'results/{task_name}_seed_bias{seed_bias}_ensemble_performance.csv', mode='a', header=not os.path.exists(f'results/{task_name}_seed_bias{seed_bias}_ensemble_performance.csv'), index=False)
        return each_embs_probs_5fold_list, record

def get_5fold_test_probs(all_embs, all_emb_ways, labels, save_path, if_save_pred = False, task_name = '', seed_bias = 0, model_est = -1):
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok = True)
    each_embs_probs_5fold_list = []
    all_test_performance = {'Acc':[], 'Mcc':[], 'Sens':[], 'Spec':[], 'AUC':[]}
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time+seed_bias)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(labels)):
            tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
            if seed_bias != 0:
                save_path_name = save_path.split('/')[-1] + f'_time{time}_fold{k}_seed_bias{seed_bias}.txt'
            else:
                save_path_name = save_path.split('/')[-1] + f'_time{time}_fold{k}.txt'
            if save_path_name not in os.listdir('/'.join(save_path.split('/')[:-1])):
                print(f'Time {time} Fold {k}')
                each_embs_probs = []
                for ei, embs in enumerate(all_embs):
                    tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
                    if model_est == -1: clf = get_model('RF', tr_embs.shape[-1])
                    else: clf = get_RF_diff_hyper(model_est)
                    clf.fit(tr_embs, tr_labels)
                    each_embs_probs.append(clf.predict_proba(tt_embs)[:, 1])
                each_embs_probs = np.stack(each_embs_probs)
                if seed_bias != 0:
                    save_txt(f'{save_path}_time{time}_fold{k}_seed_bias{seed_bias}.txt', each_embs_probs)
                else:
                    save_txt(f'{save_path}_time{time}_fold{k}.txt', each_embs_probs)
            else:
                print('Has inferenced !!!')
                if seed_bias != 0:
                    each_embs_probs = load_txt(f'{save_path}_time{time}_fold{k}_seed_bias{seed_bias}.txt')
                else:
                    each_embs_probs = load_txt(f'{save_path}_time{time}_fold{k}.txt')
            each_embs_probs_5fold_list.append(each_embs_probs)
            pred_probs = np.array(each_embs_probs).mean(0)
            preds = [1 if i > 0.5 else 0 for i in pred_probs]
            acc, mcc, sen, spe, auc = performance(preds, pred_probs, tt_labels)
            test_performance = {'Acc':acc, 'Mcc':mcc, 'Sens':sen, 'Spec':spe, 'AUC':auc}
            for k, v in test_performance.items():
                all_test_performance[k].append(v)
    if not if_save_pred:
        return each_embs_probs_5fold_list
    else:
        record = {'Embed way': [' '.join(all_emb_ways)]}
        for k in test_performance.keys():
            print(f'{k} : {np.mean(all_test_performance[k]):.4f} ({np.std(all_test_performance[k]):.4f})')
            record[k + ' mean'] = [np.mean(all_test_performance[k])]
            record[k + ' std'] = [np.std(all_test_performance[k])]
        record = pd.DataFrame(record)
        record.to_csv(f'results/{task_name}_seed_bias{seed_bias}_ensemble_performance.csv', mode='a', header=not os.path.exists(f'results/{task_name}_seed_bias{seed_bias}_ensemble_performance.csv'), index=False)
        return each_embs_probs_5fold_list, record

def get_ind_test_probs(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, save_path, if_save_pred = False, task_name = '', model_est = -1):
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok = True)
    if save_path.split('/')[-1] + '_indtest.txt' not in os.listdir('/'.join(save_path.split('/')[:-1])):
        print('Training....')
        each_embs_probs = []
        for ei in range(len(all_emb_ways)):
            tr_embs = train_embs[ei]
            tt_embs = test_embs[ei]
            if model_est == -1: clf = get_model('RF', tr_embs.shape[-1])
            else: clf = get_RF_diff_hyper(model_est)
            clf.fit(tr_embs, tr_labels)
            each_embs_probs.append(clf.predict_proba(tt_embs)[:, 1])
        each_embs_probs = np.stack(each_embs_probs)
        save_txt(f'{save_path}_indtest.txt', each_embs_probs)
    else:
        print('Has inferenced !!!')
        each_embs_probs = load_txt(f'{save_path}_indtest.txt')
    pred_probs = np.array(each_embs_probs).mean(0)
    preds = [1 if i > 0.5 else 0 for i in pred_probs]
    acc, mcc, sen, spe, auc = performance(preds, pred_probs, tt_labels)
    test_performance = {'Acc':acc, 'Mcc':mcc, 'Sens':sen, 'Spec':spe, 'AUC':auc}
    if not if_save_pred:
        return each_embs_probs
    else:
        record = {'Embed way': [' '.join(all_emb_ways)]}
        for k in test_performance.keys():
            print(f'{k} : {test_performance[k]:.4f}')
            record[k] = [test_performance[k]]
        record = pd.DataFrame(record)
        record.to_csv(f'results/{task_name}_ensemble_performance.csv', mode='a', header=not os.path.exists(f'results/{task_name}_ensemble_performance.csv'), index=False)
        return each_embs_probs, record

def analyze_corr(each_embs_probs_5fold_list, all_emb_ways, task_name):
    cos_matrix, entropy_list, std_list = [], [], []
    corr_dict = {'pearson':[], 'spearman':[], 'kendall':[], 'cosine':[], 'euclidean':[], 'manhattan':[]}
    for ep in each_embs_probs_5fold_list:
        # entropy in sample
        entropy_list.append(cal_entropy(ep))
        # std in sample
        std_list.append(cal_each_model_std(ep))
        # prediction correlation between each model
        for cor_key, cor_values in cal_correlation(ep).items():
            corr_dict[cor_key].append(cor_values)
        # cosine similarity
        cos_matrix.append(pd.DataFrame(cosine_similarity(ep), index=all_emb_ways, columns=all_emb_ways))
    cos_matrix = sum(cos_matrix) / len(cos_matrix)
    cos_matrix.to_csv(f'results/cos_sim/{task_name}_cos_sim_len{len(all_emb_ways)}.csv')
    entropy_df = pd.DataFrame([{
        'embs': '/'.join(all_emb_ways),
        'entropy mean': np.mean(entropy_list),
        'entropy std': np.std(entropy_list),
        }])
    entropy_df.to_csv(f'results/entropy_of_{task_name}.csv', mode='a', index=False, header=not os.path.exists(f'results/entropy_of_{task_name}.csv'))
    std_df = pd.DataFrame([{
        'embs': '/'.join(all_emb_ways),
        'std mean': np.mean(std_list),
        'std std': np.std(std_list),
        }])
    std_df.to_csv(f'results/std_of_{task_name}.csv', mode='a', index=False, header=not os.path.exists(f'results/std_of_{task_name}.csv'))
    stats = {f"{key} mean": pd.Series(values).mean() for key, values in corr_dict.items()}
    stats.update({f"{key} std": pd.Series(values).std(ddof=0) for key, values in corr_dict.items()})
    corr_df = pd.DataFrame([stats])
    corr_df.to_csv(f'results/each_pred_corr_of_{task_name}.csv', mode='a', index=False, header=not os.path.exists(f'results/each_pred_corr_of_{task_name}.csv'))
    return cos_matrix

def analyze_corr_ind(each_embs_probs, all_emb_ways, task_name):
    cos_matrix, entropy_list, std_list = [], [], []
    corr_dict = {'pearson':0, 'spearman':0, 'kendall':0, 'cosine':0, 'euclidean':0, 'manhattan':0}
    # entropy in sample
    entropy = cal_entropy(each_embs_probs)
    # std in sample
    std = cal_each_model_std(each_embs_probs)
    # prediction correlation between each model
    for cor_key, cor_values in cal_correlation(each_embs_probs).items():
        corr_dict[cor_key] = cor_values
    # cosine similarity
    cos_matrix = pd.DataFrame(cosine_similarity(each_embs_probs), index=all_emb_ways, columns=all_emb_ways)
    cos_matrix.to_csv(f'results/cos_sim/{task_name}_cos_sim_len{len(all_emb_ways)}.csv')
    entropy_df = pd.DataFrame([{
        'embs': '/'.join(all_emb_ways),
        'entropy': entropy,
        }])
    entropy_df.to_csv(f'results/entropy_of_{task_name}.csv', mode='a', index=False, header=not os.path.exists(f'results/entropy_of_{task_name}.csv'))
    std_df = pd.DataFrame([{
        'embs': '/'.join(all_emb_ways),
        'std': std,
        }])
    std_df.to_csv(f'results/std_of_{task_name}.csv', mode='a', index=False, header=not os.path.exists(f'results/std_of_{task_name}.csv'))
    corr_df = pd.DataFrame([corr_dict])
    corr_df.to_csv(f'results/each_pred_corr_of_{task_name}.csv', mode='a', index=False, header=not os.path.exists(f'results/each_pred_corr_of_{task_name}.csv'))
    return cos_matrix

def combine_csv(task_name):
    pred_df = pd.read_csv(f'results/{task_name}_ensemble_performance.csv')
    entropy_df = pd.read_csv(f'results/entropy_of_{task_name}.csv').iloc[:, 1:]
    std_df = pd.read_csv(f'results/std_of_{task_name}.csv').iloc[:, 1:]
    corr_df = pd.read_csv(f'results/each_pred_corr_of_{task_name}.csv')
    merge_df = pd.concat([pred_df, entropy_df, std_df, corr_df], axis = 1).round(4)
    merge_df.to_csv(f'results/{task_name}.csv')

def group_sim(data_file_name):
    embs, emb_ways, labels = get_embs_set(data_file_name, device)
    # only w2v
    # all_embs = embs[:10]
    # all_emb_ways = emb_ways[:10]
    # task_name = 'group_w2vs'
    # all
    all_embs = embs
    all_emb_ways = emb_ways
    task_name = 'group_all'
    while len(all_emb_ways) > 1:
        print(all_emb_ways)
        each_embs_probs_5fold_list = get_5fold_test_probs(all_embs, all_emb_ways, labels, f'results/probs_txt/forward_test_probs/{task_name}_len{len(all_emb_ways)}')
        cos_matrix = analyze_corr(each_embs_probs_5fold_list, all_emb_ways, task_name)
        _, _ = get_5fold_test_probs(all_embs, all_emb_ways, labels, f'results/probs_txt/forward_test_probs/{task_name}_len{len(all_emb_ways)}', True, task_name)
        # concat top 2 similar
        mean_sim = cos_matrix.mean()
        top2_sim_idx = np.argsort(mean_sim)[-2:].tolist()
        if top2_sim_idx[1] > top2_sim_idx[0]:
            top2_sim_idx = top2_sim_idx[::-1]
        print(top2_sim_idx)
        temp_embs_list, temp_way_list = [], []
        for i in top2_sim_idx:
            temp_embs_list.append(all_embs.pop(i))
            temp_way_list.append(all_emb_ways.pop(i))
        all_embs.append(np.concatenate(temp_embs_list, axis=1))
        all_emb_ways.append('+'.join(temp_way_list))
    combine_csv(task_name)

def group_sim_ind(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # only w2v
    train_embs = tr_embs[:10]
    test_embs = tt_embs[:10]
    all_emb_ways = emb_ways[:10]
    task_name = f'{dataset_name}_group_w2vs'
    # all
    # train_embs = tr_embs
    # test_embs = tt_embs
    # all_emb_ways = emb_ways
    # task_name = f'{dataset_name}_group_all'
    while len(all_emb_ways) > 1:
        print(all_emb_ways)
        each_embs_probs = get_ind_test_probs(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{task_name}_len{len(all_emb_ways)}')
        cos_matrix = analyze_corr_ind(each_embs_probs, all_emb_ways, task_name)
        _, _ = get_ind_test_probs(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{task_name}_len{len(all_emb_ways)}', True, task_name)
        # concat top 2 similar
        mean_sim = cos_matrix.mean()
        top2_sim_idx = np.argsort(mean_sim)[-2:].tolist()
        if top2_sim_idx[1] > top2_sim_idx[0]:
            top2_sim_idx = top2_sim_idx[::-1]
        print(top2_sim_idx)
        temp1_embs_list, temp2_embs_list, temp_way_list = [], [], []
        for i in top2_sim_idx:
            temp1_embs_list.append(train_embs.pop(i))
            temp2_embs_list.append(test_embs.pop(i))
            temp_way_list.append(all_emb_ways.pop(i))
        train_embs.append(np.concatenate(temp1_embs_list, axis=1))
        test_embs.append(np.concatenate(temp2_embs_list, axis=1))
        all_emb_ways.append('+'.join(temp_way_list))
    combine_csv(task_name)

def group_sim_ind_drop(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # only w2v
    train_embs = tr_embs[:10]
    test_embs = tt_embs[:10]
    all_emb_ways = emb_ways[:10]
    task_name = f'{dataset_name}_group_drop_w2vs'
    # all
    # train_embs = tr_embs
    # test_embs = tt_embs
    # all_emb_ways = emb_ways
    # task_name = f'{dataset_name}_group_drop_all'
    while len(all_emb_ways) > 0:
        print(all_emb_ways)
        each_embs_probs = get_ind_test_probs(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{task_name}_len{len(all_emb_ways)}')
        cos_matrix = analyze_corr_ind(each_embs_probs, all_emb_ways, task_name)
        _, _ = get_ind_test_probs(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{task_name}_len{len(all_emb_ways)}', True, task_name)
        # drop top 1 similar
        mean_sim = cos_matrix.mean()
        top1_sim_idx = np.argsort(mean_sim)[-1].tolist()
        train_embs.pop(top1_sim_idx)
        test_embs.pop(top1_sim_idx)
        all_emb_ways.pop(top1_sim_idx)
    combine_csv(task_name)

def group_sim_ind_watch_total(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    group_sim_ind is compare similar with each mer w2vs, but it's different to the similarity what we try to stop ensemble. Watch total means C(10, 2) and find out the most unsimilar combination, and keep going on...
    '''
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # for sim_way in ['pearson', 'spearman', 'kendall' ,'cosine', 'euclidean', 'manhattan']:
    for sim_way in ['spearman']:
        print(sim_way)
        # only w2v
        train_embs = tr_embs[:10]
        test_embs = tt_embs[:10]
        all_emb_ways = emb_ways[:10]
        task_name = f'{dataset_name}_group_watch_all_w2vs_abs'
        # all
        # train_embs = tr_embs
        # test_embs = tt_embs
        # all_emb_ways = emb_ways
        # task_name = f'{dataset_name}_group_watch_all_10times_all'
        while len(all_emb_ways) > 1:
            print(all_emb_ways)
            pairs = list(combinations(range(len(all_emb_ways)), 2))
            corr_list = []
            for emb1_index, emb2_index in pairs:
                # concat top 2 similar
                curr_train_embs = train_embs[:emb1_index] + train_embs[emb1_index+1:emb2_index] + train_embs[emb2_index+1:] + [np.concatenate([train_embs[emb1_index], train_embs[emb2_index]], axis=1)]
                curr_test_embs = test_embs[:emb1_index] + test_embs[emb1_index+1:emb2_index] + test_embs[emb2_index+1:] + [np.concatenate([test_embs[emb1_index], test_embs[emb2_index]], axis=1)]
                curr_emb_ways = all_emb_ways[:emb1_index] + all_emb_ways[emb1_index+1:emb2_index] + all_emb_ways[emb2_index+1:] + ['+'.join([all_emb_ways[emb1_index], all_emb_ways[emb2_index]])]
                sorted_emb_ways = sort_emb_ways(curr_emb_ways)
                each_embs_probs = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{task_name}_emb_{"_".join(sorted_emb_ways)}')
                correlations = cal_correlation(each_embs_probs)
                corr_list.append(correlations[sim_way])
            if sim_way in ['pearson', 'spearman', 'kendall' ,'cosine']:
                choose_pair_index = np.argmin(corr_list)
            elif sim_way in ['euclidean', 'manhattan']:
                choose_pair_index = np.argmax(corr_list)
            choose_pair1 = pairs[choose_pair_index][0]
            choose_pair2 = pairs[choose_pair_index][1]
            #concat top 2 similar
            train_embs = train_embs[:choose_pair1] + train_embs[choose_pair1+1:choose_pair2] + train_embs[choose_pair2+1:] + [np.concatenate([train_embs[choose_pair1], train_embs[choose_pair2]], axis=1)]
            test_embs = test_embs[:choose_pair1] + test_embs[choose_pair1+1:choose_pair2] + test_embs[choose_pair2+1:] + [np.concatenate([test_embs[choose_pair1], test_embs[choose_pair2]], axis=1)]
            all_emb_ways = all_emb_ways[:choose_pair1] + all_emb_ways[choose_pair1+1:choose_pair2] + all_emb_ways[choose_pair2+1:] + [all_emb_ways[choose_pair1] + '+' + all_emb_ways[choose_pair2]]
            sorted_all_emb_ways = sort_emb_ways(all_emb_ways)
            each_embs_probs, _ = get_ind_test_probs(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{task_name}_emb_{"_".join(sorted_all_emb_ways)}', True, task_name + '_' + sim_way)
            _ = analyze_corr_ind(each_embs_probs, all_emb_ways, task_name + '_' + sim_way)
        combine_csv(task_name + '_' + sim_way)

### drop is too simple, try to concat and drop at the same time
# def group_sim_ind_watch_total_drop(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    # '''
    # group_sim_ind is compare similar with each mer w2vs, but it's different to the similarity what we try to stop ensemble. Watch total means C(10, 2) and find out the most unsimilar combination, drop oone of these and keep going on...
    # '''
    # tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # for sim_way in ['pearson', 'spearman', 'kendall' ,'cosine', 'euclidean', 'manhattan']:
        # print(sim_way)
        # # only w2v
        # train_embs = tr_embs[:10]
        # test_embs = tt_embs[:10]
        # all_emb_ways = emb_ways[:10]
        # task_name = f'{dataset_name}_group_watch_all_10times_drop_w2vs'
        # # all
        # # train_embs = tr_embs
        # # test_embs = tt_embs
        # # all_emb_ways = emb_ways
        # # task_name = f'{dataset_name}_group_watch_all_10times_drop_all'
        # while len(all_emb_ways) > 1:
            # print(all_emb_ways)
            # corr_list = []
            # for emb_index in range(len(all_emb_ways)):
                # # drop top 1 similar
                # curr_train_embs = train_embs[:emb_index] + train_embs[emb_index+1:]
                # curr_test_embs = test_embs[:emb_index] + test_embs[emb_index+1:]
                # curr_emb_ways = all_emb_ways[:emb_index] + all_emb_ways[emb_index+1:]
                # curr_emb_ways = sort_emb_ways(curr_emb_ways)
                # each_embs_probs = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{task_name}_emb_{"_".join(curr_emb_ways)}')
                # correlations = cal_correlation(each_embs_probs)
                # corr_list.append(correlations[sim_way])
            # if sim_way in ['pearson', 'spearman', 'kendall' ,'cosine']:
                # choose_index = np.argmin(corr_list)
            # elif sim_way in ['euclidean', 'manhattan']:
                # choose_index = np.argmax(corr_list)
            # #concat top 2 similar
            # train_embs = train_embs[:choose_index] + train_embs[choose_index+1:]
            # test_embs = test_embs[:choose_index] + test_embs[choose_index+1:]
            # all_emb_ways = all_emb_ways[:choose_index] + all_emb_ways[choose_index+1:]
            # all_emb_ways = sort_emb_ways(all_emb_ways)
            # each_embs_probs, _ = get_ind_test_probs(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{task_name}_emb_{"_".join(all_emb_ways)}', True, task_name + '_' + sim_way)
            # _ = analyze_corr_ind(each_embs_probs, all_emb_ways, task_name + '_' + sim_way)
        # combine_csv(task_name + '_' + sim_way)

def group_sim_ind_watch_total_concat_or_drop(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    group_sim_ind is compare similar with each mer w2vs, but it's different to the similarity what we try to stop ensemble. Watch total means C(10, 2) and find out the most unsimilar combination, try to concat or drop to get lower similarity (spearman)
    '''
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    sim_way = 'spearman'
    print(sim_way)
    # only w2v
    train_embs = tr_embs[:10]
    test_embs = tt_embs[:10]
    all_emb_ways = emb_ways[:10]
    task_name = f'{dataset_name}_group_watch_all_concat_or_drop_w2vs_abs'
    # all
    # train_embs = tr_embs
    # test_embs = tt_embs
    # all_emb_ways = emb_ways
    # task_name = f'{dataset_name}_group_watch_all_concat_or_drop_all'
    while len(all_emb_ways) > 1:
        print(all_emb_ways)
        pairs = list(combinations(range(len(all_emb_ways)), 2))
        corr_list = []
        # try to concat
        for emb1_index, emb2_index in pairs:
            # concat top 2 similar
            curr_train_embs = train_embs[:emb1_index] + train_embs[emb1_index+1:emb2_index] + train_embs[emb2_index+1:] + [np.concatenate([train_embs[emb1_index], train_embs[emb2_index]], axis=1)]
            curr_test_embs = test_embs[:emb1_index] + test_embs[emb1_index+1:emb2_index] + test_embs[emb2_index+1:] + [np.concatenate([test_embs[emb1_index], test_embs[emb2_index]], axis=1)]
            curr_emb_ways = all_emb_ways[:emb1_index] + all_emb_ways[emb1_index+1:emb2_index] + all_emb_ways[emb2_index+1:] + ['+'.join([all_emb_ways[emb1_index], all_emb_ways[emb2_index]])]
            sorted_emb_ways = sort_emb_ways(curr_emb_ways)
            # each_embs_probs = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{task_name}_emb_{"_".join(sorted_emb_ways)}')
            each_embs_probs = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{dataset_name}_group_watch_all_concat_or_drop_w2vs_emb_{"_".join(sorted_emb_ways)}')
            correlations = cal_correlation(each_embs_probs)
            corr_list.append(correlations[sim_way])
        # try to drop
        for emb_index in range(len(all_emb_ways)):
            # drop top 1 similar
            curr_train_embs = train_embs[:emb_index] + train_embs[emb_index+1:]
            curr_test_embs = test_embs[:emb_index] + test_embs[emb_index+1:]
            curr_emb_ways = all_emb_ways[:emb_index] + all_emb_ways[emb_index+1:]
            sorted_emb_ways = sort_emb_ways(curr_emb_ways)
            each_embs_probs = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{dataset_name}_group_watch_all_concat_or_drop_w2vs_emb_{"_".join(sorted_emb_ways)}')
            correlations = cal_correlation(each_embs_probs)
            corr_list.append(correlations[sim_way])
        choose_pair_index = np.argmin(corr_list)
        if choose_pair_index >= len(pairs):
            # drop
            choose_index = choose_pair_index - len(pairs)
            train_embs = train_embs[:choose_index] + train_embs[choose_index+1:]
            test_embs = test_embs[:choose_index] + test_embs[choose_index+1:]
            all_emb_ways = all_emb_ways[:choose_index] + all_emb_ways[choose_index+1:]
            sorted_all_emb_ways = all_emb_ways
        else:
            # concat
            choose_pair1 = pairs[choose_pair_index][0]
            choose_pair2 = pairs[choose_pair_index][1]
            #concat top 2 similar
            train_embs = train_embs[:choose_pair1] + train_embs[choose_pair1+1:choose_pair2] + train_embs[choose_pair2+1:] + [np.concatenate([train_embs[choose_pair1], train_embs[choose_pair2]], axis=1)]
            test_embs = test_embs[:choose_pair1] + test_embs[choose_pair1+1:choose_pair2] + test_embs[choose_pair2+1:] + [np.concatenate([test_embs[choose_pair1], test_embs[choose_pair2]], axis=1)]
            all_emb_ways = all_emb_ways[:choose_pair1] + all_emb_ways[choose_pair1+1:choose_pair2] + all_emb_ways[choose_pair2+1:] + [all_emb_ways[choose_pair1] + '+' + all_emb_ways[choose_pair2]]
            sorted_all_emb_ways = sort_emb_ways(all_emb_ways)
        each_embs_probs, _ = get_ind_test_probs(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{dataset_name}_group_watch_all_concat_or_drop_w2vs_emb_{"_".join(sorted_all_emb_ways)}', True, task_name + '_' + sim_way)
        _ = analyze_corr_ind(each_embs_probs, all_emb_ways, task_name + '_' + sim_way)
    combine_csv(task_name + '_' + sim_way)

def FS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name, if_concat = False):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    hc_tr_embs, hc_tt_embs, hc_emb_ways, _, _ = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    # only w2v
    # train_embs = tr_embs[:10]
    # test_embs = tt_embs[:10]
    # all_emb_ways = emb_ways[:10]
    # task_name = f'{dataset_name}_FS_w2vs'
    # all
    train_embs = tr_embs[:10] + hc_tr_embs
    test_embs = tt_embs[:10] + hc_tt_embs
    all_emb_ways = emb_ways[:10] + hc_emb_ways
    # task_name = f'{dataset_name}_FS_w2v+hc'
    # train_embs, test_embs = select_in_feature_set(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, dataset_name)
    # task_name = f'{dataset_name}_FS_w2v+hc_select_in_feature_set_diff_seed'
    # selected_hc_train_embs, selected_hc_test_embs, _ = select_in_feature_set(train_embs[10:15], test_embs[10:15], all_emb_ways[10:15], tr_labels, tt_labels, dataset_name)
    # for i in range(10, 15):
    #     train_embs[i] = selected_hc_train_embs[i-10]
    #     test_embs[i] = selected_hc_test_embs[i-10]
    #     all_emb_ways[i] = all_emb_ways[i] + '*'
    task_name = f'{dataset_name}_FS_w2v+hc-same-feature'
    selected_train_embs, selected_test_embs, selected_emb_ways = [], [], []
    all_max_acc = 0
    while len(all_emb_ways) > 0:
        print(selected_emb_ways)
        temp_record = []
        for add_i in range(len(all_emb_ways)):
            curr_train_embs = selected_train_embs + [train_embs[add_i]]
            concat_train_embs = np.concatenate(curr_train_embs, axis=1)
            curr_emb_ways = selected_emb_ways + [all_emb_ways[add_i]]
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/FS_atp/{task_name}_len{len(all_emb_ways)}_index{add_i}', True, task_name + '_ensemble')
            # _, t = get_5fold_test_probs([concat_train_embs], '+'.join(curr_emb_ways), tr_labels, f'results/probs_txt/FS_atp/{task_name}_concat_len{len(all_emb_ways)}_index{add_i}', True, task_name + '_concat')
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"_".join(curr_emb_ways)}', True, task_name + '_ensemble')
            # _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"+".join(curr_emb_ways)}', True, task_name + '_concat')
            if if_concat:
                # _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"+".join(curr_emb_ways)}', True, task_name + '_concat')
                _, t = get_5fold_test_probs_fast([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"+".join(curr_emb_ways)}', True, task_name + '_concat')
                # _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"+".join(curr_emb_ways)}', True, task_name + '_concat', 10)
            else:
                # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(curr_emb_ways)}', True, task_name + '_ensemble')
                _, t = get_5fold_test_probs_fast(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"_".join(curr_emb_ways)}', True, task_name + '_ensemble')
                # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(curr_emb_ways)}', True, task_name + '_ensemble', 10)
            temp_record.append(t)
        record = pd.concat(temp_record)
        if dataset_name == 'ATP':
            metric = 'Mcc'
        else:
            metric = 'Acc'
        max_acc_id = record[f'{metric} mean'].to_numpy().argmax()
        max_acc = record[f'{metric} mean'].to_numpy().max()
        if max_acc < all_max_acc:
            print(f'End at {selected_emb_ways} !!!')
            if if_concat:
                return '+'.join(selected_emb_ways)
            else:
                return '/'.join(selected_emb_ways)
        else:
            all_max_acc = max_acc
            selected_train_embs.append(train_embs.pop(max_acc_id))
            selected_test_embs.append(test_embs.pop(max_acc_id))
            selected_emb_ways.append(all_emb_ways.pop(max_acc_id))

def BS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name, if_concat = False):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    hc_tr_embs, hc_tt_embs, hc_emb_ways, _, _ = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    # only w2v
    # train_embs = tr_embs[:10]
    # test_embs = tt_embs[:10]
    # all_emb_ways = emb_ways[:10]
    # task_name = f'{dataset_name}_BS_w2vs'
    # all
    train_embs = tr_embs[:10] + hc_tr_embs
    test_embs = tt_embs[:10] + hc_tt_embs
    all_emb_ways = emb_ways[:10] + hc_emb_ways
    # task_name = f'{dataset_name}_BS_w2v+hc'
    # train_embs, test_embs = select_in_feature_set(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, dataset_name)
    # task_name = f'{dataset_name}_BS_w2v+hc_select_in_feature_set_diff_seed'
    # selected_hc_train_embs, selected_hc_test_embs, _ = select_in_feature_set(train_embs[10:15], test_embs[10:15], all_emb_ways[10:15], tr_labels, tt_labels, dataset_name)
    # for i in range(10, 15):
    #     train_embs[i] = selected_hc_train_embs[i-10]
    #     test_embs[i] = selected_hc_test_embs[i-10]
    #     all_emb_ways[i] = all_emb_ways[i] + '*'
    task_name = f'{dataset_name}_BS_w2v+hc-same-feature'
    selected_train_embs = train_embs
    selected_emb_ways = all_emb_ways
    all_max_acc = 0
    best_emb_ways = selected_emb_ways
    while len(selected_emb_ways) > 1:
        print(selected_emb_ways)
        temp_record = []
        for remove_i in range(len(selected_emb_ways)):
            curr_train_embs = selected_train_embs[:remove_i] + selected_train_embs[remove_i+1:]
            curr_emb_ways = selected_emb_ways[:remove_i] + selected_emb_ways[remove_i+1:]
            concat_train_embs = np.concatenate(curr_train_embs, axis=1)
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/BS_atp/{task_name}_len{len(all_emb_ways)}_index{remove_i}', True, task_name + '_ensemble')
            # _, t = get_5fold_test_probs([concat_train_embs], '+'.join(curr_emb_ways), tr_labels, f'results/probs_txt/BS_atp/{task_name}_concat_len{len(all_emb_ways)}_index{remove_i}', True, task_name + '_concat')
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"_".join(curr_emb_ways)}', True, task_name + '_ensemble')
            # _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"+".join(curr_emb_ways)}', True, task_name + '_concat')
            if if_concat:
                # _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"+".join(curr_emb_ways)}', True, task_name + '_concat')
                _, t = get_5fold_test_probs_fast([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"+".join(curr_emb_ways)}', True, task_name + '_concat')
                # _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"+".join(curr_emb_ways)}', True, task_name + '_concat', 10)
            else:
                # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(curr_emb_ways)}', True, task_name + '_ensemble')
                _, t = get_5fold_test_probs_fast(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"_".join(curr_emb_ways)}', True, task_name + '_ensemble')
                # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(curr_emb_ways)}', True, task_name + '_ensemble', 10)
            temp_record.append(t)
        record = pd.concat(temp_record)
        max_acc_id = record['Acc mean'].to_numpy().argmax()
        max_acc = record['Acc mean'].to_numpy().max()
        if max_acc < all_max_acc:
            print(f'End at {best_emb_ways} !!!')
            if if_concat:
                return '+'.join(best_emb_ways)
            else:
                return '/'.join(best_emb_ways)
        else:
            all_max_acc = max_acc
            best_emb_ways = selected_emb_ways
            selected_train_embs.pop(max_acc_id)
            selected_emb_ways.pop(max_acc_id)

def fast_ensemble(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    train_embs = tr_embs[:10]
    test_embs = tt_embs[:10]
    all_emb_ways = emb_ways[:10]
    if dataset_name == 'ATP':
        features_id = [1, 2, 5, 6, 7, 8, 9, 10]
    elif dataset_name == 'MD':
        features_id = [1, 2, 4, 5, 6, 7, 8, 9, 10]
    elif dataset_name == 'RD':
        features_id = [1, 2, 3, 5, 7, 10]
    FS_train_embs, FS_test_embs, FS_emb_ways = [], [], []
    for fid in features_id:
        FS_train_embs.append(train_embs[fid-1])
        FS_test_embs.append(test_embs[fid-1])
        FS_emb_ways.append(all_emb_ways[fid-1])
    # if concat
    FS_train_embs = np.concatenate(FS_train_embs, axis=1)
    FS_test_embs = np.concatenate(FS_test_embs, axis=1)
    FS_emb_ways = '+'.join(FS_emb_ways)
    # _ = get_ind_test_probs(FS_train_embs, FS_test_embs, FS_emb_ways, tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{dataset_name}_FS', True, 'FS_ensmeble_fast_results')
    # _ = get_ind_test_probs([FS_train_embs], [FS_test_embs], [FS_emb_ways], tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{dataset_name}_FS_concat', True, 'FS_concat_fast_results')
    # _ = get_ind_test_probs(FS_train_embs, FS_test_embs, FS_emb_ways, tr_labels, tt_labels, f'results/probs_txt/backward_test_probs/{dataset_name}_BS', True, 'BS_ensemble_fast_results')
    # _ = get_ind_test_probs([FS_train_embs], [FS_test_embs], [FS_emb_ways], tr_labels, tt_labels, f'results/probs_txt/backward_test_probs/{dataset_name}_BS_concat', True, 'BS_concat_fast_results')
    _ = get_ind_test_probs([FS_train_embs], [FS_test_embs], [FS_emb_ways], tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{dataset_name}_emb_{FS_emb_ways}', True, 'check_fast_results')

def test_if_RF_same(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    train_embs = tr_embs[:10]
    test_embs = tt_embs[:10]
    all_emb_ways = emb_ways[:10]
    combine_set = [[0, 9, 1, 2, 8, 5, 7, 3, 4, 6], [0, 8, 2, 7, 3, 1, 9, 5, 4, 6]]
    for cset in combine_set:
        c_train_embs = []
        c_test_embs = []
        c_emb_ways = []
        for cs in cset:
            c_train_embs.append(train_embs[cs])
            c_test_embs.append(test_embs[cs])
            c_emb_ways.append(all_emb_ways[cs])
        c_train_embs = np.concatenate(c_train_embs, axis=1)
        c_test_embs = np.concatenate(c_test_embs, axis=1)
        c_emb_ways = '+'.join(c_emb_ways)
        c_emb_ways = sort_emb_ways([c_emb_ways])
        _ = get_ind_test_probs([c_train_embs], [c_test_embs], c_emb_ways, tr_labels, tt_labels, f'results/probs_txt/forward_test_probs/{dataset_name}_test_same_{"_".join(c_emb_ways)}', True, f'{dataset_name}_test_same')

def watch_test_label_FS(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # only w2v
    train_embs = tr_embs[:10]
    test_embs = tt_embs[:10]
    all_emb_ways = emb_ways[:10]
    task_name = f'{dataset_name}_FS_watch_test_label_w2vs'
    # all
    # train_embs = tr_embs
    # test_embs = tt_embs
    # all_emb_ways = emb_ways
    # task_name = f'{dataset_name}_FS_all'
    selected_train_embs, selected_test_embs, selected_emb_ways = [], [], []
    all_max_acc = 0
    while len(all_emb_ways) > 0:
        print(selected_emb_ways)
        temp_record = []
        for add_i in range(len(all_emb_ways)):
            curr_train_embs = selected_train_embs + [train_embs[add_i]]
            curr_test_embs = selected_test_embs + [test_embs[add_i]]
            concat_train_embs = np.concatenate(curr_train_embs, axis=1)
            concat_test_embs = np.concatenate(curr_test_embs, axis=1)
            curr_emb_ways = selected_emb_ways + [all_emb_ways[add_i]]
            # _, t = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/watch_test_label_FS_atp/{task_name}_emb_{"_".join(curr_emb_ways)}', True, task_name + '_ensemble')
            _, t = get_ind_test_probs([concat_train_embs], [concat_test_embs], ['+'.join(curr_emb_ways)], tr_labels, tt_labels, f'results/probs_txt/watch_test_label_FS_atp/{task_name}_emb_{"+".join(curr_emb_ways)}', True, task_name + '_concat')
            temp_record.append(t)
        record = pd.concat(temp_record)
        if dataset_name == 'ATP':
            metric = 'Mcc'
        else:
            metric = 'Acc'
        max_acc_id = record[metric].to_numpy().argmax()
        max_acc = record[metric].to_numpy().max()
        if max_acc < all_max_acc:
            print(f'End at {selected_emb_ways} !!!')
            return selected_emb_ways
        else:
            all_max_acc = max_acc
            selected_train_embs.append(train_embs.pop(max_acc_id))
            selected_test_embs.append(test_embs.pop(max_acc_id))
            selected_emb_ways.append(all_emb_ways.pop(max_acc_id))

def watch_test_label_BS(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # only w2v
    train_embs = tr_embs[:10]
    test_embs = tt_embs[:10]
    all_emb_ways = emb_ways[:10]
    task_name = f'{dataset_name}_BS_watch_test_label_w2vs'
    # all
    # train_embs = tr_embs
    # test_embs = tt_embs
    # all_emb_ways = emb_ways
    # task_name = f'{dataset_name}_FS_all'
    selected_train_embs = train_embs
    selected_test_embs = test_embs
    selected_emb_ways = all_emb_ways
    all_max_acc = 0
    best_emb_ways = selected_emb_ways
    while len(selected_emb_ways) > 1:
        print(selected_emb_ways)
        temp_record = []
        for remove_i in range(len(selected_emb_ways)):
            curr_train_embs = selected_train_embs[:remove_i] + selected_train_embs[remove_i+1:]
            curr_test_embs = selected_test_embs[:remove_i] + selected_test_embs[remove_i+1:]
            curr_emb_ways = selected_emb_ways[:remove_i] + selected_emb_ways[remove_i+1:]
            concat_train_embs = np.concatenate(curr_train_embs, axis=1)
            concat_test_embs = np.concatenate(curr_test_embs, axis=1)
            # _, t = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/watch_test_label_BS_atp/{task_name}_emb_{"_".join(curr_emb_ways)}', True, task_name + '_ensemble')
            _, t = get_ind_test_probs([concat_train_embs], [concat_test_embs], ['+'.join(curr_emb_ways)], tr_labels, tt_labels, f'results/probs_txt/watch_test_label_BS_atp/{task_name}_emb_{"+".join(curr_emb_ways)}', True, task_name + '_concat')
            temp_record.append(t)
        record = pd.concat(temp_record)
        if dataset_name == 'ATP':
            metric = 'Mcc'
        else:
            metric = 'Acc'
        max_acc_id = record[metric].to_numpy().argmax()
        max_acc = record[metric].to_numpy().max()
        if max_acc < all_max_acc:
            print(f'End at {best_emb_ways} !!!')
            return best_emb_ways
        else:
            all_max_acc = max_acc
            best_emb_ways = selected_emb_ways
            selected_train_embs.pop(max_acc_id)
            selected_test_embs.pop(max_acc_id)
            selected_emb_ways.pop(max_acc_id)

def FS_keep_or_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    Like SFS method. However, I will try concat with selected embs instead of just adding it.
    '''
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    hc_tr_embs, hc_tt_embs, hc_emb_ways, _, _ = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    # only w2v
    # train_embs = tr_embs[:10]
    # test_embs = tt_embs[:10]
    # all_emb_ways = emb_ways[:10]
    # task_name = f'{dataset_name}_FS_keep_or_concat_w2vs'
    # only w2v+hc
    train_embs = tr_embs[:10] + hc_tr_embs
    test_embs = tt_embs[:10] + hc_tt_embs
    all_emb_ways = emb_ways[:10] + hc_emb_ways
    # task_name = f'{dataset_name}_FS_keep_or_concat_w2vs+hc'
    # train_embs, test_embs = select_in_feature_set(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, dataset_name)
    # task_name = f'{dataset_name}_FS_keep_or_concat_w2vs+hc_select_in_feature_set_diff_seed'
    # selected_hc_train_embs, selected_hc_test_embs, _ = select_in_feature_set(train_embs[10:15], test_embs[10:15], all_emb_ways[10:15], tr_labels, tt_labels, dataset_name)
    # for i in range(10, 15):
    #     train_embs[i] = selected_hc_train_embs[i-10]
    #     test_embs[i] = selected_hc_test_embs[i-10]
    #     all_emb_ways[i] = all_emb_ways[i] + '*'
    task_name = f'{dataset_name}_CESFS_w2v+hc-same-feature'
    selected_train_embs, selected_test_embs, selected_emb_ways = [], [], []
    all_max_acc = 0
    while len(all_emb_ways) > 1:
        print(selected_emb_ways)
        temp_record = []
        for add_i in range(len(all_emb_ways)):
            curr_train_embs = selected_train_embs + [train_embs[add_i]]
            curr_emb_ways = selected_emb_ways + [all_emb_ways[add_i]]
            sorted_emb_ways = sort_emb_ways(curr_emb_ways)
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{task_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            _, t = get_5fold_test_probs_fast(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name, 10)
            temp_record.append(t)
        record = pd.concat(temp_record)
        if dataset_name == 'ATP':
            metric = 'Mcc'
        else:
            metric = 'Acc'
        max_acc_id = record[f'{metric} mean'].to_numpy().argmax()
        max_acc = record[f'{metric} mean'].to_numpy().max()
        if len(selected_emb_ways) == 0:
            if max_acc < all_max_acc:
                print(f'End at {selected_emb_ways} !!!')
                return '/'.join(selected_emb_ways)
            else:
                all_max_acc = max_acc
                selected_train_embs.append(train_embs.pop(max_acc_id))
                selected_test_embs.append(test_embs.pop(max_acc_id))
                selected_emb_ways.append(all_emb_ways.pop(max_acc_id))
        else:
            temp_record2 = []
            for add_id in range(len(all_emb_ways)):
                for concat_id in range(len(selected_emb_ways)):
                    curr_train_embs = selected_train_embs[:concat_id] + selected_train_embs[concat_id+1:] + [np.concatenate([selected_train_embs[concat_id], train_embs[add_id]], axis=1)]
                    curr_emb_ways = selected_emb_ways[:concat_id] + selected_emb_ways[concat_id+1:] + [selected_emb_ways[concat_id] + '+' + all_emb_ways[add_id]]
                    sorted_emb_ways = sort_emb_ways(curr_emb_ways)
                    # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{task_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
                    # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
                    # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
                    _, t = get_5fold_test_probs_fast(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
                    # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name, 10)
                    temp_record2.append(t)
            record2 = pd.concat(temp_record2)
            if dataset_name == 'ATP':
                metric = 'Mcc'
            else:
                metric = 'Acc'
            max_concat_acc_id = record2[f'{metric} mean'].to_numpy().argmax()
            max_concat_acc = record2[f'{metric} mean'].to_numpy().max()
            if max_acc >= max_concat_acc:
                # keep
                if max_acc < all_max_acc:
                    print(f'End at {selected_emb_ways} !!!')
                    return '/'.join(selected_emb_ways)
                else:
                    all_max_acc = max_acc
                    selected_train_embs.append(train_embs.pop(max_acc_id))
                    selected_test_embs.append(test_embs.pop(max_acc_id))
                    selected_emb_ways.append(all_emb_ways.pop(max_acc_id))
            else:
                # concat
                if max_concat_acc < all_max_acc:
                    print(f'End at {selected_emb_ways} !!!')
                    return '/'.join(selected_emb_ways)
                else:
                    all_max_acc = max_concat_acc
                    add_id = max_concat_acc_id // len(selected_emb_ways)
                    concat_id = max_concat_acc_id % len(selected_emb_ways)
                    selected_train_embs = selected_train_embs[:concat_id] + selected_train_embs[concat_id+1:] + [np.concatenate([selected_train_embs[concat_id], train_embs[add_id]], axis=1)]
                    selected_test_embs = selected_test_embs[:concat_id] + selected_test_embs[concat_id+1:] + [np.concatenate([selected_test_embs[concat_id], test_embs[add_id]], axis=1)]
                    selected_emb_ways = selected_emb_ways[:concat_id] + selected_emb_ways[concat_id+1:] + [selected_emb_ways[concat_id] + '+' + all_emb_ways[add_id]]
                    train_embs.pop(add_id)
                    test_embs.pop(add_id)
                    all_emb_ways.pop(add_id)

def BS_keep_or_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    Like SBS method. However, I will try concat with selected embs instead of just dropping it.
    '''
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    hc_tr_embs, hc_tt_embs, hc_emb_ways, _, _ = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    # only w2v
    # train_embs = tr_embs[:10]
    # test_embs = tt_embs[:10]
    # all_emb_ways = emb_ways[:10]
    # task_name = f'{dataset_name}_BS_keep_or_concat_w2vs'
    # only w2v+hc
    train_embs = tr_embs[:10] + hc_tr_embs
    test_embs = tt_embs[:10] + hc_tt_embs
    all_emb_ways = emb_ways[:10] + hc_emb_ways
    # task_name = f'{dataset_name}_BS_keep_or_concat_w2vs+hc'
    # selected_hc_train_embs, selected_hc_test_embs, _ = select_in_feature_set(train_embs[10:15], test_embs[10:15], all_emb_ways[10:15], tr_labels, tt_labels, dataset_name)
    # for i in range(10, 15):
    #     train_embs[i] = selected_hc_train_embs[i-10]
    #     test_embs[i] = selected_hc_test_embs[i-10]
    #     all_emb_ways[i] = all_emb_ways[i] + '*'
    task_name = f'{dataset_name}_CDSBS_w2v+hc-same-feature'
    all_max_acc = 0
    while len(all_emb_ways) > 1:
        temp_record, temp_record2 = [], []
        for remove_i in range(len(all_emb_ways)):
            curr_train_embs = train_embs[:remove_i] + train_embs[remove_i+1:]
            curr_emb_ways = all_emb_ways[:remove_i] + all_emb_ways[remove_i+1:]
            sorted_emb_ways = sort_emb_ways(curr_emb_ways)
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{task_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            _, t = get_5fold_test_probs_fast(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            temp_record.append(t)
            # try to concat instead of dropping
            for concat_id in range(len(curr_emb_ways)):
                concat_train_embs = curr_train_embs[:concat_id] + curr_train_embs[concat_id+1:] + [np.concatenate([curr_train_embs[concat_id], train_embs[remove_i]], axis=1)]
                concat_emb_ways = curr_emb_ways[:concat_id] + curr_emb_ways[concat_id+1:] + [curr_emb_ways[concat_id] + '+' + all_emb_ways[remove_i]]
                sorted_emb_ways = sort_emb_ways(concat_emb_ways)
                # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{task_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
                # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
                # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
                _, t = get_5fold_test_probs_fast(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
                temp_record2.append(t)
        record = pd.concat(temp_record)
        record2 = pd.concat(temp_record2)
        if dataset_name == 'ATP':
            metric = 'Mcc'
        else:
            metric = 'Acc'
        max_acc_id = record[f'{metric} mean'].to_numpy().argmax()
        max_acc = record[f'{metric} mean'].to_numpy().max()
        max_concat_acc_id = record2[f'{metric} mean'].to_numpy().argmax()
        max_concat_acc = record2[f'{metric} mean'].to_numpy().max()
        if all_max_acc > max_acc and all_max_acc > max_concat_acc:
            print(all_emb_ways)
            return '/'.join(all_emb_ways)
        elif max_acc >= max_concat_acc:
            all_max_acc = max_acc
            train_embs = train_embs[:max_acc_id] + train_embs[max_acc_id+1:]
            test_embs = test_embs[:max_acc_id] + test_embs[max_acc_id+1:]
            all_emb_ways = all_emb_ways[:max_acc_id] + all_emb_ways[max_acc_id+1:]
        else:
            all_max_acc = max_concat_acc
            remove_id = max_concat_acc_id // (len(all_emb_ways) - 1)
            concat_id = max_concat_acc_id % (len(all_emb_ways) - 1)
            curr_train_embs = train_embs[:remove_id] + train_embs[remove_id+1:]
            train_embs = curr_train_embs[:concat_id] + curr_train_embs[concat_id+1:] + [np.concatenate([curr_train_embs[concat_id], train_embs[remove_id]], axis=1)]
            curr_test_embs = test_embs[:remove_id] + test_embs[remove_id+1:]
            test_embs = curr_test_embs[:concat_id] + curr_test_embs[concat_id+1:] + [np.concatenate([curr_test_embs[concat_id], test_embs[remove_id]], axis=1)]
            curr_emb_ways = all_emb_ways[:remove_id] + all_emb_ways[remove_id+1:]
            all_emb_ways = curr_emb_ways[:concat_id] + curr_emb_ways[concat_id+1:] + [curr_emb_ways[concat_id] + '+' + all_emb_ways[remove_id]]

def BS_keep_or_drop_or_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    Like SBS concat method. 
    It start with all features concat, and I call it main combination.
    In the first step, I will try to take off one embedding just like SBS, however, I will try to ensemble it with main combination, like (emb1 + emb2 + emb3 + emb4) => (emb1 + emb2 + emb3) / (emb4), and I will call (emb4) "sub combination".
    So before each feature was selected to be dropped, it will try to be a new sub combination or try ro concat with current sub combinations. Like:
    (emb1 + emb2 + emb3) / (emb4) => 
    1. (emb2 + emb3) / (emb4) / (emb1)
    2. (emb2 + emb3) / (emb1 + emb4)
    '''
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    hc_tr_embs, hc_tt_embs, hc_emb_ways, _, _ = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    # only w2v
    # train_embs = tr_embs[:10]
    # test_embs = tt_embs[:10]
    # all_emb_ways = emb_ways[:10]
    # task_name = f'{dataset_name}_BS_keep_or_drop_or_concat_w2vs_again'
    # only w2v+hc
    train_embs = tr_embs[:10] + hc_tr_embs
    test_embs = tt_embs[:10] + hc_tt_embs
    all_emb_ways = emb_ways[:10] + hc_emb_ways
    # task_name = f'{dataset_name}_BS_keep_or_drop_or_concat_w2vs_hc'
    # selected_hc_train_embs, selected_hc_test_embs, _ = select_in_feature_set(train_embs[10:15], test_embs[10:15], all_emb_ways[10:15], tr_labels, tt_labels, dataset_name)
    # for i in range(10, 15):
    #     train_embs[i] = selected_hc_train_embs[i-10]
    #     test_embs[i] = selected_hc_test_embs[i-10]
    #     all_emb_ways[i] = all_emb_ways[i] + '*'
    task_name = f'{dataset_name}_CDESBS_w2v+hc-same-feature'
    all_max_acc = 0
    sub_train_embs, sub_test_embs, sub_emb_ways = [], [], []

    while len(all_emb_ways) > 1:
        temp_record_drop, temp_record_keep, temp_record_concat = [], [], []
        # drop
        for drop_i in range(len(all_emb_ways)):
            curr_train_embs = train_embs[:drop_i] + train_embs[drop_i+1:]
            concat_train_embs = [np.concatenate(curr_train_embs, axis=1)] + sub_train_embs
            curr_emb_ways = all_emb_ways[:drop_i] + all_emb_ways[drop_i+1:]
            concat_emb_ways = ['+'.join(curr_emb_ways)] + sub_emb_ways
            sorted_emb_ways = sort_emb_ways(concat_emb_ways)
            # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{dataset_name}_BS_keep_or_concat_w2vs_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)            
            # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)            
            _, t = get_5fold_test_probs_fast(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)            
            temp_record_drop.append(t)
            # keep
            keep_concat_train_embs = concat_train_embs + [train_embs[drop_i]]
            keep_concat_emb_ways = concat_emb_ways + [all_emb_ways[drop_i]]
            keep_sorted_emb_ways = sort_emb_ways(keep_concat_emb_ways)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{dataset_name}_BS_keep_or_concat_w2vs_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            temp_record_keep.append(t)
            # concat with sub
            if len(sub_emb_ways) > 0:
                for concat_i in range(len(sub_emb_ways)):
                    concat_with_sub_concat_train_embs = concat_train_embs[:concat_i+1] + concat_train_embs[concat_i+2:] + [np.concatenate([concat_train_embs[concat_i+1], train_embs[drop_i]], axis=1)]
                    concat_with_sub_concat_emb_ways = concat_emb_ways[:concat_i+1] + concat_emb_ways[concat_i+2:] + [concat_emb_ways[concat_i+1] + '+' + all_emb_ways[drop_i]]
                    concat_with_sub_sorted_emb_ways = sort_emb_ways(concat_with_sub_concat_emb_ways)
                    # _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{dataset_name}_BS_keep_or_concat_w2vs_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                    # _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                    # _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                    _, t = get_5fold_test_probs_fast(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                    temp_record_concat.append(t)
        record_drop = pd.concat(temp_record_drop)
        record_keep = pd.concat(temp_record_keep)
        if len(temp_record_concat) > 0:
            record_concat = pd.concat(temp_record_concat)
        if dataset_name == 'ATP':
            metric = 'Mcc'
        else:
            metric = 'Acc'
        max_acc_id_drop = record_drop[f'{metric} mean'].to_numpy().argmax()
        max_acc_drop = record_drop[f'{metric} mean'].to_numpy().max()
        max_acc_id_keep = record_keep[f'{metric} mean'].to_numpy().argmax()
        max_acc_keep = record_keep[f'{metric} mean'].to_numpy().max()
        if len(temp_record_concat) > 0:
            max_acc_id_concat = record_concat[f'{metric} mean'].to_numpy().argmax()
            max_acc_concat = record_concat[f'{metric} mean'].to_numpy().max()
        else:
            max_acc_id_concat = 0
            max_acc_concat = -1
        action_id = np.argmax([all_max_acc, max_acc_drop, max_acc_keep, max_acc_concat])
        if action_id == 0:
            # stop selection
            print(['+'.join(all_emb_ways)] + sub_emb_ways)
            return '/'.join(['+'.join(all_emb_ways)] + sub_emb_ways)
        elif action_id == 1:
            # drop
            all_max_acc = max_acc_drop
            train_embs.pop(max_acc_id_drop)
            test_embs.pop(max_acc_id_drop)
            all_emb_ways.pop(max_acc_id_drop)
        elif action_id == 2:
            # keep
            all_max_acc = max_acc_keep
            sub_train_embs.append(train_embs.pop(max_acc_id_keep))
            sub_test_embs.append(test_embs.pop(max_acc_id_keep))
            sub_emb_ways.append(all_emb_ways.pop(max_acc_id_keep))
        elif action_id == 3:
            # concat with sub
            all_max_acc = max_acc_concat
            drop_id = max_acc_id_concat // len(sub_emb_ways)
            concat_id = max_acc_id_concat % len(sub_emb_ways)
            sub_train_embs[concat_id] = np.concatenate([sub_train_embs[concat_id], train_embs.pop(drop_id)], axis=1)
            sub_test_embs[concat_id] = np.concatenate([sub_test_embs[concat_id], test_embs.pop(drop_id)], axis=1)
            sub_emb_ways[concat_id] = sub_emb_ways[concat_id] + '+' + all_emb_ways.pop(drop_id)

def BS_keep_or_drop(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    Like BS_keep_or_drop_or_concat, but no concat
    '''
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    hc_tr_embs, hc_tt_embs, hc_emb_ways, _, _ = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    # only w2v
    # train_embs = tr_embs[:10]
    # test_embs = tt_embs[:10]
    # all_emb_ways = emb_ways[:10]
    # task_name = f'{dataset_name}_BS_keep_or_drop_w2vs'
    # only w2v+hc
    train_embs = tr_embs[:10] + hc_tr_embs
    test_embs = tt_embs[:10] + hc_tt_embs
    all_emb_ways = emb_ways[:10] + hc_emb_ways
    # task_name = f'{dataset_name}_BS_keep_or_drop_w2vs_hc'
    # selected_hc_train_embs, selected_hc_test_embs, _ = select_in_feature_set(train_embs[10:15], test_embs[10:15], all_emb_ways[10:15], tr_labels, tt_labels, dataset_name)
    # for i in range(10, 15):
    #     train_embs[i] = selected_hc_train_embs[i-10]
    #     test_embs[i] = selected_hc_test_embs[i-10]
    #     all_emb_ways[i] = all_emb_ways[i] + '*'
    task_name = f'{dataset_name}_DESBS_w2v+hc-same-feature'
    all_max_acc = 0
    sub_train_embs, sub_test_embs, sub_emb_ways = [], [], []

    while len(all_emb_ways) > 1:
        temp_record_drop, temp_record_keep = [], []
        # drop
        for drop_i in range(len(all_emb_ways)):
            curr_train_embs = train_embs[:drop_i] + train_embs[drop_i+1:]
            concat_train_embs = [np.concatenate(curr_train_embs, axis=1)] + sub_train_embs
            curr_emb_ways = all_emb_ways[:drop_i] + all_emb_ways[drop_i+1:]
            concat_emb_ways = ['+'.join(curr_emb_ways)] + sub_emb_ways
            sorted_emb_ways = sort_emb_ways(concat_emb_ways)
            # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)            
            # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)            
            _, t = get_5fold_test_probs_fast(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"_".join(sorted_emb_ways)}', True, task_name)            
            temp_record_drop.append(t)
            # keep
            keep_concat_train_embs = concat_train_embs + [train_embs[drop_i]]
            keep_concat_emb_ways = concat_emb_ways + [all_emb_ways[drop_i]]
            keep_sorted_emb_ways = sort_emb_ways(keep_concat_emb_ways)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{dataset_name}_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            _, t = get_5fold_test_probs_fast(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            temp_record_keep.append(t)
        record_drop = pd.concat(temp_record_drop)
        record_keep = pd.concat(temp_record_keep)
        if dataset_name == 'ATP':
            metric = 'Mcc'
        else:
            metric = 'Acc'
        max_acc_id_drop = record_drop[f'{metric} mean'].to_numpy().argmax()
        max_acc_drop = record_drop[f'{metric} mean'].to_numpy().max()
        max_acc_id_keep = record_keep[f'{metric} mean'].to_numpy().argmax()
        max_acc_keep = record_keep[f'{metric} mean'].to_numpy().max()
        action_id = np.argmax([all_max_acc, max_acc_drop, max_acc_keep])
        if action_id == 0:
            # stop selection
            print(['+'.join(all_emb_ways)] + sub_emb_ways)
            return '/'.join(['+'.join(all_emb_ways)] + sub_emb_ways)
        elif action_id == 1:
            # drop
            all_max_acc = max_acc_drop
            train_embs.pop(max_acc_id_drop)
            test_embs.pop(max_acc_id_drop)
            all_emb_ways.pop(max_acc_id_drop)
        elif action_id == 2:
            # keep
            all_max_acc = max_acc_keep
            sub_train_embs.append(train_embs.pop(max_acc_id_keep))
            sub_test_embs.append(test_embs.pop(max_acc_id_keep))
            sub_emb_ways.append(all_emb_ways.pop(max_acc_id_keep))

def BS_concat_first_keep_or_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    Like SBS_keep_or_drop_or_concat, but do not drop.
    '''
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    hc_tr_embs, hc_tt_embs, hc_emb_ways, _, _ = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    # only w2v
    # train_embs = tr_embs[:10]
    # test_embs = tt_embs[:10]
    # all_emb_ways = emb_ways[:10]
    # task_name = f'{dataset_name}_BS_concat_first_keep_or_concat_w2vs'
    # only w2v+hc
    train_embs = tr_embs[:10] + hc_tr_embs
    test_embs = tt_embs[:10] + hc_tt_embs
    all_emb_ways = emb_ways[:10] + hc_emb_ways
    # task_name = f'{dataset_name}_BS_concat_first_keep_or_concat_w2vs+hc'
    # selected_hc_train_embs, selected_hc_test_embs, _ = select_in_feature_set(train_embs[10:15], test_embs[10:15], all_emb_ways[10:15], tr_labels, tt_labels, dataset_name)
    # for i in range(10, 15):
    #     train_embs[i] = selected_hc_train_embs[i-10]
    #     test_embs[i] = selected_hc_test_embs[i-10]
    #     all_emb_ways[i] = all_emb_ways[i] + '*'
    task_name = f'{dataset_name}_CESBS_w2v+hc-same-feature'
    all_max_acc = 0
    sub_train_embs, sub_test_embs, sub_emb_ways = [], [], []

    while len(all_emb_ways) > 1:
        temp_record_keep, temp_record_concat = [], []
        for drop_i in range(len(all_emb_ways)):
            curr_train_embs = train_embs[:drop_i] + train_embs[drop_i+1:]
            concat_train_embs = [np.concatenate(curr_train_embs, axis=1)] + sub_train_embs
            curr_emb_ways = all_emb_ways[:drop_i] + all_emb_ways[drop_i+1:]
            concat_emb_ways = ['+'.join(curr_emb_ways)] + sub_emb_ways
            # keep
            keep_concat_train_embs = concat_train_embs + [train_embs[drop_i]]
            keep_concat_emb_ways = concat_emb_ways + [all_emb_ways[drop_i]]
            keep_sorted_emb_ways = sort_emb_ways(keep_concat_emb_ways)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{dataset_name}_BS_keep_or_concat_w2vs_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            _, t = get_5fold_test_probs_fast(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            temp_record_keep.append(t)
            # concat with sub
            if len(sub_emb_ways) > 0:
                for concat_i in range(len(sub_emb_ways)):
                    concat_with_sub_concat_train_embs = concat_train_embs[:concat_i+1] + concat_train_embs[concat_i+2:] + [np.concatenate([concat_train_embs[concat_i+1], train_embs[drop_i]], axis=1)]
                    concat_with_sub_concat_emb_ways = concat_emb_ways[:concat_i+1] + concat_emb_ways[concat_i+2:] + [concat_emb_ways[concat_i+1] + '+' + all_emb_ways[drop_i]]
                    concat_with_sub_sorted_emb_ways = sort_emb_ways(concat_with_sub_concat_emb_ways)
                    # _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{dataset_name}_BS_keep_or_concat_w2vs_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                    # _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                    # _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                    _, t = get_5fold_test_probs_fast(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                    temp_record_concat.append(t)
        record_keep = pd.concat(temp_record_keep)
        if len(temp_record_concat) > 0:
            record_concat = pd.concat(temp_record_concat)
        if dataset_name == 'ATP':
            metric = 'Mcc'
        else:
            metric = 'Acc'
        max_acc_id_keep = record_keep[f'{metric} mean'].to_numpy().argmax()
        max_acc_keep = record_keep[f'{metric} mean'].to_numpy().max()
        if len(temp_record_concat) > 0:
            max_acc_id_concat = record_concat[f'{metric} mean'].to_numpy().argmax()
            max_acc_concat = record_concat[f'{metric} mean'].to_numpy().max()
        else:
            max_acc_id_concat = 0
            max_acc_concat = -1
        action_id = np.argmax([all_max_acc, max_acc_keep, max_acc_concat])
        if action_id == 0:
            # stop selection
            print(['+'.join(all_emb_ways)] + sub_emb_ways)
            return '/'.join(['+'.join(all_emb_ways)] + sub_emb_ways)
        elif action_id == 1:
            # keep
            all_max_acc = max_acc_keep
            sub_train_embs.append(train_embs.pop(max_acc_id_keep))
            sub_test_embs.append(test_embs.pop(max_acc_id_keep))
            sub_emb_ways.append(all_emb_ways.pop(max_acc_id_keep))
        elif action_id == 2:
            # concat with sub
            all_max_acc = max_acc_concat
            drop_id = max_acc_id_concat // len(sub_emb_ways)
            concat_id = max_acc_id_concat % len(sub_emb_ways)
            sub_train_embs[concat_id] = np.concatenate([sub_train_embs[concat_id], train_embs.pop(drop_id)], axis=1)
            sub_test_embs[concat_id] = np.concatenate([sub_test_embs[concat_id], test_embs.pop(drop_id)], axis=1)
            sub_emb_ways[concat_id] = sub_emb_ways[concat_id] + '+' + all_emb_ways.pop(drop_id)

def BS_concat_first_keep(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    Like SBS_keep_or_drop_or_concat, but do not drop and concat.
    '''
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    hc_tr_embs, hc_tt_embs, hc_emb_ways, _, _ = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    # only w2v
    # train_embs = tr_embs[:10]
    # test_embs = tt_embs[:10]
    # all_emb_ways = emb_ways[:10]
    # task_name = f'{dataset_name}_BS_concat_first_keep_w2vs'
    # only w2v+hc
    train_embs = tr_embs[:10] + hc_tr_embs
    test_embs = tt_embs[:10] + hc_tt_embs
    all_emb_ways = emb_ways[:10] + hc_emb_ways
    # task_name = f'{dataset_name}_BS_concat_first_keep_w2vs+hc'
    # selected_hc_train_embs, selected_hc_test_embs, _ = select_in_feature_set(train_embs[10:15], test_embs[10:15], all_emb_ways[10:15], tr_labels, tt_labels, dataset_name)
    # for i in range(10, 15):
    #     train_embs[i] = selected_hc_train_embs[i-10]
    #     test_embs[i] = selected_hc_test_embs[i-10]
    #     all_emb_ways[i] = all_emb_ways[i] + '*'
    task_name = f'{dataset_name}_ESBS_w2v+hc-same-feature'
    all_max_acc = 0
    sub_train_embs, sub_test_embs, sub_emb_ways = [], [], []

    while len(all_emb_ways) > 1:
        temp_record_keep, temp_record_concat = [], []
        for drop_i in range(len(all_emb_ways)):
            curr_train_embs = train_embs[:drop_i] + train_embs[drop_i+1:]
            concat_train_embs = [np.concatenate(curr_train_embs, axis=1)] + sub_train_embs
            curr_emb_ways = all_emb_ways[:drop_i] + all_emb_ways[drop_i+1:]
            concat_emb_ways = ['+'.join(curr_emb_ways)] + sub_emb_ways
            # keep
            keep_concat_train_embs = concat_train_embs + [train_embs[drop_i]]
            keep_concat_emb_ways = concat_emb_ways + [all_emb_ways[drop_i]]
            keep_sorted_emb_ways = sort_emb_ways(keep_concat_emb_ways)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{dataset_name}_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc/{dataset_name}_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            _, t = get_5fold_test_probs_fast(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            temp_record_keep.append(t)
        record_keep = pd.concat(temp_record_keep)
        if dataset_name == 'ATP':
            metric = 'Mcc'
        else:
            metric = 'Acc'
        max_acc_id_keep = record_keep[f'{metric} mean'].to_numpy().argmax()
        max_acc_keep = record_keep[f'{metric} mean'].to_numpy().max()
        action_id = np.argmax([all_max_acc, max_acc_keep])
        if action_id == 0:
            # stop selection
            print(['+'.join(all_emb_ways)] + sub_emb_ways)
            return '/'.join(['+'.join(all_emb_ways)] + sub_emb_ways)
        elif action_id == 1:
            # keep
            all_max_acc = max_acc_keep
            sub_train_embs.append(train_embs.pop(max_acc_id_keep))
            sub_test_embs.append(test_embs.pop(max_acc_id_keep))
            sub_emb_ways.append(all_emb_ways.pop(max_acc_id_keep))

def BS_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    Like SBS_keep_or_drop_or_concat, but do not drop.
    '''
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    hc_tr_embs, hc_tt_embs, hc_emb_ways, _, _ = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    # only w2v
    # train_embs = tr_embs[:10]
    # test_embs = tt_embs[:10]
    # all_emb_ways = emb_ways[:10]
    # task_name = f'{dataset_name}_BS_concat_w2vs'
    # only w2v+hc
    train_embs = tr_embs[:10] + hc_tr_embs
    test_embs = tt_embs[:10] + hc_tt_embs
    all_emb_ways = emb_ways[:10] + hc_emb_ways
    # task_name = f'{dataset_name}_BS_concat_w2vs+hc'
    # selected_hc_train_embs, selected_hc_test_embs, _ = select_in_feature_set(train_embs[10:15], test_embs[10:15], all_emb_ways[10:15], tr_labels, tt_labels, dataset_name)
    # for i in range(10, 15):
    #     train_embs[i] = selected_hc_train_embs[i-10]
    #     test_embs[i] = selected_hc_test_embs[i-10]
    #     all_emb_ways[i] = all_emb_ways[i] + '*'
    task_name = f'{dataset_name}_CSBS_w2v+hc-same-feature'
    all_max_acc = 0
    while len(all_emb_ways) > 1:
        temp_record_concat = []
        for drop_i in range(len(all_emb_ways)):
            curr_train_embs = train_embs[:drop_i] + train_embs[drop_i+1:]
            curr_emb_ways = all_emb_ways[:drop_i] + all_emb_ways[drop_i+1:]
            # concat with each
            for concat_i in range(len(curr_emb_ways)):
                concat_with_sub_curr_train_embs = curr_train_embs[:concat_i] + curr_train_embs[concat_i+1:] + [np.concatenate([curr_train_embs[concat_i], train_embs[drop_i]], axis=1)]
                concat_with_sub_curr_emb_ways = curr_emb_ways[:concat_i] + curr_emb_ways[concat_i+1:] + [curr_emb_ways[concat_i] + '+' + all_emb_ways[drop_i]]
                concat_with_sub_sorted_emb_ways = sort_emb_ways(concat_with_sub_curr_emb_ways)
                # _, t = get_5fold_test_probs(concat_with_sub_curr_train_embs, concat_with_sub_curr_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{dataset_name}_BS_keep_or_concat_w2vs_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                # _, t = get_5fold_test_probs(concat_with_sub_curr_train_embs, concat_with_sub_curr_emb_ways, tr_labels, f'results/probs_txt/w2vs_hc/{dataset_name}_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                # _, t = get_5fold_test_probs(concat_with_sub_curr_train_embs, concat_with_sub_curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{dataset_name}_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                _, t = get_5fold_test_probs_fast(concat_with_sub_curr_train_embs, concat_with_sub_curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{dataset_name}_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                temp_record_concat.append(t)
        record_concat = pd.concat(temp_record_concat)
        if dataset_name == 'ATP':
            metric = 'Mcc'
        else:
            metric = 'Acc'
        max_acc_id_concat = record_concat[f'{metric} mean'].to_numpy().argmax()
        max_acc_concat = record_concat[f'{metric} mean'].to_numpy().max()
        action_id = np.argmax([all_max_acc, max_acc_concat])
        if action_id == 0:
            # stop selection
            print('/'.join(all_emb_ways))
            return '/'.join(all_emb_ways)
        elif action_id == 1:
            # concat with sub
            all_max_acc = max_acc_concat
            drop_id = max_acc_id_concat // (len(all_emb_ways) - 1)
            concat_id = max_acc_id_concat % (len(all_emb_ways) - 1)
            curr_train_embs = train_embs[:drop_id] + train_embs[drop_id+1:]
            curr_emb_ways = all_emb_ways[:drop_id] + all_emb_ways[drop_id+1:]
            train_embs = curr_train_embs[:concat_id] + curr_train_embs[concat_id+1:] + [np.concatenate([train_embs[drop_id], curr_train_embs[concat_id]], axis = 1)]
            all_emb_ways = curr_emb_ways[:concat_id] + curr_emb_ways[concat_id+1:] + [all_emb_ways[drop_id] + '+' + curr_emb_ways[concat_id]]

def fast_ensemble2(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    hc_tr_embs, hc_tt_embs, hc_emb_ways, _, _ = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    train_embs = tr_embs[:10] + hc_tr_embs
    test_embs = tt_embs[:10] + hc_tt_embs
    all_emb_ways = emb_ways[:10] + hc_emb_ways
    old = {
        # train_embs = tr_embs
        # test_embs = tt_embs
        # all_emb_ways = emb_ways
        # task_name = 'fast_train_on_MD_test_on_ATP'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '1+2+3+4+5+6+7+8+9+10'}, 
        #         {'dataset_name': 'MD', 'combine': '1+2+3+4+5+6+7+8+9+10'},
        #         {'dataset_name': 'RD', 'combine': '1+2+3+4+5+6+7+8+9+10'}]
        # task_name = 'fast_combine_BS'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '1+5/2+7/4+6'}, 
                # {'dataset_name': 'MD', 'combine': '2/4+6+7+8/5'},
                # {'dataset_name': 'RD', 'combine': '1/2/3/5'}]
        # task_name = 'fast_combine_FS'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '1+5/6+7+8'}, 
                # {'dataset_name': 'MD', 'combine': '1+3/2/5+8'},
                # {'dataset_name': 'RD', 'combine': '1/2/3+9/5'}]
        # task_name = 'fast_combine_drop_keep_concat'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '1+2+3+4+7+8/5+6+9+10'}, 
                # {'dataset_name': 'MD', 'combine': '1+3+4+5+6+7+9+10/2'},
                # {'dataset_name': 'RD', 'combine': '1/2/3+4+6+7+9+10/5'}]
        # task_name = 'fast_combine_concat_first_keep_concat'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '1+2+3+4+7+8/5+6+9+10'}, 
                # {'dataset_name': 'MD', 'combine': '1/2/3+4+5+6+7+8+9+10'},
                # {'dataset_name': 'RD', 'combine': '1/2+5/3+4+6+7+8+9+10'}]
        # task_name = 'fast_combine_BS_concat'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '1+2+3+4+5+7+9+10/6+8'}, 
                # {'dataset_name': 'MD', 'combine': '1+3/2/4+6+8+10/5/7/9'},
                # {'dataset_name': 'RD', 'combine': '1/2+5/3+4+6+7+8+9+10'}]
        # task_name = 'fast_combine_FS_ensemble_to_CESFS'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '1+6/2+7/5'}, 
                # {'dataset_name': 'MD', 'combine': '1+6/2/3/5'},
                # {'dataset_name': 'RD', 'combine': '1/2/3+9/5'}]
        # task_name = 'fast_combine_SFS_concat_to_CDESBS'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '3+6/5'}, 
                # {'dataset_name': 'MD', 'combine': '1+3+5/2'},
                # {'dataset_name': 'RD', 'combine': '1/2/3+7/5'}]
        # task_name = 'fast_combine_SBS_concat_to_CDESBS'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '1+5/2+7/6+8'}, 
                # {'dataset_name': 'MD', 'combine': '1+4+5+6+7+8+10/2'},
                # {'dataset_name': 'RD', 'combine': '2/3/5+9'}]
        # task_name = 'fast_combine_SFS_concat_ensemble_SFS_ensemble'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '5/6/7/3+5+6'}, 
                # {'dataset_name': 'MD', 'combine': '1/2/3/5/1+2+3+5'},
                # {'dataset_name': 'RD', 'combine': '1/2/3/5/1+2+3+5+7+10'}]
        # task_name = 'fast_combine_SBS_concat_ensemble_SBS_ensemble'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '5/6/7/1+2+5+6+7+8+9+10'}, 
                # {'dataset_name': 'MD', 'combine': '1/2/3/5/1+2+4+5+6+7+8+9+10'},
                # {'dataset_name': 'RD', 'combine': '1/2/3/5/2+3+5+9'}]
        # task_name = 'fast_combine_two_stage'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '1+2+7/4+5/6'}, 
        #         {'dataset_name': 'MD', 'combine': '1+4/2/3/5'},
        #         {'dataset_name': 'RD', 'combine': '1/2/3/5'},
        #         {'dataset_name': 'ATP', 'combine': '5/6'}, 
        #         {'dataset_name': 'MD', 'combine': '1+2+3+5'},
        #         {'dataset_name': 'RD', 'combine': '1+2+3+5+10'},
        #         {'dataset_name': 'ATP', 'combine': '1+6/2+7/5'}, 
        #         {'dataset_name': 'MD', 'combine': '1+4+6+7+8/2/5'},
        #         {'dataset_name': 'RD', 'combine': '2/3+5'},
        #         ]
        # task_name = 'fast_combine_all_FS_ensemble'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '5/6/7/3+5+6/5/6/7/1+2+5+6+7+8+9+10'}, 
        #         {'dataset_name': 'MD', 'combine': '1/2/3/5/1+2+3+5/1/2/3/5/1+2+4+5+6+7+8+9+10'},
        #         {'dataset_name': 'RD', 'combine': '1/2/3/5/1+2+3+5+7+10/1/2/3/5/2+3+5+9'}]
        # task_name = 'fast_combine_1-10mer_w2vs_concat_baseline_best_est_val'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '1/2/3/4/5/6/7/8/9/10', 'best_est': 200}, 
        #         {'dataset_name': 'MD', 'combine': '1/2/3/4/5/6/7/8/9/10', 'best_est': 2000},
        #         {'dataset_name': 'RD', 'combine': '1/2/3/4/5/6/7/8/9/10', 'best_est': 2000},
        #         {'dataset_name': 'ATP', 'combine': '1+2+3+4+5+6+7+8+9+10', 'best_est': 100}, 
        #         {'dataset_name': 'MD', 'combine': '1+2+3+4+5+6+7+8+9+10', 'best_est': 300},
        #         {'dataset_name': 'RD', 'combine': '1+2+3+4+5+6+7+8+9+10', 'best_est': 1000},
        #         ]
        # task_name = 'fast_combine_by_feature_importance'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '5+6+7+8+9+10'}, 
        #         {'dataset_name': 'MD', 'combine': '2+3+4+5+7+8'},
        #         {'dataset_name': 'RD', 'combine': '2+3+4+5'}]
        # task_name = 'fast_combine_w2vs+hc-plus'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '1+2+3+4+5+6+7+8+9+10+14+15+16+17+18+19+20+21'}, 
        #         {'dataset_name': 'MD', 'combine': '1+2+3+4+5+6+7+8+9+10+14+15+16+17+18+19+20+21'},
        #         {'dataset_name': 'RD', 'combine': '1+2+3+4+5+6+7+8+9+10+14+15+16+17+18+19+20+21'},
        #         {'dataset_name': 'ATP', 'combine': '1/2/3/4/5/6/7/8/9/10/14/15/16/17/18/19/20/21'}, 
        #         {'dataset_name': 'MD', 'combine': '1/2/3/4/5/6/7/8/9/10/14/15/16/17/18/19/20/21'},
        #         {'dataset_name': 'RD', 'combine': '1/2/3/4/5/6/7/8/9/10/14/15/16/17/18/19/20/21'},
        #         ]
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '15+16+17+19+20+21'}, 
        #         {'dataset_name': 'MD', 'combine': '15+16+17+19+20+21'},
        #         {'dataset_name': 'RD', 'combine': '15+16+17+19+20+21'},
        #         {'dataset_name': 'ATP', 'combine': '15/16/17/19/20/21'}, 
        #         {'dataset_name': 'MD', 'combine': '15/16/17/19/20/21'},
        #         {'dataset_name': 'RD', 'combine': '15/16/17/19/20/21'},
        #         ]
        # task_name = 'fast_combine_w2vs+hc_FS'
        # ensemble_list = [
        #         {'dataset_name': 'ATP', 'combine': '5/6/7/18'}, 
        #         {'dataset_name': 'ATP', 'combine': '1/2/3/4/5/6/8/9/10/15/16/18'}, 
        #         {'dataset_name': 'ATP', 'combine': '1+2+5+6+16'}, 
        #         {'dataset_name': 'ATP', 'combine': '1+2+3+4+5+6+7+8+9+14+15+16+17+18'}, 
        #         {'dataset_name': 'MD', 'combine': '2/4/5/14/15/16/17/18'},
        #         {'dataset_name': 'MD', 'combine': '2/4/7/14/15/16/17/18'},
        #         {'dataset_name': 'MD', 'combine': '2/5/16'},
        #         {'dataset_name': 'MD', 'combine': '1+2+3+4+5+6+7+9+10+14+15+16+17+18'},
        #         {'dataset_name': 'RD', 'combine': '2/3/5/15'},
        #         {'dataset_name': 'RD', 'combine': '1/2/3/5/7/9/14/15/16/17'},
        #         {'dataset_name': 'RD', 'combine': '2+3+5+7+14'},
        #         {'dataset_name': 'RD', 'combine': '1+2+3+5+7+14+15+16+17'},
        #         ]
        # task_name = 'fast_combine_w2vs+hc_mywork'
        # ensemble_list = [
        #         {'dataset_name': 'ATP', 'combine': '5+8/6+7+9/18'},
        #         {'dataset_name': 'ATP', 'combine': '3+6+16+17/4+10/5/7+14/8+9/15/18'},
        #         {'dataset_name': 'ATP', 'combine': '1+2+3+5+6+8+9+10+14+15+16+17/7/18'},
        #         {'dataset_name': 'ATP', 'combine': '1+2+3+4+6+8+9+10+14+15+16/5+17/7/18'},
        #         {'dataset_name': 'ATP', 'combine': '1+4+9+15/2+5/3+7/6+8+10+14+16+17/18'},
        #         {'dataset_name': 'MD', 'combine': '2+17/4+5/14+15/16+18'},
        #         {'dataset_name': 'MD', 'combine': '2/4/8/14/15/16/18'},
        #         {'dataset_name': 'MD', 'combine': '1+2+4+5+6+7+8+9+10+15+16/14/18'},
        #         {'dataset_name': 'MD', 'combine': '1+4+17/2+3+5+6+7+8+9+10+15+16/14/18'},
        #         {'dataset_name': 'MD', 'combine': '1+3+4+5+6+9+10/2+8/7+17/14/15/16/18'},
        #         {'dataset_name': 'RD', 'combine': '2+18/3+9+14/5+17/15+16'},
        #         {'dataset_name': 'RD', 'combine': '1/2/3/4/5/6+9/7/14/15/16/17+18'},
        #         {'dataset_name': 'RD', 'combine': '2+3+6+8+9+10+14+17+18/5/15/16'},
        #         {'dataset_name': 'RD', 'combine': '1+3+4+6+7+8+9+10+14+15+17/2/5+18/16'},
        #         {'dataset_name': 'RD', 'combine': '1/2/3+7/4+6+9/5/8+10/14/15/16/17/18'},
        #         ]
        # task_name = 'fast_combine_w2vs_ESBS_and_DESBS'
        # ensemble_list = [{'dataset_name': 'ATP', 'combine': '1+2+3+5+8+9+10/4/6/7'}, 
        #         {'dataset_name': 'MD', 'combine': '1/2/3+4+5+6+7+8+9+10'},
        #         {'dataset_name': 'RD', 'combine': '1/2/3+4+6+7+8+9+10/5'},
        #         {'dataset_name': 'ATP', 'combine': '1+2+7/5/6'}, 
        #         {'dataset_name': 'MD', 'combine': '1+3+4+5+6+7+9+10/2'},
        #         {'dataset_name': 'RD', 'combine': '1/2/3+4+6+7+9+10/5'},
        #         ]
        # task_name = 'fast_combine_w2vs+hc_ESBS_and_DESBS'
        # ensemble_list = [
        #         {'dataset_name': 'ATP', 'combine': '1+2+3+6+8+9+10+14+15+16+17/4/5/7/18'}, 
        #         {'dataset_name': 'MD', 'combine': '1+2+3+5+6+7+8+9+10+15+16+17/4/14/18'},
        #         {'dataset_name': 'RD', 'combine': '1+3+4+6+7+8+9+10+14+15+17+18/2/5/16'},
        #         {'dataset_name': 'ATP', 'combine': '1+2+3+5+6+8+9+10+14+15+16+17/7/18'}, 
        #         {'dataset_name': 'MD', 'combine': '1+2+4+5+6+7+8+9+10+15+16/14/18'},
        #         {'dataset_name': 'RD', 'combine': '2+3+6+8+9+10+14+17+18/5/16/15'},
        #         ]
        # selected_hc_train_embs, selected_hc_test_embs, _ = select_in_feature_set(train_embs[13:18], test_embs[13:18], all_emb_ways[13:18], tr_labels, tt_labels, dataset_name)
        # for i in range(13, 18):
        #     train_embs[i] = selected_hc_train_embs[i-13]
        #     test_embs[i] = selected_hc_test_embs[i-13]
        #     all_emb_ways[i] = all_emb_ways[i] + '*'
        # task_name = 'fast_combine_w2vs_selected-hc-BS_FS'
        # ensemble_list = [
        #         {'dataset_name': 'ATP', 'combine': '1+2+3+4+5+6+7+8+9+10+14+15+16+17+18'},
        #         {'dataset_name': 'MD', 'combine': '1+2+3+4+5+6+7+8+9+10+14+15+16+17+18'},
        #         {'dataset_name': 'RD', 'combine': '1+2+3+4+5+6+7+8+9+10+14+15+16+17+18'},
        #         {'dataset_name': 'ATP', 'combine': '1/2/3/4/5/6/7/8/9/10/14/15/16/17/18'}, 
        #         {'dataset_name': 'MD', 'combine': '1/2/3/4/5/6/7/8/9/10/14/15/16/17/18'},
        #         {'dataset_name': 'RD', 'combine': '1/2/3/4/5/6/7/8/9/10/14/15/16/17/18'},
        # ]
        # task_name = 'fast_combine_selected-hc-BS_FS'
        # ensemble_list = [
        #         {'dataset_name': 'ATP', 'combine': '14+15+16+17+18'},
        #         {'dataset_name': 'MD', 'combine': '14+15+16+17+18'},
        #         {'dataset_name': 'RD', 'combine': '14+15+16+17+18'},
        #         {'dataset_name': 'ATP', 'combine': '14/15/16/17/18'}, 
        #         {'dataset_name': 'MD', 'combine': '14/15/16/17/18'},
        #         {'dataset_name': 'RD', 'combine': '14/15/16/17/18'},
        # ]
        # ensemble_list = [
        #         {'dataset_name': 'ATP', 'combine': '5/6/7/18'},
        #         {'dataset_name': 'ATP', 'combine': '1/3/4/5/6/7/8/9/10/14/15/16/18'},
        #         {'dataset_name': 'ATP', 'combine': '5+6+8+14+16+18'},
        #         {'dataset_name': 'ATP', 'combine': '2+3+4+6+7+8+9+10+14+15+16+17+18'},
        #         {'dataset_name': 'MD', 'combine': '5/14/15/17'},
        #         {'dataset_name': 'MD', 'combine': '2/7/14/16/17/18'},
        #         {'dataset_name': 'MD', 'combine': '2+3+5+15+16'},
        #         {'dataset_name': 'MD', 'combine': '1+2+3+4+5+6+7+8+9+10+14+15+17+18'},
        #         {'dataset_name': 'RD', 'combine': '1/3/5/16/17'},
        #         {'dataset_name': 'RD', 'combine': '2/5/7/9/14/15/16/18'},
        #         {'dataset_name': 'RD', 'combine': '2+3+5+7+14'},
        #         {'dataset_name': 'RD', 'combine': '1+2+3+5+9+15+16'},
        # ]
        # task_name = 'fast_combine_w2vs_selected-hc-BS_CDEFS'
        # ensemble_list = [
        #         {'dataset_name': 'ATP', 'combine': '5/6/7/18'},
        #         {'dataset_name': 'ATP', 'combine': '1/3/4/5/6/7/8/9/10/14/15/16/18'},
        #         {'dataset_name': 'ATP', 'combine': '5+6+8+14+16+18'},
        #         {'dataset_name': 'ATP', 'combine': '2+3+4+6+7+8+9+10+14+15+16+17+18'},
        #         {'dataset_name': 'MD', 'combine': '5/14/15/17'},
        #         {'dataset_name': 'MD', 'combine': '2/7/14/16/17/18'},
        #         {'dataset_name': 'MD', 'combine': '2+3+5+15+16'},
        #         {'dataset_name': 'MD', 'combine': '1+2+3+4+5+6+7+8+9+10+14+15+17+18'},
        #         {'dataset_name': 'RD', 'combine': '1/3/5/16/17'},
        #         {'dataset_name': 'RD', 'combine': '2/5/7/9/14/15/16/18'},
        #         {'dataset_name': 'RD', 'combine': '2+3+5+7+14'},
        #         {'dataset_name': 'RD', 'combine': '1+2+3+5+9+15+16'},
        #         {'dataset_name': 'ATP', 'combine': '1+5+18/6+7+14+15'},
        #         {'dataset_name': 'ATP', 'combine': '1+2+3+4+5+7+8+9+10+14+15+16+17+18/6'},
        #         {'dataset_name': 'ATP', 'combine': '2+3+4+6+7+8+9+10+14+15+16+17+18'},
        #         {'dataset_name': 'ATP', 'combine': '1/2/3+7/4+14/5/6/8+17/9+10/15/16/18'},
        #         {'dataset_name': 'ATP', 'combine': '2+3+4+6+7+8+9+10+14+15+16+17+18'},
        #         {'dataset_name': 'ATP', 'combine': '1+2+3+4+5+7+8+10+14+17+18/6+9+15+16'},
        #         {'dataset_name': 'ATP', 'combine': '1+4/2+8+14+17/3+7/5+10/6+9/15+16/18'},
        #         {'dataset_name': 'MD', 'combine': '4+5+9/14/15+18/17'},
        #         {'dataset_name': 'MD', 'combine': '1+4+5+6+7+8+9+10+15/2/3/14/16/17/18'},
        #         {'dataset_name': 'MD', 'combine': '1+4+5+7+9+10+15/2/14/15/17/18'},
        #         {'dataset_name': 'MD', 'combine': '1+2/3+4+5+7+8+10/14/16/17/18'},
        #         {'dataset_name': 'MD', 'combine': '1+2+4+5+6+7+8+9+10/14+18/15+17/16'},
        #         {'dataset_name': 'MD', 'combine': '1+2+3+4+5+6+7+8+9+10+15+16+17/14+18'},
        #         {'dataset_name': 'MD', 'combine': '1+3+7+8+10/2/4+5+6+9/14/15/16/17/18'},
        #         {'dataset_name': 'RD', 'combine': '1/3+7/5/14+15/17+18'},
        #         {'dataset_name': 'RD', 'combine': '1+3+4+6+7+8+9+10+16+17+18/2/5/14/15'},
        #         {'dataset_name': 'RD', 'combine': '1+2+3+4+6+7+8+9+16+17+18/5/14/15'},
        #         {'dataset_name': 'RD', 'combine': '1+6/2/3/4+9/5/7/8/10/14/15/16/17/18'},
        #         {'dataset_name': 'RD', 'combine': '1+2+3+4+6+7+8+9+10+18/5+16+17/14/15'},
        #         {'dataset_name': 'RD', 'combine': '1+2+3+4+6+7+8+9+10+18/5+16+17/14/15'},
        #         {'dataset_name': 'RD', 'combine': '1+6+7/2/3+8/4+9+10/5/14/15/16/17/18'},
        # ]
    }
    task_name = 'fast_atp-hc'
    ensemble_list = [
            {'dataset_name': 'ATP', 'combine': '1/2/3/4/5/6/7/8/9/10/11/12/13/14/15'},
            {'dataset_name': 'ATP', 'combine': '1+2+3+4+5+6+7+8+9+10+11+12+13+14+15'},
            {'dataset_name': 'ATP', 'combine': '5/6/7'},
            {'dataset_name': 'ATP', 'combine': '4/5/6/7/8/12'},
            {'dataset_name': 'ATP', 'combine': '3+5+6'},
            {'dataset_name': 'ATP', 'combine': '1+2+3+4+5+6+7+8+9+11+12+13+14+15'},
            {'dataset_name': 'ATP', 'combine': '2+5+7+8/6+11+15'},
            {'dataset_name': 'ATP', 'combine': '1+2+3+4+8+9+10+11+13+14+15/5/6/7/12'},
            {'dataset_name': 'ATP', 'combine': '1+2+3+4+5+6+7+8+9+11+12+13+14+15'},
            {'dataset_name': 'ATP', 'combine': '1+11/2/3+8+14/4/5+10/6+9+15/7/12/13'},
            {'dataset_name': 'ATP', 'combine': '1+2+3+4+5+6+7+8+9+11+12+13+14+15'},
            {'dataset_name': 'ATP', 'combine': '1+2+3+4+5+7+8+9+10+12+13+14+15/6+11'},
            {'dataset_name': 'ATP', 'combine': '1/2/3'},
            {'dataset_name': 'MD', 'combine': '1/2/3/4/5/6/7/8/9/10/11/12/13/14/15'},
            {'dataset_name': 'MD', 'combine': '1+2+3+4+5+6+7+8+9+10+11+12+13+14+15'},
            {'dataset_name': 'MD', 'combine': '2/5/8/14/15'},
            {'dataset_name': 'MD', 'combine': '2/4/8/11/13/14/15'},
            {'dataset_name': 'MD', 'combine': '1+2+3+5'},
            {'dataset_name': 'MD', 'combine': '1+2+3+4+5+6+7+8+10+11+12+13+14+15'},
            {'dataset_name': 'MD', 'combine': '5+11/8/14/15'},
            {'dataset_name': 'MD', 'combine': '1+2+4+5+6+7+8+9+10+12+15/3/11/13/14'},
            {'dataset_name': 'MD', 'combine': '1+2+4+5+6+7+8+9+10+12+15/3/11/13/14'},
            {'dataset_name': 'MD', 'combine': '2+12/4/8/11/13/14/15'},
            {'dataset_name': 'MD', 'combine': '1+2+3+4+5+6+7+8+9+10+11+13/14+15'},
            {'dataset_name': 'MD', 'combine': '1+2+3+4+5+6+7+8+9+10+11+13/14+15'},
            {'dataset_name': 'MD', 'combine': '1+4/2+3+5+6+7+9/8+10/11/12/13/14/15'},
            {'dataset_name': 'RD', 'combine': '1/2/3/4/5/6/7/8/9/10/11/12/13/14/15'},
            {'dataset_name': 'RD', 'combine': '1+2+3+4+5+6+7+8+9+10+11+12+13+14+15'},
            {'dataset_name': 'RD', 'combine': '1/3/5/11/13/14'},
            {'dataset_name': 'RD', 'combine': '1/2/3/4/5/6/7/10/11/12/13/14/15'},
            {'dataset_name': 'RD', 'combine': '2+3+5+7+11+15'},
            {'dataset_name': 'RD', 'combine': '1+2+3+5+6+9+10+13+14+15'},
            {'dataset_name': 'RD', 'combine': '3+7+11/5/13/14'},
            {'dataset_name': 'RD', 'combine': '1+2+3+4+6+7+8+9+10+11+12+13/5/14/15'},
            {'dataset_name': 'RD', 'combine': '1+2+3+4+6+7+8+9+10+11+12+13/5/14/15'},
            {'dataset_name': 'RD', 'combine': '1/2/3+7+8/4+6/5+9/10/11/12/13/14/15'},
            {'dataset_name': 'RD', 'combine': '1+3+4+7+8+9+10+13/5+6+11/12+14/15'},
            {'dataset_name': 'RD', 'combine': '1+2+3+4+7+8+9+10+13/5+6+11/12+14/15'},
            {'dataset_name': 'RD', 'combine': '1/2/3+7+8/4+6/5+9/10/11/12/13/14/15'},
    ]
    for e_list in ensemble_list:
        if e_list['dataset_name'] != dataset_name:
            continue
        curr_train_embs, curr_test_embs, curr_emb_ways = [], [], []
        for c in e_list['combine'].split('/'):
            if '+' in c:
                temp_train_embs, temp_test_embs, temp_emb_ways = [], [], []
                for concat_c in c.split('+'):
                    temp_train_embs.append(train_embs[int(concat_c)-1])
                    temp_test_embs.append(test_embs[int(concat_c)-1])
                    temp_emb_ways.append(all_emb_ways[int(concat_c)-1])
                curr_train_embs.append(np.concatenate(temp_train_embs, axis=1))
                curr_test_embs.append(np.concatenate(temp_test_embs, axis=1))
                curr_emb_ways.append('+'.join(temp_emb_ways))
            else:
                curr_train_embs.append(train_embs[int(c)-1])
                curr_test_embs.append(test_embs[int(c)-1])
                curr_emb_ways.append(all_emb_ways[int(c)-1])
        # _, t = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/keep_or_concat/{task_name}_{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, task_name)
        # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{task_name}_{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, task_name, 0, e_list['best_est'])
        # _, t = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/train_MD_test_ATP/{task_name}_{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, task_name)
        # _, t = get_5fold_test_probs_fast(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/train_MD_test_ATP/{task_name}_{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, task_name, 0, e_list['best_est'])
        # _, t = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/w2vs+hc/fast_ensemble_{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, task_name + '_ind')
        # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc/{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, task_name)
        # _, t = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/w2vs+hc_select_itself/fast_ensemble_{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, task_name + '_ind')
        # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_select_itself/{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, task_name)
        _, t = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/w2v+hc-same-feature/fast_ensemble_{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, task_name + '_ind')
        _, t = get_5fold_test_probs_fast(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc-same-feature/{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, task_name)

def FFS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name, if_concat = True):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # only w2v
    train_embs = tr_embs[:10]
    test_embs = tt_embs[:10]
    all_emb_ways = emb_ways[:10]
    task_name = f'{dataset_name}_FFS_w2vs'
    selected_train_embs, selected_test_embs, selected_emb_ways = [], [], []
    all_max_acc = 0
    while len(all_emb_ways) > 0:
        print(selected_emb_ways)
        temp_record = []
        for add_i in range(len(all_emb_ways)):
            curr_train_embs = selected_train_embs + [train_embs[add_i]]
            curr_emb_ways = selected_emb_ways + [all_emb_ways[add_i]]
            sorted_emb_ways = sort_emb_ways(curr_emb_ways)
            if if_concat:
                concat_train_embs = np.concatenate(curr_train_embs, axis=1)
                _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/FS_atp/{dataset_name}_w2v_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_concat')
            else:
                _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/FS_atp/{dataset_name}_w2v_emb_{"_".join(sorted_emb_ways)}', True, task_name + '_ensemble')
            temp_record.append(t)
        record = pd.concat(temp_record)
        if dataset_name == 'ATP':
            metric = 'Mcc'
        else:
            metric = 'Acc'
        max_acc_id = record[f'{metric} mean'].to_numpy().argmax()
        max_acc = record[f'{metric} mean'].to_numpy().max()
        if max_acc < all_max_acc:
            print(f'End at {selected_emb_ways} !!!')
            return selected_emb_ways
        else:
            all_max_acc = max_acc
            selected_train_embs.append(train_embs.pop(max_acc_id))
            selected_emb_ways.append(all_emb_ways.pop(max_acc_id))
            # floating
            if len(selected_emb_ways[:-1]) == 0: continue
            temp_record_float = []
            for float_i in range(len(selected_emb_ways[:-1])):
                curr_train_embs = selected_train_embs[:float_i] + selected_train_embs[float_i+1:]
                curr_emb_ways = selected_emb_ways[:float_i] + selected_emb_ways[float_i+1:]
                sorted_emb_ways = sort_emb_ways(curr_emb_ways)
                if if_concat:
                    concat_train_embs = np.concatenate(curr_train_embs, axis=1)
                    _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/FS_atp/{dataset_name}_w2v_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_concat')
                else:
                    _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/FS_atp/{dataset_name}_w2v_emb_{"_".join(sorted_emb_ways)}', True, task_name + '_ensemble')
                temp_record_float.append(t)
            float_record = pd.concat(temp_record_float)
            max_acc_id_float = float_record[f'{metric} mean'].to_numpy().argmax()
            max_acc_float = float_record[f'{metric} mean'].to_numpy().max()
            if max_acc_float >= all_max_acc:
                all_max_acc = max_acc_float
                train_embs.append(selected_train_embs.pop(max_acc_id_float))
                all_emb_ways.append(selected_emb_ways.pop(max_acc_id_float))

def FBS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name, if_concat = True):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # only w2v
    train_embs = tr_embs[:10]
    test_embs = tt_embs[:10]
    all_emb_ways = emb_ways[:10]
    task_name = f'{dataset_name}_FBS_w2vs'
    selected_train_embs = train_embs
    selected_emb_ways = all_emb_ways
    removed_train_embs, removed_emb_ways = [], []
    all_max_acc = 0
    best_emb_ways = selected_emb_ways
    while len(selected_emb_ways) > 1:
        print(selected_emb_ways)
        temp_record = []
        for remove_i in range(len(selected_emb_ways)):
            curr_train_embs = selected_train_embs[:remove_i] + selected_train_embs[remove_i+1:]
            curr_emb_ways = selected_emb_ways[:remove_i] + selected_emb_ways[remove_i+1:]
            sorted_emb_ways = sort_emb_ways(curr_emb_ways)
            if if_concat:
                concat_train_embs = np.concatenate(curr_train_embs, axis=1)
                _, t = get_5fold_test_probs([concat_train_embs], '+'.join(curr_emb_ways), tr_labels, f'results/probs_txt/FS_atp/{dataset_name}_w2v_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_concat')
            else:
                _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/FS_atp/{dataset_name}_w2v_emb_{"_".join(sorted_emb_ways)}', True, task_name + '_ensemble')
            temp_record.append(t)
        record = pd.concat(temp_record)
        max_acc_id = record['Acc mean'].to_numpy().argmax()
        max_acc = record['Acc mean'].to_numpy().max()
        if max_acc < all_max_acc:
            print(f'End at {best_emb_ways} !!!')
            return best_emb_ways
        else:
            all_max_acc = max_acc
            best_emb_ways = selected_emb_ways
            removed_train_embs.append(selected_train_embs.pop(max_acc_id))
            removed_emb_ways.append(selected_emb_ways.pop(max_acc_id))
            # floating
            if len(removed_emb_ways) == 0: continue
            temp_record_float = []
            for float_i in range(len(removed_emb_ways)):
                curr_train_embs = selected_train_embs + [removed_train_embs[float_i]]
                curr_emb_ways = selected_emb_ways + [removed_emb_ways[float_i]]
                sorted_emb_ways = sort_emb_ways(curr_emb_ways)
                if if_concat:
                    concat_train_embs = np.concatenate(curr_train_embs, axis=1)
                    _, t = get_5fold_test_probs([concat_train_embs], '+'.join(curr_emb_ways), tr_labels, f'results/probs_txt/FS_atp/{dataset_name}_w2v_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_concat')
                else:
                    _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/FS_atp/{dataset_name}_w2v_emb_{"_".join(sorted_emb_ways)}', True, task_name + '_ensemble')
                temp_record_float.append(t)
            record_float = pd.concat(temp_record_float)
            max_acc_id_float = record_float['Acc mean'].to_numpy().argmax()
            max_acc_float = record_float['Acc mean'].to_numpy().max()
            if max_acc_float >= all_max_acc:
                all_max_acc = max_acc_float
                selected_train_embs.append(removed_train_embs.pop(max_acc_id_float))
                selected_emb_ways.append(removed_emb_ways.pop(max_acc_id_float))

def FS_ensemble_to_CESFS(train_data_file_name, test_data_file_name, pad_size, dataset_name, seed_bias = 0):
    '''
    Start from ensemble feature selection result:
    |    ATP    |       MD       |       RD       |
    | 5 / 6 / 7 | 1 / 2 / 3 / 5  | 1 / 2 / 3 / 5  |
    and keep going on SFS concat_ensemble
    '''
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # only w2v
    train_embs = tr_embs[:10]
    test_embs = tt_embs[:10]
    all_emb_ways = emb_ways[:10]
    task_name = f'{dataset_name}_FS_ensmeble_to_CESFS'
    FS_ensemble_results = [{'dataset_name': 'ATP', 'combine': '5/6/7', 'max_acc': 0.7872}, 
            {'dataset_name': 'MD', 'combine': '1/2/3/5', 'max_acc': 0.802},
            {'dataset_name': 'RD', 'combine': '1/2/3/5', 'max_acc': 0.8812}]
    for e_list in FS_ensemble_results:
        if e_list['dataset_name'] != dataset_name:
            continue
        left_emb_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        selected_train_embs, selected_emb_ways = [], []
        for c in e_list['combine'].split('/'):
            selected_train_embs.append(train_embs[int(c)-1])
            selected_emb_ways.append(all_emb_ways[int(c)-1])
            left_emb_index = [lew for lew in left_emb_index if lew != int(c)]
        # CESFS
        left_train_embs = [train_embs[i-1] for i in left_emb_index]
        left_emb_ways = [all_emb_ways[i-1] for i in left_emb_index]
        all_max_acc = e_list['max_acc']
        while len(left_emb_ways) > 1:
            print(selected_emb_ways)
            temp_record_ensemble, temp_record_concat = [], []
            # ensemble
            for ensemble_i in range(len(left_emb_ways)):
                curr_train_embs = selected_train_embs + [left_train_embs[ensemble_i]]
                curr_emb_ways = selected_emb_ways + [left_emb_ways[ensemble_i]]
                sorted_emb_ways = sort_emb_ways(curr_emb_ways)
                _, t = get_5fold_test_probs(curr_train_embs, ['/'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/FS_atp/{dataset_name}_w2v_emb_{"+".join(sorted_emb_ways)}', True, task_name, seed_bias)
                temp_record_ensemble.append(t)
                # concat with others
                for concat_i in range(len(selected_emb_ways)):
                    curr_train_embs = selected_train_embs[:concat_i] + selected_train_embs[concat_i+1:] + [np.concatenate([selected_train_embs[concat_i], left_train_embs[ensemble_i]], axis=1)]
                    curr_emb_ways = selected_emb_ways[:concat_i] + selected_emb_ways[concat_i+1:] + [selected_emb_ways[concat_i] + '+' + left_emb_ways[ensemble_i]]
                    sorted_emb_ways = sort_emb_ways(curr_emb_ways)
                    _, t = get_5fold_test_probs(curr_train_embs, ['/'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/FS_atp/{dataset_name}_w2v_emb_{"+".join(sorted_emb_ways)}', True, task_name, seed_bias)
                    temp_record_concat.append(t)
            record_ensemble = pd.concat(temp_record_ensemble)
            record_concat = pd.concat(temp_record_concat)
            if dataset_name == 'ATP':
                metric = 'Mcc'
            else:
                metric = 'Acc'
            max_ensemble_acc_id = record_ensemble[f'{metric} mean'].to_numpy().argmax()
            max_ensemble_acc = record_ensemble[f'{metric} mean'].to_numpy().max()
            max_concat_acc_id = record_concat[f'{metric} mean'].to_numpy().argmax()
            max_concat_acc = record_concat[f'{metric} mean'].to_numpy().max()
            action_id = np.argmax([all_max_acc, max_ensemble_acc, max_concat_acc])
            if action_id == 0:
                # stop
                print(selected_emb_ways)
                return selected_emb_ways
            elif action_id == 1:
                # ensemble
                all_max_acc = max_ensemble_acc
                selected_train_embs.append(left_train_embs.pop(max_ensemble_acc_id))
                selected_emb_ways.append(left_emb_ways.pop(max_ensemble_acc_id))
            elif action_id == 2:
                # concat with others
                all_max_acc = max_concat_acc
                add_id = max_concat_acc_id // len(selected_emb_ways)
                concat_id = max_concat_acc_id % len(selected_emb_ways)
                selected_train_embs = selected_train_embs[:concat_id] + selected_train_embs[concat_id+1:] + [np.concatenate([selected_train_embs[concat_id], left_train_embs[add_id]], axis=1)]
                selected_emb_ways = selected_emb_ways[:concat_id] + selected_emb_ways[concat_id+1:] + [selected_emb_ways[concat_id] + '+' + left_emb_ways[add_id]]
                left_train_embs.pop(add_id)
                left_emb_ways.pop(add_id)

def FS_concat_to_CDESBS(train_data_file_name, test_data_file_name, pad_size, dataset_name, seed_bias = 0):
    '''
    Start from concat feature selection result:
         |        ATP       |         MD         |      RD      |
     SFS |       3+5+6      |       1+2+3+5      | 1+2+3+5+7+10 |
     SBS | 1+2+5+6+7+8+9+10 | 1+2+4+5+6+7+8+9+10 |    2+3+5+9   |
    and keep going on SBS concat_drop_ensemble
    '''
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # only w2v
    train_embs = tr_embs[:10]
    test_embs = tt_embs[:10]
    all_emb_ways = emb_ways[:10]
    # task_name = f'{dataset_name}_SFS_concat_to_CDESBS'
    # FS_ensemble_results = [{'dataset_name': 'ATP', 'combine': '3+5+6', 'max_acc': 0.7839}, 
            # {'dataset_name': 'MD', 'combine': '1+2+3+5', 'max_acc': 0.8018},
            # {'dataset_name': 'RD', 'combine': '1+2+3+5+7+10', 'max_acc': 0.8719}]
    task_name = f'{dataset_name}_SBS_concat_to_CDESBS'
    FS_ensemble_results = [{'dataset_name': 'ATP', 'combine': '1+2+5+6+7+8+9+10', 'max_acc':0.7845}, 
            {'dataset_name': 'MD', 'combine': '1+2+4+5+6+7+8+9+10', 'max_acc': 0.7834},
            {'dataset_name': 'RD', 'combine': '2+3+5+9', 'max_acc': 0.8716}]
    for e_list in FS_ensemble_results:
        if e_list['dataset_name'] != dataset_name:
            continue
        selected_train_embs, selected_emb_ways = [], []
        for c in e_list['combine'].split('+'):
            selected_train_embs.append(train_embs[int(c)-1])
            selected_emb_ways.append(all_emb_ways[int(c)-1])
        # CDESBS
        all_max_acc = e_list['max_acc']
        sub_train_embs, sub_test_embs, sub_emb_ways = [], [], []

        while len(selected_emb_ways) > 0:
            temp_record_drop, temp_record_keep, temp_record_concat = [], [], []
            # drop
            for drop_i in range(len(selected_emb_ways)):
                if len(selected_emb_ways) == 1:
                    curr_train_embs = []
                    concat_train_embs = sub_train_embs
                    curr_emb_ways = []
                    concat_emb_ways = sub_emb_ways
                else:
                    curr_train_embs = selected_train_embs[:drop_i] + selected_train_embs[drop_i+1:]
                    concat_train_embs = [np.concatenate(curr_train_embs, axis=1)] + sub_train_embs
                    curr_emb_ways = selected_emb_ways[:drop_i] + selected_emb_ways[drop_i+1:]
                    concat_emb_ways = ['+'.join(curr_emb_ways)] + sub_emb_ways
                sorted_emb_ways = sort_emb_ways(concat_emb_ways)
                _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{dataset_name}_BS_keep_or_concat_w2vs_emb_{"_".join(sorted_emb_ways)}', True, task_name, seed_bias)
                temp_record_drop.append(t)
                # keep
                keep_concat_train_embs = concat_train_embs + [selected_train_embs[drop_i]]
                keep_concat_emb_ways = concat_emb_ways + [selected_emb_ways[drop_i]]
                keep_sorted_emb_ways = sort_emb_ways(keep_concat_emb_ways)
                _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{dataset_name}_BS_keep_or_concat_w2vs_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name, seed_bias)
                temp_record_keep.append(t)
                # concat with sub
                if len(sub_emb_ways) > 0:
                    for concat_i in range(len(sub_emb_ways)):
                        if len(curr_emb_ways) == 0:
                            concat_with_sub_concat_train_embs = concat_train_embs[:concat_i] + concat_train_embs[concat_i+1:] + [np.concatenate([concat_train_embs[concat_i], selected_train_embs[drop_i]], axis=1)]
                            concat_with_sub_concat_emb_ways = concat_emb_ways[:concat_i] + concat_emb_ways[concat_i+1:] + [concat_emb_ways[concat_i] + '+' + selected_emb_ways[drop_i]]
                        else:
                            concat_with_sub_concat_train_embs = concat_train_embs[:concat_i+1] + concat_train_embs[concat_i+2:] + [np.concatenate([concat_train_embs[concat_i+1], selected_train_embs[drop_i]], axis=1)]
                            concat_with_sub_concat_emb_ways = concat_emb_ways[:concat_i+1] + concat_emb_ways[concat_i+2:] + [concat_emb_ways[concat_i+1] + '+' + selected_emb_ways[drop_i]]
                        concat_with_sub_sorted_emb_ways = sort_emb_ways(concat_with_sub_concat_emb_ways)
                        _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{dataset_name}_BS_keep_or_concat_w2vs_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name, seed_bias)
                        temp_record_concat.append(t)
            record_drop = pd.concat(temp_record_drop)
            record_keep = pd.concat(temp_record_keep)
            if len(temp_record_concat) > 0:
                record_concat = pd.concat(temp_record_concat)
            if dataset_name == 'ATP':
                metric = 'Mcc'
            else:
                metric = 'Acc'
            max_acc_id_drop = record_drop[f'{metric} mean'].to_numpy().argmax()
            max_acc_drop = record_drop[f'{metric} mean'].to_numpy().max()
            max_acc_id_keep = record_keep[f'{metric} mean'].to_numpy().argmax()
            max_acc_keep = record_keep[f'{metric} mean'].to_numpy().max()
            if len(temp_record_concat) > 0:
                max_acc_id_concat = record_concat[f'{metric} mean'].to_numpy().argmax()
                max_acc_concat = record_concat[f'{metric} mean'].to_numpy().max()
            else:
                max_acc_id_concat = 0
                max_acc_concat = -1
            action_id = np.argmax([all_max_acc, max_acc_drop, max_acc_keep, max_acc_concat])
            if action_id == 0:
                # stop selection
                print(['+'.join(selected_emb_ways)] + sub_emb_ways)
                return ['+'.join(selected_emb_ways)] + sub_emb_ways
            elif action_id == 1:
                # drop
                all_max_acc = max_acc_drop
                selected_train_embs.pop(max_acc_id_drop)
                selected_emb_ways.pop(max_acc_id_drop)
            elif action_id == 2:
                # keep
                all_max_acc = max_acc_keep
                sub_train_embs.append(selected_train_embs.pop(max_acc_id_keep))
                sub_emb_ways.append(selected_emb_ways.pop(max_acc_id_keep))
            elif action_id == 3:
                # concat with sub
                all_max_acc = max_acc_concat
                drop_id = max_acc_id_concat // len(sub_emb_ways)
                concat_id = max_acc_id_concat % len(sub_emb_ways)
                sub_train_embs[concat_id] = np.concatenate([sub_train_embs[concat_id], selected_train_embs.pop(drop_id)], axis=1)
                sub_emb_ways[concat_id] = sub_emb_ways[concat_id] + '+' + selected_emb_ways.pop(drop_id)

def ensemble_two_models(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    for ensemble SFS(concat) & SFS(ensemble) in 1:1
    '''
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    train_embs = tr_embs[:10]
    test_embs = tt_embs[:10]
    all_emb_ways = emb_ways[:10]
    task_name = 'ensemble_two_models_SFS_concat_SFS_ensemble'
    ensemble_list = [{'dataset_name': 'ATP', 'combine1': '5/6/7', 'combine2': '3+5+6'}, 
            {'dataset_name': 'MD', 'combine1': '1/2/3/5', 'combine2': '1+2+3+5'},
            {'dataset_name': 'RD', 'combine1': '1/2/3/5', 'combine2': '1+2+3+5+7+10'}]
    task_name = 'ensemble_two_models_SBS_concat_SBS_ensemble'
    ensemble_list = [{'dataset_name': 'ATP', 'combine1': '5/6/7', 'combine2': '1+2+5+6+7+8+9+10'}, 
            {'dataset_name': 'MD', 'combine1': '1/2/3/5', 'combine2': '1+2+4+5+6+7+8+9+10'},
            {'dataset_name': 'RD', 'combine1': '1/2/3/5', 'combine2': '2+3+5+9'}]
    for e_list in ensemble_list:
        if e_list['dataset_name'] != dataset_name:
            continue
        models_probs = []
        for i in [1, 2]:
            curr_train_embs, curr_test_embs, curr_emb_ways = [], [], []
            for c in e_list[f'combine{i}'].split('/'):
                if '+' in c:
                    temp_train_embs, temp_test_embs, temp_emb_ways = [], [], []
                    for concat_c in c.split('+'):
                        temp_train_embs.append(train_embs[int(concat_c)-1])
                        temp_test_embs.append(test_embs[int(concat_c)-1])
                        temp_emb_ways.append(all_emb_ways[int(concat_c)-1])
                    curr_train_embs.append(np.concatenate(temp_train_embs, axis=1))
                    curr_test_embs.append(np.concatenate(temp_test_embs, axis=1))
                    curr_emb_ways.append('+'.join(temp_emb_ways))
                else:
                    curr_train_embs.append(train_embs[int(c)-1])
                    curr_test_embs.append(test_embs[int(c)-1])
                    curr_emb_ways.append(all_emb_ways[int(c)-1])
            t_probs = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/keep_or_concat/two_models_emb_{e_list[f"combine{i}"]}', False, task_name)
            models_probs.append(np.array(t_probs))
        models_probs = np.concatenate(models_probs, axis = 0)
        pred_probs = np.mean(models_probs, axis = 0)
        preds = [1 if i > 0.5 else 0 for i in pred_probs]
        acc, mcc, sen, spe, auc = performance(preds, pred_probs, tt_labels)
        test_performance = f'{acc},{mcc},{sen},{spe},{auc}'
        print(task_name)
        print(f'model1 : {e_list["combine1"]} , model2 : {e_list["combine2"]}')
        print(test_performance)
        return test_performance

def SFS_by_feature_importance(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    SFS by rank of feature importance(1~10).
    '''
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    train_embs = tr_embs[:10]
    test_embs = tt_embs[:10]
    all_emb_ways = emb_ways[:10]
    task_name = 'SFS_by_feature_importance'
    feature_importance_rank = {
        'ATP': [6, 5, 9, 8, 7, 4, 3, 2, 1, 0],
        'MD': [2, 3, 4, 1, 6, 7, 5, 9, 8, 0],
        'RD': [2, 4, 3, 1, 5, 7, 6, 0, 8, 9]
    }
    for data_name, fi_rank in feature_importance_rank.items():
        if data_name != dataset_name: continue
        for end in range(1, 11):
            curr_train_embs = [np.concatenate([train_embs[r] for r in fi_rank[:end]], axis=1)]
            curr_test_embs = [np.concatenate([test_embs[r] for r in fi_rank[:end]], axis=1)]
            curr_emb_ways = ['+'.join([f'{r+1}mer' for r in fi_rank[:end]])]
            sorted_emb_ways = ['+'.join(sort_emb_ways([f'{r+1}mer' for r in fi_rank[:end]]))]
            _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/keep_or_concat/{dataset_name}_BS_keep_or_concat_w2vs_emb_{"_".join(sorted_emb_ways)}', True, task_name)

def select_RF_hyper(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # train_embs = tr_embs[:10]
    # test_embs = tt_embs[:10]
    # all_emb_ways = emb_ways[:10]
    train_embs = tr_embs
    test_embs = tt_embs
    all_emb_ways = emb_ways
    diff_est = [100, 200, 300, 500, 1000, 2000]
    # task_name = 'select_RF_est'
    # ensemble_list = [
    #     {'dataset_name': 'ATP', 'combine': '5/6/7'},
    #     {'dataset_name': 'ATP', 'combine': '5/6/7'},
    #     {'dataset_name': 'ATP', 'combine': '3+5+6'},
    #     {'dataset_name': 'ATP', 'combine': '1+2+5+6+7+8+9+10'},
    #     {'dataset_name': 'ATP', 'combine': '1+5/6+7+8'},
    #     {'dataset_name': 'ATP', 'combine': '1+2+3+5+8+9+10/4/6/7'},
    #     {'dataset_name': 'ATP', 'combine': '1+2+7/5/6'},
    #     {'dataset_name': 'ATP', 'combine': '1+5/2+7/4+6'},
    #     {'dataset_name': 'ATP', 'combine': '1+2+3+4+7+8/5+6+9+10'},
    #     {'dataset_name': 'ATP', 'combine': '1+2+3+4+7+8/5+6+9+10'},
    #     {'dataset_name': 'ATP', 'combine': '1+2+3+4+5+7+9+10/6+8'},
    #     {'dataset_name': 'MD', 'combine': '1/2/3/5'},
    #     {'dataset_name': 'MD', 'combine': '1/2/3/5'},
    #     {'dataset_name': 'MD', 'combine': '1+2+3+5'},
    #     {'dataset_name': 'MD', 'combine': '1+2+4+5+6+7+8+9+10'},
    #     {'dataset_name': 'MD', 'combine': '1+3/2/5+8'},
    #     {'dataset_name': 'MD', 'combine': '1/2/3+4+5+6+7+8+9+10'},
    #     {'dataset_name': 'MD', 'combine': '1+3+4+5+6+7+9+10/2'},
    #     {'dataset_name': 'MD', 'combine': '2/4+6+7+8/5'},
    #     {'dataset_name': 'MD', 'combine': '1+3+4+5+6+7+9+10/2'},
    #     {'dataset_name': 'MD', 'combine': '1/2/3+4+5+6+7+8+9+10'},
    #     {'dataset_name': 'MD', 'combine': '1+3/2/4+6+8+10/5/7/9'},
    #     {'dataset_name': 'RD', 'combine': '1/2/3/5'},
    #     {'dataset_name': 'RD', 'combine': '1/2/3/5'},
    #     {'dataset_name': 'RD', 'combine': '1+2+3+5+7+10'},
    #     {'dataset_name': 'RD', 'combine': '2+3+5+9'},
    #     {'dataset_name': 'RD', 'combine': '1/2/3+9/5'},
    #     {'dataset_name': 'RD', 'combine': '1/2/3+4+6+7+9+10/5'},
    #     {'dataset_name': 'RD', 'combine': '1/2/3+4+6+7+9+10/5'},
    #     {'dataset_name': 'RD', 'combine': '1/2/3/5'},
    #     {'dataset_name': 'RD', 'combine': '1/2/3+4+6+7+9+10/5'},
    #     {'dataset_name': 'RD', 'combine': '1/2+5/3+4+6+7+8+9+10'},
    #     {'dataset_name': 'RD', 'combine': '1/2+5/3+4+6+7+8+9+10'},
    # ]

    # task_name = 'select_RF_est_w2vs+hc'
    # ensemble_list = [
            # {'dataset_name': 'ATP', 'combine': '1+2+3+4+5+6+7+8+9+10+14+15+16+17+18'}, 
            # {'dataset_name': 'ATP', 'combine': '1/2/3/4/5/6/7/8/9/10/14/15/16/17/18'}, 
            # {'dataset_name': 'ATP', 'combine': '5/6/7/18'}, 
            # {'dataset_name': 'ATP', 'combine': '1/2/3/4/5/6/8/9/10/15/16/18'}, 
            # {'dataset_name': 'ATP', 'combine': '1+2+5+6+16'}, 
            # {'dataset_name': 'ATP', 'combine': '1+2+3+4+5+6+7+8+9+14+15+16+17+18'}, 
            # {'dataset_name': 'MD', 'combine': '1+2+3+4+5+6+7+8+9+10+14+15+16+17+18'}, 
            # {'dataset_name': 'MD', 'combine': '1/2/3/4/5/6/7/8/9/10/14/15/16/17/18'}, 
            # {'dataset_name': 'MD', 'combine': '2/4/5/14/15/16/17/18'},
            # {'dataset_name': 'MD', 'combine': '2/4/7/14/15/16/17/18'},
            # {'dataset_name': 'MD', 'combine': '2/5/16'},
            # {'dataset_name': 'MD', 'combine': '1+2+3+4+5+6+7+9+10+14+15+16+17+18'},
            # {'dataset_name': 'RD', 'combine': '1+2+3+4+5+6+7+8+9+10+14+15+16+17+18'}, 
            # {'dataset_name': 'RD', 'combine': '1/2/3/4/5/6/7/8/9/10/14/15/16/17/18'}, 
            # {'dataset_name': 'RD', 'combine': '2/3/5/15'},
            # {'dataset_name': 'RD', 'combine': '1/2/3/5/7/9/14/15/16/17'},
            # {'dataset_name': 'RD', 'combine': '2+3+5+7+14'},
            # {'dataset_name': 'RD', 'combine': '1+2+3+5+7+14+15+16+17'},
            # {'dataset_name': 'ATP', 'combine': '5+8/6+7+9/18'},
            # {'dataset_name': 'ATP', 'combine': '1+2+3+6+8+9+10+14+15+16+17/4/5/7/18'},
            # {'dataset_name': 'ATP', 'combine': '1+2+3+5+6+8+9+10+14+15+16+17/7/18'}, 
            # {'dataset_name': 'ATP', 'combine': '3+6+16+17/4+10/5/7+14/8+9/15/18'},
            # {'dataset_name': 'ATP', 'combine': '1+2+3+5+6+8+9+10+14+15+16+17/7/18'},
            # {'dataset_name': 'ATP', 'combine': '1+2+3+4+6+8+9+10+14+15+16/5+17/7/18'},
            # {'dataset_name': 'ATP', 'combine': '1+4+9+15/2+5/3+7/6+8+10+14+16+17/18'},
            # {'dataset_name': 'MD', 'combine': '2+17/4+5/14+15/16+18'},
            # {'dataset_name': 'MD', 'combine': '1+2+3+5+6+7+8+9+10+15+16+17/4/14/18'},
            # {'dataset_name': 'MD', 'combine': '1+2+4+5+6+7+8+9+10+15+16/14/18'},
            # {'dataset_name': 'MD', 'combine': '2/4/8/14/15/16/18'},
            # {'dataset_name': 'MD', 'combine': '1+2+4+5+6+7+8+9+10+15+16/14/18'},
            # {'dataset_name': 'MD', 'combine': '1+4+17/2+3+5+6+7+8+9+10+15+16/14/18'},
            # {'dataset_name': 'MD', 'combine': '1+3+4+5+6+9+10/2+8/7+17/14/15/16/18'},
            # {'dataset_name': 'RD', 'combine': '2+18/3+9+14/5+17/15+16'},
            # {'dataset_name': 'RD', 'combine': '1+3+4+6+7+8+9+10+14+15+17+18/2/5/16'},
            # {'dataset_name': 'RD', 'combine': '2+3+6+8+9+10+14+17+18/5/16/15'},
            # {'dataset_name': 'RD', 'combine': '1/2/3/4/5/6+9/7/14/15/16/17+18'},
            # {'dataset_name': 'RD', 'combine': '2+3+6+8+9+10+14+17+18/5/15/16'},
            # {'dataset_name': 'RD', 'combine': '1+3+4+6+7+8+9+10+14+15+17/2/5+18/16'},
            # {'dataset_name': 'RD', 'combine': '1/2/3+7/4+6+9/5/8+10/14/15/16/17/18'}, 
    # ]
    val_acc_list, est_selected_list = [], []
    for e_list in ensemble_list:
        if e_list['dataset_name'] != dataset_name:
            continue
        curr_train_embs, curr_test_embs, curr_emb_ways = [], [], []
        for c in e_list['combine'].split('/'):
            if '+' in c:
                temp_train_embs, temp_test_embs, temp_emb_ways = [], [], []
                for concat_c in c.split('+'):
                    temp_train_embs.append(train_embs[int(concat_c)-1])
                    temp_test_embs.append(test_embs[int(concat_c)-1])
                    temp_emb_ways.append(all_emb_ways[int(concat_c)-1])
                curr_train_embs.append(np.concatenate(temp_train_embs, axis=1))
                curr_test_embs.append(np.concatenate(temp_test_embs, axis=1))
                curr_emb_ways.append('+'.join(temp_emb_ways))
            else:
                curr_train_embs.append(train_embs[int(c)-1])
                curr_test_embs.append(test_embs[int(c)-1])
                curr_emb_ways.append(all_emb_ways[int(c)-1])
        temp_record = []
        for est in diff_est:
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs_best_est_RF{est}/{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, dataset_name+ '_' + task_name, 0, est)
            _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/w2vs+hc_best_est_RF{est}_test/{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, dataset_name+ '_' + task_name, 0, est)
            temp_record.append(t)
        record = pd.concat(temp_record)
        if dataset_name == 'ATP':
            metric = 'Mcc'
        else:
            metric = 'Acc'
        max_acc_id = record[f'{metric} mean'].to_numpy().argmax()
        max_acc = record[f'{metric} mean'].to_numpy().max()
        # _, t = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/w2vs_best_est_RF{diff_est[max_acc_id]}/fast_ensemble_{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, dataset_name + '_' + task_name + '_ind', diff_est[max_acc_id])
        _, t = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/w2vs+hc_best_est_RF{diff_est[max_acc_id]}_test/fast_ensemble_{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, dataset_name + '_' + task_name + '_ind', diff_est[max_acc_id])
        val_acc_list.append(max_acc)
        est_selected_list.append(diff_est[max_acc_id])
    return (val_acc_list, est_selected_list)

def BS_feature_set(tr_emb, tt_emb, emb_name, tr_labels, tt_labels, dataset_name):
    selected_train_embs = tr_emb
    selected_test_embs = tt_emb
    task_name = f'{dataset_name}_{emb_name}_select_itself'
    delete_id_list = []
    if_not_selected = not os.path.exists(f'results/w2vs+hc_select_itself/{task_name}.npy')
    if not if_not_selected:
        print(f'{dataset_name} {emb_name} has selected itself : )')
        delete_id_list = np.load(f'results/w2vs+hc_select_itself/{task_name}.npy')
        for delete_i in delete_id_list:
            selected_train_embs = np.delete(selected_train_embs, delete_i, axis=1)
            selected_test_embs = np.delete(selected_test_embs, delete_i, axis=1)
        return selected_train_embs, selected_test_embs
    print(dkkdkdkdk)
    _, t = get_5fold_test_probs([selected_train_embs], [emb_name], tr_labels, f'results/probs_txt/w2vs+hand_single_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}', if_not_selected, task_name)
    if dataset_name == 'ATP':
        metric = 'Mcc'
    else:
        metric = 'Acc'
    all_max_acc = pd.concat([t])[f'{metric} mean'].to_numpy().max()
    while selected_train_embs.shape[1] > 1:
        temp_record = []
        for remove_i in range(selected_train_embs.shape[1]):
            concat_train_embs = np.concatenate([selected_train_embs[:, :remove_i], selected_train_embs[:, remove_i+1:]], axis=1)
            _, t = get_5fold_test_probs([concat_train_embs], [f'len{selected_train_embs.shape[1]} drop {remove_i}'], tr_labels, f'results/probs_txt/w2vs+hand_single_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}_drop{remove_i}', if_not_selected, task_name)
            temp_record.append(t)
        record = pd.concat(temp_record)
        max_acc_id = record[f'{metric} mean'].to_numpy().argmax()
        max_acc = record[f'{metric} mean'].to_numpy().max()
        if max_acc < all_max_acc:
            _, t = get_ind_test_probs([selected_train_embs], [selected_test_embs], [f'{emb_name} depart len{selected_train_embs.shape[1]}'], tr_labels, tt_labels, f'results/probs_txt/w2vs+hand_single_depart/fast_ensemble_{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}', if_not_selected, task_name + '_ind')
            print(f'{emb_name} : {tr_emb.shape[1]} => {selected_train_embs.shape[1]}')
            print(f'Has delete {delete_id_list}')
            np.save(f'results/w2vs+hc_select_itself/{task_name}.npy', delete_id_list)
            return selected_train_embs, selected_test_embs
        else:
            all_max_acc = max_acc
            selected_train_embs = np.delete(selected_train_embs, max_acc_id, axis=1)
            selected_test_embs = np.delete(selected_test_embs, max_acc_id, axis=1)
            delete_id_list.append(max_acc_id)

# def select_in_feature_set(tr_embs, tt_embs, emb_ways, tr_labels, tt_labels, dataset_name):
    # selected_tr_embs, selected_tt_embs = [], []
    # for i in range(len(tr_embs)):
    #     selected_tr_emb, selected_tt_emb = BS_feature_set(tr_embs[i], tt_embs[i], emb_ways[i], tr_labels, tt_labels, dataset_name)
    #     selected_tr_embs.append(selected_tr_emb)
    #     selected_tt_embs.append(selected_tt_emb)
    # return selected_tr_embs, selected_tt_embs

def select_in_feature_set(tr_embs, tt_embs, emb_ways, tr_labels, tt_labels, dataset_name):
    selected_tr_embs, selected_tt_embs, all_emb_ways = [], [], []
    for i in range(len(tr_embs)):
        # selected_tr_emb, selected_tt_emb = FS_feature_set(tr_embs[i], tt_embs[i], emb_ways[i], tr_labels, tt_labels, dataset_name)
        selected_tr_emb, selected_tt_emb = BS_feature_set(tr_embs[i], tt_embs[i], emb_ways[i], tr_labels, tt_labels, dataset_name)
        selected_tr_embs.append(selected_tr_emb)
        selected_tt_embs.append(selected_tt_emb)
        # selected_tr_emb, selected_tt_emb, emb_ways = ESBS_festure_set(tr_embs[i], tt_embs[i], emb_ways[i], tr_labels, tt_labels, dataset_name)
        # selected_tr_emb, selected_tt_emb, emb_ways = DESBS_festure_set(tr_embs[i], tt_embs[i], emb_ways[i], tr_labels, tt_labels, dataset_name)
        all_emb_ways.append(emb_ways)
        # for j in range(len(selected_tr_emb)):
        #     selected_tr_embs.append(selected_tr_emb[j])
        #     selected_tt_embs.append(selected_tt_emb[j])
        #     all_emb_ways.append(emb_ways[j])

    return selected_tr_embs, selected_tt_embs, all_emb_ways

def find_w2vs_best_model_param(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    print(tr_embs[25].shape)
    print(dkdkdkdkdk)
    train_embs = tr_embs[:10]
    test_embs = tt_embs[:10]
    train_embs = np.concatenate(train_embs, axis=1)
    test_embs = np.concatenate(test_embs, axis=1)

def w2vs_stacking(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    train_embs = tr_embs[:10]
    test_embs = tt_embs[:10]
    train_embs = np.concatenate(train_embs, axis=1)
    test_embs = np.concatenate(test_embs, axis=1)
    # DT-RFE
    estimator = DecisionTreeClassifier(random_state=42)
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),  # 5 折交叉驗證
        scoring='matthews_corrcoef' if dataset_name == 'ATP' else 'accuracy',
        n_jobs=-1
    )
    rfecv.fit(train_embs, tr_labels)
    print("最佳特徵數：", rfecv.n_features_)
    X_train_selected = rfecv.transform(X_train)
    X_test_selected = rfecv.transform(X_test)
    clf = get_model('RF', X_train_selected.shape[-1])
    clf.fit(X_train_selected, tr_labels)
    y_pred = clf.predict(X_test_selected)
    y_pred_prob = clf.predict_proba(X_test_selected)[:, 1]
    acc, mcc, sen, spe, auc = performance(preds, pred_probs, tt_labels)
    test_performance = {'Acc':acc, 'Mcc':mcc, 'Sens':sen, 'Spec':spe, 'AUC':auc}
    return test_performance

def print_data(data_file_name):
    seqs, labels = generate_features(data_file_name, pad_size, 'csv')
    n = 0
    for i in range(len(seqs)):
        if labels[i] == 0:
            n += 1
            print(f'> label{labels[i]}')
            print(seqs[i])
    for i in range(len(seqs)):
        if labels[i] == 1:
            print(f'> label{labels[i]}')
            print(seqs[i])

if __name__ == '__main__':
    torch_seed = 42
    random.seed(torch_seed)
    np.random.seed(torch_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_file_name = 'data/AntiDMP/benchmark.fasta'
    dataset_names = ['ATP', 'MD', 'RD']
    pad_sizes = [75, 61, 61]
    # dataset_names = ['ATP']
    # pad_sizes = [75]
    # dataset_names = ['MD']
    # pad_sizes = [61]
    # dataset_names = ['RD']
    # pad_sizes = [61]
    FS_list, BS_list = [], []
    for i, dataset_name in enumerate(dataset_names):
        train_data_file_name = f'data/ATPdataset/{dataset_name}_train.csv'
        test_data_file_name = f'data/ATPdataset/{dataset_name}_test.csv'
        pad_size = pad_sizes[i]
        # group_sim_ind(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        # FS_list.append(FS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # BS_list.append(BS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # FS_list.append(FS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name, True))
        # BS_list.append(BS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name, True))
        # group_sim_ind_drop(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        # group_sim_ind_watch_total(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        # group_sim_ind_watch_total_drop(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        # fast_ensemble(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        fast_ensemble2(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        # fast_ensemble2(f'data/ATPdataset/MD_train.csv', f'data/ATPdataset/ATP_test.csv', pad_size, dataset_name)
        # test_if_RF_same(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        # FS_list.append(watch_test_label_FS(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # BS_list.append(watch_test_label_BS(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # group_sim_ind_watch_total_concat_or_drop(train_data_file_name, test_data_file_name, pad_size, dataset_name)    
        # FS_list.append(FS_keep_or_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # BS_list.append(BS_concat_first_keep(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # BS_list.append(BS_keep_or_drop(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # BS_list.append(BS_keep_or_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # BS_list.append(BS_keep_or_drop_or_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # BS_list.append(BS_concat_first_keep_or_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # BS_list.append(BS_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # FS_list.append(FFS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name, True))
        # FS_list.append(FFS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name, False))
        # BS_list.append(FBS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name, True))
        # BS_list.append(FBS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name, False))
        # FS_list.append(FS_ensemble_to_CESFS(train_data_file_name, test_data_file_name, pad_size, dataset_name, 10))
        # BS_list.append(FS_concat_to_CDESBS(train_data_file_name, test_data_file_name, pad_size, dataset_name, 10))
        # BS_list.append(ensemble_two_models(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # SFS_by_feature_importance(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        # FS_list.append(select_RF_hyper(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # FS_list.append(w2vs_stacking(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # print_data(test_data_file_name)
        # find_w2vs_best_model_param(train_data_file_name, test_data_file_name, pad_size, dataset_name)
    if len(FS_list) != 0:
        print('FS')
        print(FS_list)
    if len(BS_list) != 0:
        print('BS')
        print(BS_list)
    # group_sim(data_file_name)
