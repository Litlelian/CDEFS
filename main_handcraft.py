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
        record.to_csv(f'results/new_hand/{task_name}_seed_bias{seed_bias}_ensemble_performance.csv', mode='a', header=not os.path.exists(f'results/new_hand/{task_name}_seed_bias{seed_bias}_ensemble_performance.csv'), index=False)
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
                    # if model_est == -1: clf = get_model('RF', tr_embs.shape[-1])
                    if model_est == -1: clf = get_model('XGB', tr_embs.shape[-1])
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
                print(f'{save_path}_time{time}_fold{k}_seed_bias{seed_bias}.txt')
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
    record = {'Embed way': [' '.join(all_emb_ways)]}
    for k in test_performance.keys():
        print(f'{k} : {np.mean(all_test_performance[k]):.4f} ({np.std(all_test_performance[k]):.4f})')
        record[k + ' mean'] = [np.mean(all_test_performance[k])]
        record[k + ' std'] = [np.std(all_test_performance[k])]
    record = pd.DataFrame(record)
    if if_save_pred:
        if seed_bias != 0:
            record.to_csv(f'results/hand/{task_name}_seed_bias{seed_bias}_5fold_performance.csv', mode='a', header=not os.path.exists(f'results/hand/{task_name}_seed_bias{seed_bias}_5fold_performance.csv'), index=False)
        else:
            record.to_csv(f'results/hand/{task_name}_5fold_performance.csv', mode='a', header=not os.path.exists(f'results/hand/{task_name}_5fold_performance.csv'), index=False)
    return each_embs_probs_5fold_list, record

def get_ind_test_probs(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, save_path, if_save_pred = False, task_name = '', model_est = -1):
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok = True)
    if save_path.split('/')[-1] + '_indtest.txt' not in os.listdir('/'.join(save_path.split('/')[:-1])):
        print('Training....')
        each_embs_probs = []
        for ei in range(len(all_emb_ways)):
            tr_embs = train_embs[ei]
            tt_embs = test_embs[ei]
            # if model_est == -1: clf = get_model('RF', tr_embs.shape[-1])
            if model_est == -1: clf = get_model('XGB', tr_embs.shape[-1])
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
        record.to_csv(f'results/hand/{task_name}_performance.csv', mode='a', header=not os.path.exists(f'results/hand/{task_name}_performance.csv'), index=False)
        return each_embs_probs, record

def find_best_est(train_data_file_name, test_data_file_name, pad_size, dataset_name, if_concat = False):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # only hand-craft
    train_embs = tr_embs[13:18]
    all_emb_ways = emb_ways[13:18]
    task_name = f'{dataset_name}_best_est_hand'
    each_est = [100, 200, 300, 500, 1000, 2000]
    for e in each_est:
        if if_concat:
            _, t = get_5fold_test_probs([np.concatenate(train_embs, axis = 1)], [str(e)] + ['+'.join(all_emb_ways)], tr_labels, f'results/probs_txt/hand_est{e}/{dataset_name}_hc_emb_{"+".join(all_emb_ways)}', True, task_name + '_concat', e)
        else:
            _, t = get_5fold_test_probs(train_embs, [str(e)] + all_emb_ways, tr_labels, f'results/probs_txt/hand_est{e}/{dataset_name}_hc_emb_{"_".join(all_emb_ways)}', True, task_name + '_ensemble', e)

def fast_ensemble(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    # tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # only hand-craft
    # train_embs = tr_embs[13:18]
    # test_embs = tt_embs[13:18]
    # all_emb_ways = emb_ways[13:18]
    # ['AAC', 'DPC', 'PAAC', 'CKS', 'CTD']
    # task_name = 'fast_combine_handcraft_baseline_best_est'
    # ensemble_list = [{'dataset_name': 'ATP', 'combine': '1/2/3/4/5', 'best_est': 1000}, 
    #         {'dataset_name': 'MD', 'combine': '1/2/3/4/5', 'best_est': 200},
    #         {'dataset_name': 'RD', 'combine': '1/2/3/4/5', 'best_est': 2000},
    #         {'dataset_name': 'ATP', 'combine': '1+2+3+4+5', 'best_est': 100}, 
    #         {'dataset_name': 'MD', 'combine': '1+2+3+4+5', 'best_est': 2000},
    #         {'dataset_name': 'RD', 'combine': '1+2+3+4+5', 'best_est': 2000},
    #         ]
    # task_name = 'fast_combine_handcraft_FS_best_est'
    # ensemble_list = [{'dataset_name': 'ATP', 'combine': '2', 'best_est': 1000}, 
    #         {'dataset_name': 'MD', 'combine': '1/2/4', 'best_est': 200},
    #         {'dataset_name': 'RD', 'combine': '1/2/4/5', 'best_est': 2000},
    #         {'dataset_name': 'ATP', 'combine': '2', 'best_est': 1000}, 
    #         {'dataset_name': 'MD', 'combine': '1/2/4', 'best_est': 200},
    #         {'dataset_name': 'RD', 'combine': '1/5', 'best_est': 2000},
    #         {'dataset_name': 'ATP', 'combine': '2+5', 'best_est': 100}, 
    #         {'dataset_name': 'MD', 'combine': '1+4+5', 'best_est': 2000},
    #         {'dataset_name': 'RD', 'combine': '1+2+4+5', 'best_est': 2000},
    #         {'dataset_name': 'ATP', 'combine': '2+5', 'best_est': 100}, 
    #         {'dataset_name': 'MD', 'combine': '1+4+5', 'best_est': 2000},
    #         {'dataset_name': 'RD', 'combine': '2+3+5', 'best_est': 2000},
    #         ]
    # task_name = 'fast_combine_handcraft_mywork_best_est'
    # ensemble_list = [{'dataset_name': 'ATP', 'combine': '2+5', 'best_est': 1000}, 
    #         {'dataset_name': 'MD', 'combine': '1+5/4', 'best_est': 200},
    #         {'dataset_name': 'RD', 'combine': '1+2+3+4/5', 'best_est': 2000},
    #         {'dataset_name': 'ATP', 'combine': '2', 'best_est': 1000}, 
    #         {'dataset_name': 'MD', 'combine': '1+5/4', 'best_est': 200},
    #         {'dataset_name': 'RD', 'combine': '1+2/5', 'best_est': 2000},
    #         {'dataset_name': 'ATP', 'combine': '1+5/2', 'best_est': 1000}, 
    #         {'dataset_name': 'MD', 'combine': '1/2+5/4', 'best_est': 200},
    #         {'dataset_name': 'RD', 'combine': '1+2/5', 'best_est': 2000},
    #         {'dataset_name': 'ATP', 'combine': '1/2/3+4+5', 'best_est': 1000}, 
    #         {'dataset_name': 'MD', 'combine': '1/2+3+5/4', 'best_est': 200},
    #         {'dataset_name': 'RD', 'combine': '1+2+3+4/5', 'best_est': 2000},
    #         {'dataset_name': 'ATP', 'combine': '1/2/3+4+5', 'best_est': 1000}, 
    #         {'dataset_name': 'MD', 'combine': '1/2+3+5/4', 'best_est': 200},
    #         {'dataset_name': 'RD', 'combine': '1+2+3+4/5', 'best_est': 2000},
    #         ]
    # ['AAC', 'DPC', 'PAAC', 'CKS', 'CTD']
    # task_name = 'fast_combine_handcraft_mywork'
    # ensemble_list = [
    #         {'dataset_name': 'ATP', 'combine': '2'},
    #         {'dataset_name': 'ATP', 'combine': '2'},
    #         {'dataset_name': 'ATP', 'combine': '2+5'},
    #         {'dataset_name': 'ATP', 'combine': '2+5'},
    #         {'dataset_name': 'ATP', 'combine': '2+5'},
    #         {'dataset_name': 'ATP', 'combine': '1+3+4+5/2'},
    #         {'dataset_name': 'ATP', 'combine': '1+5/2'},
    #         {'dataset_name': 'ATP', 'combine': '2+5'},
    #         {'dataset_name': 'ATP', 'combine': '1/2+5'},
    #         {'dataset_name': 'ATP', 'combine': '1+3+4+5/2'},
    #         {'dataset_name': 'ATP', 'combine': '1+3+4+5/2'},
    #         {'dataset_name': 'MD', 'combine': '1/2/4'},
    #         {'dataset_name': 'MD', 'combine': '1/2/4'},
    #         {'dataset_name': 'MD', 'combine': '1+2+4+5'},
    #         {'dataset_name': 'MD', 'combine': '1+2+4+5'},
    #         {'dataset_name': 'MD', 'combine': '1+5/3'},
    #         {'dataset_name': 'MD', 'combine': '1/2+3+5/4'},
    #         {'dataset_name': 'MD', 'combine': '1/2+5/4'},
    #         {'dataset_name': 'MD', 'combine': '1/2+5/4'},
    #         {'dataset_name': 'MD', 'combine': '1/2+5/4'},
    #         {'dataset_name': 'MD', 'combine': '1/2+3+5/4'},
    #         {'dataset_name': 'MD', 'combine': '1/2+3+5/4'},
    #         {'dataset_name': 'RD', 'combine': '1/2/4/5'},
    #         {'dataset_name': 'RD', 'combine': '2/3/5'},
    #         {'dataset_name': 'RD', 'combine': '1+2+3+4+5'},
    #         {'dataset_name': 'RD', 'combine': '1+2+5'},
    #         {'dataset_name': 'RD', 'combine': '1/2+3+4/5'},
    #         {'dataset_name': 'RD', 'combine': '1+2+3+4/5'},
    #         {'dataset_name': 'RD', 'combine': '1+2/5'},
    #         {'dataset_name': 'RD', 'combine': '2/3/5'},
    #         {'dataset_name': 'RD', 'combine': '1+2/5'},
    #         {'dataset_name': 'RD', 'combine': '1+2+3+4/5'},
    #         {'dataset_name': 'RD', 'combine': '1+4/2+3/5'},
    #         ]
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    train_embs = tr_embs
    test_embs = tt_embs
    all_emb_ways = emb_ways
    task_name = 'fast_combine_handcraft_truely_baseline'
    ensemble_list = [
            {'dataset_name': 'ATP', 'combine': '1/2/3/4/5'},
            {'dataset_name': 'ATP', 'combine': '1+2+3+4+5'},
            {'dataset_name': 'MD', 'combine': '1/2/3/4/5'},
            {'dataset_name': 'MD', 'combine': '1+2+3+4+5'},
            {'dataset_name': 'RD', 'combine': '1/2/3/4/5'},
            {'dataset_name': 'RD', 'combine': '1+2+3+4+5'},
    ]
    # train_embs, test_embs = select_in_feature_set(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, dataset_name)
    # task_name = 'fast_combine_handcraft_mywork_select_itself'
    # ensemble_list = [
    #         {'dataset_name': 'ATP', 'combine': '2'},
    #         {'dataset_name': 'ATP', 'combine': '2'},
    #         {'dataset_name': 'ATP', 'combine': '2'},
    #         {'dataset_name': 'ATP', 'combine': '2'},
    #         {'dataset_name': 'ATP', 'combine': '2'},
    #         {'dataset_name': 'ATP', 'combine': '1+3+4+5/2'},
    #         {'dataset_name': 'ATP', 'combine': '1+3+4+5/2'},
    #         {'dataset_name': 'ATP', 'combine': '2'},
    #         {'dataset_name': 'ATP', 'combine': '1+3+4+5/2'},
    #         {'dataset_name': 'ATP', 'combine': '1+3+4+5/2'},
    #         {'dataset_name': 'ATP', 'combine': '1+3+4+5/2'},
    #         {'dataset_name': 'MD', 'combine': '1/2/4'},
    #         {'dataset_name': 'MD', 'combine': '1/4/5'},
    #         {'dataset_name': 'MD', 'combine': '1'},
    #         {'dataset_name': 'MD', 'combine': '1+4+5'},
    #         {'dataset_name': 'MD', 'combine': '1/2+5/4'},
    #         {'dataset_name': 'MD', 'combine': '1/2+5/3/4'},
    #         {'dataset_name': 'MD', 'combine': '1/2+5/4'},
    #         {'dataset_name': 'MD', 'combine': '1/2+5/4'},
    #         {'dataset_name': 'MD', 'combine': '1/2+5/4'},
    #         {'dataset_name': 'MD', 'combine': '1+3/2+5/4'},
    #         {'dataset_name': 'MD', 'combine': '1+3/2+5/4'},
    #         {'dataset_name': 'RD', 'combine': '1/2/5'},
    #         {'dataset_name': 'RD', 'combine': '3/5'},
    #         {'dataset_name': 'RD', 'combine': '1+5'},
    #         {'dataset_name': 'RD', 'combine': '1+2+3+5'},
    #         {'dataset_name': 'RD', 'combine': '1+2/5'},
    #         {'dataset_name': 'RD', 'combine': '1+3+4/2/5'},
    #         {'dataset_name': 'RD', 'combine': '1+2/5'},
    #         {'dataset_name': 'RD', 'combine': '1+2/5'},
    #         {'dataset_name': 'RD', 'combine': '1+2/5'},
    #         {'dataset_name': 'RD', 'combine': '1+3+4/2/5'},
    #         {'dataset_name': 'RD', 'combine': '1+3+4/2/5'},
    # ]
    # train_embs, test_embs = select_in_feature_set(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, dataset_name)
    # task_name = 'fast_combine_handcraft_mywork_select_itself_diff_seed'
    # ensemble_list = [
    #         {'dataset_name': 'ATP', 'combine': '2'},
    #         {'dataset_name': 'ATP', 'combine': '1/2/5'},
    #         {'dataset_name': 'ATP', 'combine': '2+5'},
    #         {'dataset_name': 'ATP', 'combine': '2+5'},
    #         {'dataset_name': 'ATP', 'combine': '2+5'},
    #         {'dataset_name': 'ATP', 'combine': '1+3+4+5/2'},
    #         {'dataset_name': 'ATP', 'combine': '1+3+4+5/2'},
    #         {'dataset_name': 'ATP', 'combine': '1+5/2'},
    #         {'dataset_name': 'ATP', 'combine': '1+3+4+5/2'},
    #         {'dataset_name': 'ATP', 'combine': '1+3+4+5/2'},
    #         {'dataset_name': 'ATP', 'combine': '1/2/3+4+5'},
    #         {'dataset_name': 'MD', 'combine': '1/2/4'},
    #         {'dataset_name': 'MD', 'combine': '1/2/4'},
    #         {'dataset_name': 'MD', 'combine': '1+5'},
    #         {'dataset_name': 'MD', 'combine': '1+5'},
    #         {'dataset_name': 'MD', 'combine': '1/2/4'},
    #         {'dataset_name': 'MD', 'combine': '1+2+3+5/4'},
    #         {'dataset_name': 'MD', 'combine': '1+2+3+5/4'},
    #         {'dataset_name': 'MD', 'combine': '1/2/4'},
    #         {'dataset_name': 'MD', 'combine': '1+2+3+5/4'},
    #         {'dataset_name': 'MD', 'combine': '1+2+3+5/4'},
    #         {'dataset_name': 'MD', 'combine': '1+5/2/3/4'},
    #         {'dataset_name': 'RD', 'combine': '1/2/5'},
    #         {'dataset_name': 'RD', 'combine': '1/2/5'},
    #         {'dataset_name': 'RD', 'combine': '2+3+5'},
    #         {'dataset_name': 'RD', 'combine': '1+2+5'},
    #         {'dataset_name': 'RD', 'combine': '1+2/3/5'},
    #         {'dataset_name': 'RD', 'combine': '1+2+3+4/5'},
    #         {'dataset_name': 'RD', 'combine': '1+2/3/5'},
    #         {'dataset_name': 'RD', 'combine': '1+2/5'},
    #         {'dataset_name': 'RD', 'combine': '1+2/3/5'},
    #         {'dataset_name': 'RD', 'combine': '1+2+3+4/5'},
    #         {'dataset_name': 'RD', 'combine': '1/2/3+4/5'},
    # ]

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
        # _, t = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/hand/fast_ensemble_{e_list["dataset_name"]}_hc_emb_{"_".join(curr_emb_ways)}', True, dataset_name + '_' + task_name + '_ind')
        # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand/{e_list["dataset_name"]}_hc_emb_{"_".join(curr_emb_ways)}', True, dataset_name + '_' + task_name)
        # _, t = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/hand_select_in_feature_set/fast_ensemble_{e_list["dataset_name"]}_hc_emb_{"_".join(curr_emb_ways)}', True, dataset_name + '_' + task_name + '_ind')
        # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{e_list["dataset_name"]}_hc_emb_{"_".join(curr_emb_ways)}', True, dataset_name + '_' + task_name)
        # _, t = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/hand_select_in_feature_set/fast_ensemble_{e_list["dataset_name"]}_hc_emb_{"_".join(curr_emb_ways)}_diff_seed', True, dataset_name + '_' + task_name + '_ind')
        # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{e_list["dataset_name"]}_hc_emb_{"_".join(curr_emb_ways)}', True, dataset_name + '_' + task_name, 10)
        # _, t = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/hand_XGB/fast_ensemble_{e_list["dataset_name"]}_hc_emb_{"_".join(curr_emb_ways)}', True, dataset_name + '_' + task_name + '_ind')
        # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand_XGB/{e_list["dataset_name"]}_hc_emb_{"_".join(curr_emb_ways)}', True, dataset_name + '_' + task_name)
        _, t = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/ATP_hc/fast_ensemble_{e_list["dataset_name"]}_hc_emb_{"_".join(curr_emb_ways)}', True, dataset_name + '_' + task_name + '_ind')
        _, t = get_5fold_test_probs_fast(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/ATP_hc/{e_list["dataset_name"]}_hc_emb_{"_".join(curr_emb_ways)}', True, dataset_name + '_' + task_name)

def FS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name, if_concat = False):
    # tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # # only hand-craft
    # train_embs = tr_embs[13:18]
    # test_embs = tt_embs[13:18]
    # all_emb_ways = emb_ways[13:18]
    # train_embs, test_embs = select_in_feature_set(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, dataset_name)
    # task_name = f'{dataset_name}_FS_hc'
    # task_name = f'{dataset_name}_best_est_hand_FS_hand'
    # task_name = f'{dataset_name}_FS_hc_select_in_feature_set'
    # task_name = f'{dataset_name}_FS_hc_select_in_feature_set_diff_seed'
    # task_name = f'{dataset_name}_FS_hc_select_in_feature_set_FS'
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    train_embs = tr_embs
    test_embs = tt_embs
    all_emb_ways = emb_ways
    task_name = f'{dataset_name}_FS_hc-same-feature'
    selected_train_embs, selected_test_embs, selected_emb_ways = [], [], []
    all_max_acc = 0
    # best est
    if if_concat: each_best_est = {'ATP': 100, 'MD': 2000, 'RD': 2000}
    else: each_best_est = {'ATP': 1000, 'MD': 200, 'RD': 2000}
    while len(all_emb_ways) > 0:
        print(selected_emb_ways)
        temp_record = []
        for add_i in range(len(all_emb_ways)):
            curr_train_embs = selected_train_embs + [train_embs[add_i]]
            concat_train_embs = np.concatenate(curr_train_embs, axis=1)
            curr_emb_ways = selected_emb_ways + [all_emb_ways[add_i]]
            sorted_emb_ways = sort_emb_ways(curr_emb_ways)
            if if_concat:
                # _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_concat')
                # _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/best_est_hand/{dataset_name}_hc_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_concat', each_best_est[dataset_name])
                # _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_concat')
                _, t = get_5fold_test_probs_fast([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_concat')
                # _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_concat', 10)
            else:
                # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name + '_ensemble')
                # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/best_est_hand/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name + '_ensemble', each_best_est[dataset_name])
                # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name + '_ensemble')
                _, t = get_5fold_test_probs_fast(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name + '_ensemble')
                # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name + '_ensemble', 10)
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
            selected_test_embs.append(test_embs.pop(max_acc_id))
            selected_emb_ways.append(all_emb_ways.pop(max_acc_id))

def BS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name, if_concat = False):
    # tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # # only hand-craft
    # train_embs = tr_embs[13:18]
    # test_embs = tt_embs[13:18]
    # all_emb_ways = emb_ways[13:18]
    # train_embs, test_embs = select_in_feature_set(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, dataset_name)
    # task_name = f'{dataset_name}_BS_hc'
    # task_name = f'{dataset_name}_best_est_hand_BS_w2vs'
    # task_name = f'{dataset_name}_BS_hc_select_in_feature_set'
    # task_name = f'{dataset_name}_BS_hc_select_in_feature_set_diff_seed'
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    train_embs = tr_embs
    test_embs = tt_embs
    all_emb_ways = emb_ways
    task_name = f'{dataset_name}_BS_hc-same-feature'
    selected_train_embs = train_embs
    selected_emb_ways = all_emb_ways
    all_max_acc = 0
    best_emb_ways = selected_emb_ways
    # best est
    if if_concat: each_best_est = {'ATP': 100, 'MD': 2000, 'RD': 2000}
    else: each_best_est = {'ATP': 1000, 'MD': 200, 'RD': 2000}
    while len(selected_emb_ways) > 1:
        print(selected_emb_ways)
        temp_record = []
        for remove_i in range(len(selected_emb_ways)):
            curr_train_embs = selected_train_embs[:remove_i] + selected_train_embs[remove_i+1:]
            curr_emb_ways = selected_emb_ways[:remove_i] + selected_emb_ways[remove_i+1:]
            sorted_emb_ways = sort_emb_ways(curr_emb_ways)
            concat_train_embs = np.concatenate(curr_train_embs, axis=1)
            if if_concat:
                # _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_concat')
                # _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/best_est_hand/{dataset_name}_hc_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_concat', each_best_est[dataset_name])
                # _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_concat')            
                _, t = get_5fold_test_probs_fast([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_concat')            
                # _, t = get_5fold_test_probs([concat_train_embs], ['+'.join(curr_emb_ways)], tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_concat', 10)            
            else:
                # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name + '_ensemble')
                # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/best_est_hand/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name + '_ensemble', each_best_est[dataset_name])
                # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_ensemble')
                _, t = get_5fold_test_probs_fast(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_ensemble')
                # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"+".join(sorted_emb_ways)}', True, task_name + '_ensemble', 10)
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
            selected_train_embs.pop(max_acc_id)
            selected_emb_ways.pop(max_acc_id)

def FS_keep_or_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    Like SFS method. However, I will try concat with selected embs instead of just adding it.
    '''
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # only hand-craft
    # train_embs = tr_embs[13:18]
    # test_embs = tt_embs[13:18]
    # all_emb_ways = emb_ways[13:18]
    # train_embs, test_embs = select_in_feature_set(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, dataset_name)
    # task_name = f'{dataset_name}_RF100_FS_keep_or_concat_hc'
    # task_name = f'{dataset_name}_FS_keep_or_concat_hc'
    # task_name = f'{dataset_name}_FS_keep_or_concat_hc_select_in_feature_set'
    # task_name = f'{dataset_name}_FS_keep_or_concat_hc_select_in_feature_set_diff_seed'
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    train_embs = tr_embs
    test_embs = tt_embs
    all_emb_ways = emb_ways
    task_name = f'{dataset_name}_CESFS_hc-same-feature'
    selected_train_embs, selected_test_embs, selected_emb_ways = [], [], []
    all_max_acc = 0
    each_best_est = {'ATP': 1000, 'MD': 200, 'RD': 2000}
    while len(all_emb_ways) > 0:
        print(selected_emb_ways)
        temp_record = []
        for add_i in range(len(all_emb_ways)):
            curr_train_embs = selected_train_embs + [train_embs[add_i]]
            curr_emb_ways = selected_emb_ways + [all_emb_ways[add_i]]
            sorted_emb_ways = sort_emb_ways(curr_emb_ways)
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/best_est_hand/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name, each_best_est[dataset_name])
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            _, t = get_5fold_test_probs_fast(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name, 10)
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
                return selected_emb_ways
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
                    # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
                    # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/best_est_hand/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name, each_best_est[dataset_name])
                    # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
                    _, t = get_5fold_test_probs_fast(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
                    # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name, 10)
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
                    return selected_emb_ways
                else:
                    all_max_acc = max_acc
                    selected_train_embs.append(train_embs.pop(max_acc_id))
                    selected_test_embs.append(test_embs.pop(max_acc_id))
                    selected_emb_ways.append(all_emb_ways.pop(max_acc_id))
            else:
                # concat
                if max_concat_acc < all_max_acc:
                    print(f'End at {selected_emb_ways} !!!')
                    return selected_emb_ways
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
    # tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # # only hand-craft
    # train_embs = tr_embs[13:18]
    # test_embs = tt_embs[13:18]
    # all_emb_ways = emb_ways[13:18]
    # train_embs, test_embs = select_in_feature_set(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, dataset_name)
    # task_name = f'{dataset_name}_RF100_BS_keep_or_concat_hc'
    # task_name = f'{dataset_name}_BS_keep_or_concat_hc'
    # task_name = f'{dataset_name}_BS_keep_or_concat_hc_select_in_feature_set'
    # task_name = f'{dataset_name}_BS_keep_or_concat_hc_select_in_feature_set_diff_seed'
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    train_embs = tr_embs
    test_embs = tt_embs
    all_emb_ways = emb_ways
    task_name = f'{dataset_name}_CDSBS_hc-same-feature'
    all_max_acc = 0
    each_best_est = {'ATP': 1000, 'MD': 200, 'RD': 2000}
    while len(all_emb_ways) > 1:
        temp_record, temp_record2 = [], []
        for remove_i in range(len(all_emb_ways)):
            curr_train_embs = train_embs[:remove_i] + train_embs[remove_i+1:]
            curr_emb_ways = all_emb_ways[:remove_i] + all_emb_ways[remove_i+1:]
            sorted_emb_ways = sort_emb_ways(curr_emb_ways)
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/best_est_hand/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name, each_best_est[dataset_name])
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            _, t = get_5fold_test_probs_fast(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name, 10)
            temp_record.append(t)
            # try to concat instead of dropping
            for concat_id in range(len(curr_emb_ways)):
                concat_train_embs = curr_train_embs[:concat_id] + curr_train_embs[concat_id+1:] + [np.concatenate([curr_train_embs[concat_id], train_embs[remove_i]], axis=1)]
                concat_emb_ways = curr_emb_ways[:concat_id] + curr_emb_ways[concat_id+1:] + [curr_emb_ways[concat_id] + '+' + all_emb_ways[remove_i]]
                sorted_emb_ways = sort_emb_ways(concat_emb_ways)
                # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
                # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/best_est_hand/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name, each_best_est[dataset_name])
                # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
                _, t = get_5fold_test_probs_fast(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
                # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name, 10)
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
    # tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # # only hand-craft
    # train_embs = tr_embs[13:18]
    # test_embs = tt_embs[13:18]
    # all_emb_ways = emb_ways[13:18]
    # train_embs, test_embs = select_in_feature_set(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, dataset_name)
    # task_name = f'{dataset_name}_RF100_BS_keep_or_drop_or_concat_hc'
    # task_name = f'{dataset_name}_BS_keep_or_drop_or_concat_hc'
    # task_name = f'{dataset_name}_BS_keep_or_drop_or_concat_hc_select_in_feature_set'
    # task_name = f'{dataset_name}_BS_keep_or_drop_or_concat_hc_select_in_feature_set_diff_seed'
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    train_embs = tr_embs
    test_embs = tt_embs
    all_emb_ways = emb_ways
    task_name = f'{dataset_name}_CDESBS_hc-same-feature'
    each_best_est = {'ATP': 1000, 'MD': 200, 'RD': 2000}
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
            # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/best_est_hand/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name, each_best_est[dataset_name])
            # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            _, t = get_5fold_test_probs_fast(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name, 10)
            temp_record_drop.append(t)
            # keep
            keep_concat_train_embs = concat_train_embs + [train_embs[drop_i]]
            keep_concat_emb_ways = concat_emb_ways + [all_emb_ways[drop_i]]
            keep_sorted_emb_ways = sort_emb_ways(keep_concat_emb_ways)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/best_est_hand/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name, each_best_est[dataset_name])
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            _, t = get_5fold_test_probs_fast(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name, 10)
            temp_record_keep.append(t)
            # concat with sub
            if len(sub_emb_ways) > 0:
                for concat_i in range(len(sub_emb_ways)):
                    concat_with_sub_concat_train_embs = concat_train_embs[:concat_i+1] + concat_train_embs[concat_i+2:] + [np.concatenate([concat_train_embs[concat_i+1], train_embs[drop_i]], axis=1)]
                    concat_with_sub_concat_emb_ways = concat_emb_ways[:concat_i+1] + concat_emb_ways[concat_i+2:] + [concat_emb_ways[concat_i+1] + '+' + all_emb_ways[drop_i]]
                    concat_with_sub_sorted_emb_ways = sort_emb_ways(concat_with_sub_concat_emb_ways)
                    # _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                    # _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/best_est_hand/{dataset_name}_hc_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name, each_best_est[dataset_name])
                    # _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                    _, t = get_5fold_test_probs_fast(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                    # _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name, 10)
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

def BS_concat_first_keep_or_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    Like SBS_keep_or_drop_or_concat, but do not drop.
    '''
    # tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # # only hand-craft
    # train_embs = tr_embs[13:18]
    # test_embs = tt_embs[13:18]
    # all_emb_ways = emb_ways[13:18]
    # train_embs, test_embs = select_in_feature_set(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, dataset_name)
    # task_name = f'{dataset_name}_RF100_BS_concat_first_keep_or_concat_hc'
    # task_name = f'{dataset_name}_BS_concat_first_keep_or_concat_hc'
    # task_name = f'{dataset_name}_BS_concat_first_keep_or_concat_hc_select_in_feature_set'
    # task_name = f'{dataset_name}_BS_concat_first_keep_or_concat_hc_select_in_feature_set_diff_seed'
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    train_embs = tr_embs
    test_embs = tt_embs
    all_emb_ways = emb_ways
    task_name = f'{dataset_name}_CESBS_hc-same-feature'
    each_best_est = {'ATP': 1000, 'MD': 200, 'RD': 2000}
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
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/best_est_hand/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name, each_best_est[dataset_name])
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            _, t = get_5fold_test_probs_fast(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name, 10)
            temp_record_keep.append(t)
            # concat with sub
            if len(sub_emb_ways) > 0:
                for concat_i in range(len(sub_emb_ways)):
                    concat_with_sub_concat_train_embs = concat_train_embs[:concat_i+1] + concat_train_embs[concat_i+2:] + [np.concatenate([concat_train_embs[concat_i+1], train_embs[drop_i]], axis=1)]
                    concat_with_sub_concat_emb_ways = concat_emb_ways[:concat_i+1] + concat_emb_ways[concat_i+2:] + [concat_emb_ways[concat_i+1] + '+' + all_emb_ways[drop_i]]
                    concat_with_sub_sorted_emb_ways = sort_emb_ways(concat_with_sub_concat_emb_ways)
                    # _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                    # _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/best_est_hand/{dataset_name}_hc_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name, each_best_est[dataset_name])
                    # _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                    _, t = get_5fold_test_probs_fast(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                    # _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, concat_with_sub_concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name, 10)
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

def BS_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    Like SBS_keep_or_drop_or_concat, but do not drop.
    '''
    # tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # # only hand-craft
    # train_embs = tr_embs[13:18]
    # test_embs = tt_embs[13:18]
    # all_emb_ways = emb_ways[13:18]
    # train_embs, test_embs = select_in_feature_set(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, dataset_name)
    # task_name = f'{dataset_name}_RF100_BS_concat_hc'
    # task_name = f'{dataset_name}_BS_concat_hc'
    # task_name = f'{dataset_name}_BS_concat_hc_select_in_feature_set'
    # task_name = f'{dataset_name}_BS_concat_hc_select_in_feature_set_diff_seed'
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    train_embs = tr_embs
    test_embs = tt_embs
    all_emb_ways = emb_ways
    task_name = f'{dataset_name}_CSBS_hc-same-feature'
    each_best_est = {'ATP': 1000, 'MD': 200, 'RD': 2000}
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
                # _, t = get_5fold_test_probs(concat_with_sub_curr_train_embs, concat_with_sub_curr_emb_ways, tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                # _, t = get_5fold_test_probs(concat_with_sub_curr_train_embs, concat_with_sub_curr_emb_ways, tr_labels, f'results/probs_txt/best_est_hand/{dataset_name}_hc_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name, each_best_est[dataset_name])
                # _, t = get_5fold_test_probs(concat_with_sub_curr_train_embs, concat_with_sub_curr_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                _, t = get_5fold_test_probs_fast(concat_with_sub_curr_train_embs, concat_with_sub_curr_emb_ways, tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name)
                # _, t = get_5fold_test_probs(concat_with_sub_curr_train_embs, concat_with_sub_curr_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(concat_with_sub_sorted_emb_ways)}', True, task_name, 10)
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

def BS_concat_first_keep(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    Like SBS_keep_or_drop_or_concat, but do not drop and concat.
    '''
    # tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # # only hc
    # train_embs = tr_embs[13:18]
    # test_embs = tt_embs[13:18]
    # all_emb_ways = emb_ways[13:18]
    # train_embs, test_embs = select_in_feature_set(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, dataset_name)
    # task_name = f'{dataset_name}_RF100_BS_concat_first_keep_hc'
    # task_name = f'{dataset_name}_BS_concat_first_keep_hc_select_in_feature_set'
    # task_name = f'{dataset_name}_BS_concat_first_keep_hc_select_in_feature_set_diff_seed'
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    train_embs = tr_embs
    test_embs = tt_embs
    all_emb_ways = emb_ways
    task_name = f'{dataset_name}_ESBS_hc-same-feature'
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
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            _, t = get_5fold_test_probs_fast(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name, 10)
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

def BS_keep_or_drop(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    '''
    Like BS_keep_or_drop_or_concat, but no concat
    '''
    # tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # # only hc
    # train_embs = tr_embs[13:18]
    # test_embs = tt_embs[13:18]
    # all_emb_ways = emb_ways[13:18]
    # train_embs, test_embs = select_in_feature_set(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, dataset_name)
    # task_name = f'{dataset_name}_RF100_BS_keep_or_drop_hc'
    # task_name = f'{dataset_name}_BS_keep_or_drop_hc_select_in_feature_set'
    # task_name = f'{dataset_name}_BS_keep_or_drop_hc_select_in_feature_set_diff_seed'
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    train_embs = tr_embs
    test_embs = tt_embs
    all_emb_ways = emb_ways
    task_name = f'{dataset_name}_DESBS_hc-same-feature'
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
            # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            _, t = get_5fold_test_probs_fast(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(concat_train_embs, concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(sorted_emb_ways)}', True, task_name, 10)
            temp_record_drop.append(t)
            # keep
            keep_concat_train_embs = concat_train_embs + [train_embs[drop_i]]
            keep_concat_emb_ways = concat_emb_ways + [all_emb_ways[drop_i]]
            keep_sorted_emb_ways = sort_emb_ways(keep_concat_emb_ways)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hand/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            _, t = get_5fold_test_probs_fast(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hc-atp/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name)
            # _, t = get_5fold_test_probs(keep_concat_train_embs, keep_concat_emb_ways, tr_labels, f'results/probs_txt/hand_select_in_feature_set/{dataset_name}_hc_emb_{"_".join(keep_sorted_emb_ways)}', True, task_name, 10)
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

def select_RF_hyper(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    train_embs = tr_embs[13:18]
    test_embs = tt_embs[13:18]
    all_emb_ways = emb_ways[13:18]
    diff_est = [100, 200, 300, 500, 1000, 2000]

    task_name = 'select_RF_est_hc'
    ensemble_list = [
            {'dataset_name': 'ATP', 'combine': '2'},
            {'dataset_name': 'ATP', 'combine': '2'},
            {'dataset_name': 'ATP', 'combine': '2+5'},
            {'dataset_name': 'ATP', 'combine': '2+5'},
            {'dataset_name': 'ATP', 'combine': '2+5'},
            {'dataset_name': 'ATP', 'combine': '1+3+4+5/2'},
            {'dataset_name': 'ATP', 'combine': '1+5/2'},
            {'dataset_name': 'ATP', 'combine': '2+5'},
            {'dataset_name': 'ATP', 'combine': '1/2+5'},
            {'dataset_name': 'ATP', 'combine': '1+3+4+5/2'},
            {'dataset_name': 'ATP', 'combine': '1+3+4+5/2'},
            {'dataset_name': 'MD', 'combine': '1/2/4'},
            {'dataset_name': 'MD', 'combine': '1/2/4'},
            {'dataset_name': 'MD', 'combine': '1+2+4+5'},
            {'dataset_name': 'MD', 'combine': '1+2+4+5'},
            {'dataset_name': 'MD', 'combine': '1+5/3'},
            {'dataset_name': 'MD', 'combine': '1/2+3+5/4'},
            {'dataset_name': 'MD', 'combine': '1/2+5/4'},
            {'dataset_name': 'MD', 'combine': '1/2+5/4'},
            {'dataset_name': 'MD', 'combine': '1/2+5/4'},
            {'dataset_name': 'MD', 'combine': '1/2+3+5/4'},
            {'dataset_name': 'MD', 'combine': '1/2+3+5/4'},
            {'dataset_name': 'RD', 'combine': '1/2/4/5'},
            {'dataset_name': 'RD', 'combine': '2/3/5'},
            {'dataset_name': 'RD', 'combine': '1+2+3+4+5'},
            {'dataset_name': 'RD', 'combine': '1+2+5'},
            {'dataset_name': 'RD', 'combine': '1/2+3+4/5'},
            {'dataset_name': 'RD', 'combine': '1+2+3+4/5'},
            {'dataset_name': 'RD', 'combine': '1+2/5'},
            {'dataset_name': 'RD', 'combine': '2/3/5'},
            {'dataset_name': 'RD', 'combine': '1+2/5'},
            {'dataset_name': 'RD', 'combine': '1+2+3+4/5'},
            {'dataset_name': 'RD', 'combine': '1+4/2+3/5'},
            ]

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
            _, t = get_5fold_test_probs(curr_train_embs, curr_emb_ways, tr_labels, f'results/probs_txt/hc_best_est_RF{est}/{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, dataset_name+ '_' + task_name, est)
            temp_record.append(t)
        record = pd.concat(temp_record)
        if dataset_name == 'ATP':
            metric = 'Mcc'
        else:
            metric = 'Acc'
        max_acc_id = record[f'{metric} mean'].to_numpy().argmax()
        max_acc = record[f'{metric} mean'].to_numpy().max()
        _, t = get_ind_test_probs(curr_train_embs, curr_test_embs, curr_emb_ways, tr_labels, tt_labels, f'results/probs_txt/hc_best_est_RF{diff_est[max_acc_id]}/fast_ensemble_{e_list["dataset_name"]}_emb_{"_".join(curr_emb_ways)}', True, dataset_name + '_' + task_name + '_ind', diff_est[max_acc_id])
        val_acc_list.append(max_acc)
        est_selected_list.append(diff_est[max_acc_id])
    return (val_acc_list, est_selected_list)

# 做不動
def BS_hc_depart(train_data_file_name, test_data_file_name, pad_size, dataset_name, if_concat = False):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    train_embs = tr_embs[13:18]
    test_embs = tt_embs[13:18]
    all_emb_ways = emb_ways[13:18]
    task_name = 'BS_hc_depart'
    selected_train_embs = np.concatenate(train_embs, axis=1)
    selected_test_embs = np.concatenate(test_embs, axis=1)
    if not if_concat:
        selected_train_embs = selected_train_embs.T.reshape(selected_train_embs.shape[1], selected_train_embs.shape[0], 1)
        selected_test_embs = selected_test_embs.T.reshape(selected_test_embs.shape[1], selected_test_embs.shape[0], 1)
        _, t = get_5fold_test_probs(selected_train_embs, ['+'.join(all_emb_ways)], tr_labels, f'results/probs_txt/hand_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}_ensemble', True, dataset_name + '_' + task_name + '_ensemble')
    else:
        _, t = get_5fold_test_probs([selected_train_embs], ['+'.join(all_emb_ways)], tr_labels, f'results/probs_txt/hand_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}', True, dataset_name + '_' + task_name)
    if dataset_name == 'ATP':
        metric = 'Mcc'
    else:
        metric = 'Acc'
    all_max_acc = pd.concat([t])[f'{metric} mean'].to_numpy().max()
    while selected_train_embs.shape[1] > 1:
        temp_record = []
        for remove_i in range(selected_train_embs.shape[1]):
            if if_concat:
                concat_train_embs = np.concatenate([selected_train_embs[:, :remove_i], selected_train_embs[:, remove_i+1:]], axis=1)
                _, t = get_5fold_test_probs([concat_train_embs], [f'len{selected_train_embs.shape[1]} drop {remove_i}'], tr_labels, f'results/probs_txt/hand_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}_drop{remove_i}', True, dataset_name + '_' + task_name)
            else:
                curr_train_embs = selected_train_embs[:remove_i] + selected_train_embs[remove_i+1:]
                print(curr_train_embs.shape)
                print(dkdkdkkdkd)
                _, t = get_5fold_test_probs(curr_train_embs, [f'len{selected_train_embs.shape[1]} drop {remove_i}'], tr_labels, f'results/probs_txt/hand_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}_drop{remove_i}_ensemble', True, dataset_name + '_' + task_name + '_ensemble')
            temp_record.append(t)
        record = pd.concat(temp_record)
        max_acc_id = record[f'{metric} mean'].to_numpy().argmax()
        max_acc = record[f'{metric} mean'].to_numpy().max()
        if max_acc < all_max_acc:
            if if_concat:
                _, t = get_ind_test_probs([selected_train_embs], [selected_test_embs], [f'hc depart len{selected_train_embs.shape[1]}'], tr_labels, tt_labels, f'results/probs_txt/hand_depart/fast_ensemble_{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}', True, dataset_name + '_' + task_name + '_ind')
            else:
                _, t = get_ind_test_probs(selected_train_embs, selected_test_embs, [f'hc depart len{selected_train_embs.shape[1]}'], tr_labels, tt_labels, f'results/probs_txt/hand_depart/fast_ensemble_{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}', True, dataset_name + '_' + task_name + '_ensemble_ind')
            print(f'End !!! val {metric} = {all_max_acc}')
            break
        else:
            all_max_acc = max_acc
            selected_train_embs = np.delete(selected_train_embs, max_acc_id, axis=1)
            selected_test_embs = np.delete(selected_test_embs, max_acc_id, axis=1)

def FS_feature_set(tr_emb, tt_emb, emb_name, tr_labels, tt_labels, dataset_name):
    selected_train_embs, selected_test_embs = [], []
    train_embs = tr_emb
    test_embs = tt_emb
    task_name = f'{dataset_name}_{emb_name}_select_itself_FS'
    add_id_list = []
    if_not_selected = not os.path.exists(f'results/hand/{task_name}.npy')
    if not if_not_selected:
        print(f'{dataset_name} {emb_name} has selected itself : )')
        add_id_list = np.load(f'results/hand/{task_name}.npy')
        for add_i in add_id_list:
            selected_train_embs.append(train_embs[:, add_i])
            selected_test_embs.append(test_embs[:, add_i])
            train_embs = np.delete(train_embs, add_i, axis=1)
            test_embs = np.delete(test_embs, add_i, axis=1)
        selected_train_embs = np.stack(selected_train_embs, axis=1).T
        selected_test_embs = np.stack(selected_test_embs, axis=1).T
        return selected_train_embs, selected_test_embs
    all_max_acc = 0
    if dataset_name == 'ATP':
        metric = 'Mcc'
    else:
        metric = 'Acc'
    while train_embs.shape[1] > 0:
        temp_record = []
        for add_i in range(train_embs.shape[1]):
            curr_train_embs = selected_train_embs + [train_embs[:, add_i]]
            concat_train_embs = np.stack(curr_train_embs, axis=0).T
            _, t = get_5fold_test_probs([concat_train_embs], [f'len{concat_train_embs.shape[1]} add {add_i}'], tr_labels, f'results/probs_txt/hand_single_depart/{dataset_name}_{task_name}_len{concat_train_embs.shape[1]}_add{add_i}', if_not_selected, task_name)
            temp_record.append(t)
        record = pd.concat(temp_record)
        max_acc_id = record[f'{metric} mean'].to_numpy().argmax()
        max_acc = record[f'{metric} mean'].to_numpy().max()
        if max_acc < all_max_acc:
            concat_train_embs = np.stack(selected_train_embs, axis=0).T
            concat_test_embs = np.stack(selected_test_embs, axis=0).T
            _, t = get_ind_test_probs([concat_train_embs], [concat_test_embs], [f'{emb_name} depart len{concat_train_embs.shape[1]}'], tr_labels, tt_labels, f'results/probs_txt/hand_single_depart/fast_ensemble_{dataset_name}_{task_name}_len{concat_train_embs.shape[1]}', if_not_selected, task_name + '_ind')
            print(f'{emb_name} : {tr_emb.shape[1]} => {concat_train_embs.shape[1]}')
            print(f'Has add {add_id_list}')
            np.save(f'results/hand/{task_name}.npy', add_id_list)
            return concat_train_embs, concat_test_embs
        else:
            all_max_acc = max_acc
            selected_train_embs.append(train_embs[:, max_acc_id])
            selected_test_embs.append(test_embs[:, max_acc_id])
            train_embs = np.delete(train_embs, max_acc_id, axis=1)
            test_embs = np.delete(test_embs, max_acc_id, axis=1)
            add_id_list.append(max_acc_id)

def BS_feature_set(tr_emb, tt_emb, emb_name, tr_labels, tt_labels, dataset_name):
    selected_train_embs = tr_emb
    selected_test_embs = tt_emb
    task_name = f'{dataset_name}_{emb_name}_select_itself'
    delete_id_list = []
    if_not_selected = not os.path.exists(f'results/hand/{task_name}.npy')
    if not if_not_selected:
        print(f'{dataset_name} {emb_name} has selected itself : )')
        delete_id_list = np.load(f'results/hand/{task_name}.npy')
        for delete_i in delete_id_list:
            selected_train_embs = np.delete(selected_train_embs, delete_i, axis=1)
            selected_test_embs = np.delete(selected_test_embs, delete_i, axis=1)
        return selected_train_embs, selected_test_embs
    _, t = get_5fold_test_probs([selected_train_embs], [emb_name], tr_labels, f'results/probs_txt/hand_single_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}', if_not_selected, task_name)
    if dataset_name == 'ATP':
        metric = 'Mcc'
    else:
        metric = 'Acc'
    all_max_acc = pd.concat([t])[f'{metric} mean'].to_numpy().max()
    while selected_train_embs.shape[1] > 1:
        temp_record = []
        for remove_i in range(selected_train_embs.shape[1]):
            concat_train_embs = np.concatenate([selected_train_embs[:, :remove_i], selected_train_embs[:, remove_i+1:]], axis=1)
            _, t = get_5fold_test_probs([concat_train_embs], [f'len{selected_train_embs.shape[1]} drop {remove_i}'], tr_labels, f'results/probs_txt/hand_single_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}_drop{remove_i}', if_not_selected, task_name)
            temp_record.append(t)
        record = pd.concat(temp_record)
        max_acc_id = record[f'{metric} mean'].to_numpy().argmax()
        max_acc = record[f'{metric} mean'].to_numpy().max()
        if max_acc < all_max_acc:
            _, t = get_ind_test_probs([selected_train_embs], [selected_test_embs], [f'{emb_name} depart len{selected_train_embs.shape[1]}'], tr_labels, tt_labels, f'results/probs_txt/hand_single_depart/fast_ensemble_{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}', if_not_selected, task_name + '_ind')
            print(f'{emb_name} : {tr_emb.shape[1]} => {selected_train_embs.shape[1]}')
            print(f'Has delete {delete_id_list}')
            np.save(f'results/hand/{task_name}.npy', delete_id_list)
            return selected_train_embs, selected_test_embs
        else:
            all_max_acc = max_acc
            selected_train_embs = np.delete(selected_train_embs, max_acc_id, axis=1)
            selected_test_embs = np.delete(selected_test_embs, max_acc_id, axis=1)
            delete_id_list.append(max_acc_id)

def ESBS_festure_set(tr_emb, tt_emb, emb_name, tr_labels, tt_labels, dataset_name):
    selected_train_embs = tr_emb
    selected_test_embs = tt_emb
    task_name = f'{dataset_name}_{emb_name}_select_itself_ESBS'
    if_not_selected = not os.path.exists(f'results/hand/{task_name}.npy')
    ensemble_id_list = []
    sub_train_embs, sub_test_embs = [], []
    if not if_not_selected:
        print(f'{dataset_name} {emb_name} has selected itself : )')
        ensemble_id_list = np.load(f'results/hand/{task_name}.npy')
        for ensemble_i in ensemble_id_list:
            sub_train_embs.append(selected_train_embs[:, ensemble_i].reshape(selected_train_embs.shape[0], 1))
            sub_test_embs.append(selected_test_embs[:, ensemble_i].reshape(selected_test_embs.shape[0], 1))
            selected_train_embs = np.delete(selected_train_embs, ensemble_i, axis=1)
            selected_test_embs = np.delete(selected_test_embs, ensemble_i, axis=1)
        if len(sub_train_embs) > 0: new_emb_ways = [f'{emb_name}#{i+1}' for i in range(len(sub_train_embs) + 1)]
        else: new_emb_ways = [emb_name]
        return [selected_train_embs] + sub_train_embs, [selected_test_embs] + sub_test_embs, new_emb_ways
    _, t = get_5fold_test_probs([selected_train_embs], [emb_name], tr_labels, f'results/probs_txt/hand_single_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}', if_not_selected, task_name)
    if dataset_name == 'ATP':
        metric = 'Mcc'
    else:
        metric = 'Acc'
    all_max_acc = pd.concat([t])[f'{metric} mean'].to_numpy().max()
    while selected_train_embs.shape[1] > 1:
        temp_record_keep = []
        for drop_i in range(selected_train_embs.shape[1]):
            concat_train_embs = [np.concatenate([selected_train_embs[:, :drop_i], selected_train_embs[:, drop_i+1:]], axis=1)] + sub_train_embs
            keep_concat_train_embs = concat_train_embs + [selected_train_embs[:, drop_i].reshape(selected_train_embs.shape[0], 1)]
            _, t = get_5fold_test_probs(keep_concat_train_embs, [f'len {selected_train_embs.shape[1]} ensemble {drop_i}'], tr_labels, f'results/probs_txt/hand_single_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}_ensemble{drop_i}', if_not_selected, task_name)
            temp_record_keep.append(t)
        record = pd.concat(temp_record_keep)
        max_acc_id_keep = record[f'{metric} mean'].to_numpy().argmax()
        max_acc_keep = record[f'{metric} mean'].to_numpy().max()
        action_id = np.argmax([all_max_acc, max_acc_keep])
        if action_id == 0:
            # stop selection
            _, t = get_ind_test_probs([selected_train_embs], [selected_test_embs], [f'{emb_name} depart len{selected_train_embs.shape[1]}'], tr_labels, tt_labels, f'results/probs_txt/hand_single_depart/fast_ensemble_{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}', if_not_selected, task_name + '_ind')
            print(f'{emb_name} : {tr_emb.shape[1]} => {selected_train_embs.shape[1]}')
            print(f'Has ensemble {ensemble_id_list}')
            np.save(f'results/hand/{task_name}.npy', ensemble_id_list)
            if len(sub_train_embs) > 0: new_emb_ways = [f'{emb_name}#{i+1}' for i in range(len(sub_train_embs) + 1)]
            else: new_emb_ways = [emb_name]
            return [selected_train_embs] + sub_train_embs, [selected_test_embs] + sub_test_embs, new_emb_ways
        elif action_id == 1:
            # keep
            all_max_acc = max_acc_keep
            sub_train_embs.append(selected_train_embs[:, max_acc_id_keep].reshape(selected_train_embs.shape[0], 1))
            sub_test_embs.append(selected_test_embs[:, max_acc_id_keep].reshape(selected_test_embs.shape[0], 1))
            selected_train_embs = np.delete(selected_train_embs, max_acc_id_keep, axis=1)
            selected_test_embs = np.delete(selected_test_embs, max_acc_id_keep, axis=1)
            ensemble_id_list.append(max_acc_id_keep)

def DESBS_festure_set(tr_emb, tt_emb, emb_name, tr_labels, tt_labels, dataset_name):
    selected_train_embs = tr_emb
    selected_test_embs = tt_emb
    task_name = f'{dataset_name}_{emb_name}_select_itself_DESBS'
    if_not_selected = not os.path.exists(f'results/hand/{task_name}.npy')
    drop_ensemble_id_list = []
    sub_train_embs, sub_test_embs = [], []
    if not if_not_selected:
        print(f'{dataset_name} {emb_name} has selected itself : )')
        drop_ensemble_id_list = np.load(f'results/hand/{task_name}.npy')
        for (action, de_i)  in drop_ensemble_id_list:
            if action == 'e':
                de_i = int(de_i)
                sub_train_embs.append(selected_train_embs[:, de_i].reshape(selected_train_embs.shape[0], 1))
                sub_test_embs.append(selected_test_embs[:, de_i].reshape(selected_test_embs.shape[0], 1))
            selected_train_embs = np.delete(selected_train_embs, de_i, axis=1)
            selected_test_embs = np.delete(selected_test_embs, de_i, axis=1)
        if len(sub_train_embs) > 0: new_emb_ways = [f'{emb_name}#{i+1}' for i in range(len(sub_train_embs) + 1)]
        else: new_emb_ways = [emb_name]
        return [selected_train_embs] + sub_train_embs, [selected_test_embs] + sub_test_embs, new_emb_ways
    _, t = get_5fold_test_probs([selected_train_embs], [emb_name], tr_labels, f'results/probs_txt/hand_single_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}', if_not_selected, task_name)
    if dataset_name == 'ATP':
        metric = 'Mcc'
    else:
        metric = 'Acc'
    all_max_acc = pd.concat([t])[f'{metric} mean'].to_numpy().max()
    while selected_train_embs.shape[1] > 1:
        temp_record_drop, temp_record_keep = [], []
        for drop_i in range(selected_train_embs.shape[1]):
            concat_train_embs = [np.concatenate([selected_train_embs[:, :drop_i], selected_train_embs[:, drop_i+1:]], axis=1)] + sub_train_embs
            _, t = get_5fold_test_probs(concat_train_embs, [f'len {selected_train_embs.shape[1]} drop {drop_i}'], tr_labels, f'results/probs_txt/hand_single_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}_drop{drop_i}', if_not_selected, task_name)
            temp_record_drop.append(t)
            keep_concat_train_embs = concat_train_embs + [selected_train_embs[:, drop_i].reshape(selected_train_embs.shape[0], 1)]
            _, t = get_5fold_test_probs(keep_concat_train_embs, [f'len {selected_train_embs.shape[1]} ensemble {drop_i}'], tr_labels, f'results/probs_txt/hand_single_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}_ensemble{drop_i}', if_not_selected, task_name)
            temp_record_keep.append(t)
        record_drop = pd.concat(temp_record_drop)
        record_keep = pd.concat(temp_record_keep)
        max_acc_id_drop = record_drop[f'{metric} mean'].to_numpy().argmax()
        max_acc_drop = record_drop[f'{metric} mean'].to_numpy().max()
        max_acc_id_keep = record_keep[f'{metric} mean'].to_numpy().argmax()
        max_acc_keep = record_keep[f'{metric} mean'].to_numpy().max()
        action_id = np.argmax([all_max_acc, max_acc_drop, max_acc_keep])
        if action_id == 0:
            # stop selection
            _, t = get_ind_test_probs([selected_train_embs], [selected_test_embs], [f'{emb_name} depart len{selected_train_embs.shape[1]}'], tr_labels, tt_labels, f'results/probs_txt/hand_single_depart/fast_ensemble_{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}', if_not_selected, task_name + '_ind')
            print(f'{emb_name} : {tr_emb.shape[1]} => {selected_train_embs.shape[1]}')
            print(f'Has ensemble and drop {drop_ensemble_id_list}')
            np.save(f'results/hand/{task_name}.npy', drop_ensemble_id_list)
            if len(sub_train_embs) > 0: new_emb_ways = [f'{emb_name}#{i+1}' for i in range(len(sub_train_embs) + 1)]
            else: new_emb_ways = [emb_name]
            return [selected_train_embs] + sub_train_embs, [selected_test_embs] + sub_test_embs, new_emb_ways
        elif action_id == 1:
            # drop
            all_max_acc = max_acc_drop
            selected_train_embs = np.delete(selected_train_embs, max_acc_id_drop, axis=1)
            selected_test_embs = np.delete(selected_test_embs, max_acc_id_drop, axis=1)
            drop_ensemble_id_list.append(('d', max_acc_id_drop))
        elif action_id == 2:
            # keep
            all_max_acc = max_acc_keep
            sub_train_embs.append(selected_train_embs[:, max_acc_id_keep].reshape(selected_train_embs.shape[0], 1))
            sub_test_embs.append(selected_test_embs[:, max_acc_id_keep].reshape(selected_test_embs.shape[0], 1))
            selected_train_embs = np.delete(selected_train_embs, max_acc_id_keep, axis=1)
            selected_test_embs = np.delete(selected_test_embs, max_acc_id_keep, axis=1)
            drop_ensemble_id_list.append(('e', max_acc_id_keep))

def CDESBS_festure_set(tr_emb, tt_emb, emb_name, tr_labels, tt_labels, dataset_name):
    selected_train_embs = tr_emb
    selected_test_embs = tt_emb
    task_name = f'{dataset_name}_{emb_name}_select_itself_CDESBS'
    if_not_selected = not os.path.exists(f'results/hand/{task_name}.npy')
    concat_drop_ensemble_id_list = []
    sub_train_embs, sub_test_embs = [], []
    if not if_not_selected:
        print(f'{dataset_name} {emb_name} has selected itself : )')
        concat_drop_ensemble_id_list = np.load(f'results/hand/{task_name}.npy')
        for (action, cde_i)  in concat_drop_ensemble_id_list:
            if action == 'd':
                cde_i = int(cde_i)
                selected_train_embs = np.delete(selected_train_embs, cde_i, axis=1)
                selected_test_embs = np.delete(selected_test_embs, cde_i, axis=1)
            elif action == 'e':
                cde_i = int(cde_i)
                sub_train_embs.append(selected_train_embs[:, cde_i].reshape(selected_train_embs.shape[0], 1))
                sub_test_embs.append(selected_test_embs[:, cde_i].reshape(selected_test_embs.shape[0], 1))
                selected_train_embs = np.delete(selected_train_embs, cde_i, axis=1)
                selected_test_embs = np.delete(selected_test_embs, cde_i, axis=1)
            elif action == 'c':
                di = int(cdi[0])
                ci = int(cdi[1])
                sub_train_embs[ci] = np.concatenate([sub_train_embs[ci], selected_train_embs[:, di].reshape(selected_train_embs.shape[0], 1)], axis=1)
                sub_test_embs[ci] = np.concatenate([sub_test_embs[ci], selected_test_embs[:, di].reshape(selected_test_embs.shape[0], 1)], axis=1)
                selected_train_embs = np.delete(selected_train_embs, di, axis=1)
                selected_test_embs = np.delete(selected_test_embs, di, axis=1)
            
        if len(sub_train_embs) > 0: new_emb_ways = [f'{emb_name}#{i+1}' for i in range(len(sub_train_embs) + 1)]
        else: new_emb_ways = [emb_name]
        return [selected_train_embs] + sub_train_embs, [selected_test_embs] + sub_test_embs, new_emb_ways
    # _, t = get_5fold_test_probs([selected_train_embs], [emb_name], tr_labels, f'results/probs_txt/hand_single_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}', if_not_selected, task_name)
    _, t = get_5fold_test_probs_fast([selected_train_embs], [emb_name], tr_labels, f'results/probs_txt/hand_single_depart_fast/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}', if_not_selected, task_name)
    if dataset_name == 'ATP':
        metric = 'Mcc'
    else:
        metric = 'Acc'
    all_max_acc = pd.concat([t])[f'{metric} mean'].to_numpy().max()
    while selected_train_embs.shape[1] > 1:
        temp_record_drop, temp_record_keep, temp_record_concat = [], [], []
        for drop_i in range(selected_train_embs.shape[1]):
            concat_train_embs = [np.concatenate([selected_train_embs[:, :drop_i], selected_train_embs[:, drop_i+1:]], axis=1)] + sub_train_embs
            # _, t = get_5fold_test_probs(concat_train_embs, [f'len {selected_train_embs.shape[1]} drop {drop_i}'], tr_labels, f'results/probs_txt/hand_single_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}_drop{drop_i}', if_not_selected, task_name)
            _, t = get_5fold_test_probs_fast(concat_train_embs, [f'len {selected_train_embs.shape[1]} drop {drop_i}'], tr_labels, f'results/probs_txt/hand_single_depart_fast/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}_drop{drop_i}', if_not_selected, task_name)
            temp_record_drop.append(t)
            keep_concat_train_embs = concat_train_embs + [selected_train_embs[:, drop_i].reshape(selected_train_embs.shape[0], 1)]
            # _, t = get_5fold_test_probs(keep_concat_train_embs, [f'len {selected_train_embs.shape[1]} ensemble {drop_i}'], tr_labels, f'results/probs_txt/hand_single_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}_ensemble{drop_i}', if_not_selected, task_name)
            _, t = get_5fold_test_probs_fast(keep_concat_train_embs, [f'len {selected_train_embs.shape[1]} ensemble {drop_i}'], tr_labels, f'results/probs_txt/hand_single_depart_fast/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}_ensemble{drop_i}', if_not_selected, task_name)
            temp_record_keep.append(t)
            # concat with sub
            if len(sub_train_embs) > 0:
                for concat_i in range(len(sub_train_embs)):
                    concat_with_sub_concat_train_embs = concat_train_embs[:concat_i+1] + concat_train_embs[concat_i+2:] + [np.concatenate([concat_train_embs[concat_i+1], selected_train_embs[:, drop_i].reshape(selected_train_embs.shape[0], 1)], axis=1)]
                    # _, t = get_5fold_test_probs(concat_with_sub_concat_train_embs, [f'len {selected_train_embs.shape[1]} concat {drop_i} to sub{concat_i}'], tr_labels, f'results/probs_txt/hand_single_depart/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}_concat{drop_i}to{concat_i}', if_not_selected, task_name)
                    _, t = get_5fold_test_probs_fast(concat_with_sub_concat_train_embs, [f'len {selected_train_embs.shape[1]} concat {drop_i} to sub{concat_i}'], tr_labels, f'results/probs_txt/hand_single_depart_fast/{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}_concat{drop_i}to{concat_i}', if_not_selected, task_name)
                    temp_record_concat.append(t)
        record_drop = pd.concat(temp_record_drop)
        record_keep = pd.concat(temp_record_keep)
        if len(temp_record_concat) > 0:
            record_concat = pd.concat(temp_record_concat)
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
            _, t = get_ind_test_probs([selected_train_embs], [selected_test_embs], [f'{emb_name} depart len{selected_train_embs.shape[1]}'], tr_labels, tt_labels, f'results/probs_txt/hand_single_depart/fast_ensemble_{dataset_name}_{task_name}_len{selected_train_embs.shape[1]}', if_not_selected, task_name + '_ind')
            print(f'{emb_name} : {tr_emb.shape[1]} => {selected_train_embs.shape[1]}')
            print(f'Has concat, ensemble and drop {concat_drop_ensemble_id_list}')
            np.save(f'results/hand/{task_name}.npy', concat_drop_ensemble_id_list)
            if len(sub_train_embs) > 0: new_emb_ways = [f'{emb_name}#{i+1}' for i in range(len(sub_train_embs) + 1)]
            else: new_emb_ways = [emb_name]
            return [selected_train_embs] + sub_train_embs, [selected_test_embs] + sub_test_embs, new_emb_ways
        elif action_id == 1:
            # drop
            all_max_acc = max_acc_drop
            selected_train_embs = np.delete(selected_train_embs, max_acc_id_drop, axis=1)
            selected_test_embs = np.delete(selected_test_embs, max_acc_id_drop, axis=1)
            concat_drop_ensemble_id_list.append(('d', max_acc_id_drop))
        elif action_id == 2:
            # ensemble
            all_max_acc = max_acc_keep
            sub_train_embs.append(selected_train_embs[:, max_acc_id_keep].reshape(selected_train_embs.shape[0], 1))
            sub_test_embs.append(selected_test_embs[:, max_acc_id_keep].reshape(selected_test_embs.shape[0], 1))
            selected_train_embs = np.delete(selected_train_embs, max_acc_id_keep, axis=1)
            selected_test_embs = np.delete(selected_test_embs, max_acc_id_keep, axis=1)
            concat_drop_ensemble_id_list.append(('e', max_acc_id_keep))
        elif action_id == 3:
            # concat
            all_max_acc = max_acc_concat
            drop_id = max_acc_id_concat // len(sub_train_embs)
            concat_id = max_acc_id_concat % len(sub_train_embs)
            sub_train_embs[concat_id] = np.concatenate([sub_train_embs[concat_id], selected_train_embs[:, drop_id].reshape(selected_train_embs.shape[0], 1)], axis=1)
            sub_test_embs[concat_id] = np.concatenate([sub_test_embs[concat_id], selected_test_embs[:, drop_id].reshape(selected_test_embs.shape[0], 1)], axis=1)
            selected_train_embs = np.delete(selected_train_embs, drop_id, axis=1)
            selected_test_embs = np.delete(selected_test_embs, drop_id, axis=1)
            concat_drop_ensemble_id_list.append(('c', [drop_id, concat_id]))

def select_in_feature_set(tr_embs, tt_embs, emb_ways, tr_labels, tt_labels, dataset_name):
    selected_tr_embs, selected_tt_embs, all_emb_ways = [], [], []
    for i in range(len(tr_embs)):
        # selected_tr_emb, selected_tt_emb = FS_feature_set(tr_embs[i], tt_embs[i], emb_ways[i], tr_labels, tt_labels, dataset_name)
        # selected_tr_emb, selected_tt_emb = BS_feature_set(tr_embs[i], tt_embs[i], emb_ways[i], tr_labels, tt_labels, dataset_name)
        # selected_tr_embs.append(selected_tr_emb)
        # selected_tt_embs.append(selected_tt_emb)
        # selected_tr_emb, selected_tt_emb, emb_ways = ESBS_festure_set(tr_embs[i], tt_embs[i], emb_ways[i], tr_labels, tt_labels, dataset_name)
        # selected_tr_emb, selected_tt_emb, emb_ways = DESBS_festure_set(tr_embs[i], tt_embs[i], emb_ways[i], tr_labels, tt_labels, dataset_name)
        selected_tr_emb, selected_tt_emb, emb_ways = CDESBS_festure_set(tr_embs[i], tt_embs[i], emb_ways[i], tr_labels, tt_labels, dataset_name)
        all_emb_ways.append(emb_ways)
        for j in range(len(selected_tr_emb)):
            selected_tr_embs.append(selected_tr_emb[j])
            selected_tt_embs.append(selected_tt_emb[j])
            all_emb_ways.append(emb_ways[j])

    return selected_tr_embs, selected_tt_embs, all_emb_ways

def first_stage(train_data_file_name, test_data_file_name, pad_size, dataset_name, feature_id):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    # only hand-craft
    train_embs = [tr_embs[feature_id]]
    test_embs = [tt_embs[feature_id]]
    all_emb_ways = [emb_ways[feature_id]]
    train_embs, test_embs, all_emb_ways = select_in_feature_set(train_embs, test_embs, all_emb_ways, tr_labels, tt_labels, dataset_name)

def check_features_same(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = load_hc_from_ATPFinder(train_data_file_name, test_data_file_name, dataset_name)
    print(np.concatenate(tr_embs, axis=1).shape)
    print(tr_labels.shape)

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
    dataset_names = ['ATP']
    pad_sizes = [75]
    dataset_names = ['MD']
    pad_sizes = [61]
    dataset_names = ['RD']
    pad_sizes = [61]
    FS_list, BS_list = [], []
    for i, dataset_name in enumerate(dataset_names):
        train_data_file_name = f'data/ATPdataset/{dataset_name}_train.csv'
        test_data_file_name = f'data/ATPdataset/{dataset_name}_test.csv'
        pad_size = pad_sizes[i]
        # FS_list.append(FS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # BS_list.append(BS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # FS_list.append(FS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name, True))
        # BS_list.append(BS_ATP(train_data_file_name, test_data_file_name, pad_size, dataset_name, True))
        # find_best_est(train_data_file_name, test_data_file_name, pad_size, dataset_name, True)
        # fast_ensemble(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        # FS_list.append(FS_keep_or_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # BS_list.append(BS_concat_first_keep(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # BS_list.append(BS_keep_or_drop(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # BS_list.append(BS_keep_or_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # BS_list.append(BS_keep_or_drop_or_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # BS_list.append(BS_concat_first_keep_or_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        BS_list.append(BS_concat(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # FS_list.append(select_RF_hyper(train_data_file_name, test_data_file_name, pad_size, dataset_name))
        # BS_hc_depart(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        # first_stage(train_data_file_name, test_data_file_name, pad_size, dataset_name, 13)
        # first_stage(train_data_file_name, test_data_file_name, pad_size, dataset_name, 14)
        # first_stage(train_data_file_name, test_data_file_name, pad_size, dataset_name, 15)
        # first_stage(train_data_file_name, test_data_file_name, pad_size, dataset_name, 16)
        # first_stage(train_data_file_name, test_data_file_name, pad_size, dataset_name, 17)
        # check_features_same(train_data_file_name, test_data_file_name, pad_size, dataset_name)
    if len(FS_list) != 0:
        print('FS')
        print(FS_list)
    if len(BS_list) != 0:
        print('BS')
        print(BS_list)