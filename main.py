import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import r_regression, SelectKBest
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

from utils import *
from feature import *
from model import CNN, RNN, LSTM, get_model
# from data2feature import AAC_DPC_TPC, PseAAC, CKSAAGP

def esm2_CNN(data_file_path, device):
    # HyperParameter
    hyper_args = {
            'model': 'CNN',
            'max_epoch' : 50,
            'batch_size' : 32,
            'lr' : 0.001,
            }
    all_performance = {'Acc':[], 'Mcc':[], 'Sensitivity':[], 'Specificity':[], 'AUC':[]}
    # load data
    # embs = get_esm_embs(data_file_path, 41, device)
    embs = get_esm_embs(data_file_name, 41, 'seq', device, False)
    _, labels = generate_features(data_file_path, 41)

    # start training
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(embs)):
            print(f'Time {time} Fold {k}')
            tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
            tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
            tr_dataset = ADPDataset(tr_embs, tr_labels)
            tt_dataset = ADPDataset(tt_embs, tt_labels)
            tr_set = DataLoader(tr_dataset, batch_size=hyper_args['batch_size'], shuffle=True)
            tt_set = DataLoader(tt_dataset, batch_size=hyper_args['batch_size'], shuffle=False)
            model = CNN(embs.shape[-1])
            optimizer = torch.optim.Adam(model.parameters(), lr=hyper_args['lr'])
            train_perform, best_model_name, best_epoch = evaluate(tr_set, tt_set, model, optimizer, hyper_args, time, k, device, 'train')
            print(f'Finish Training at epoch {best_epoch}')
            for k, v in train_perform.items():
                print(f'{k} : {v}')
            print(f'Load {best_model_name}.pth')
            model.load_state_dict(torch.load(f'models/{best_model_name}.pth'))
            test_perform = evaluate(tr_set, tt_set, model, '', '', 0, 0, device, 'test')
            print('Test Result:')
            for k, v in test_perform.items():
                print(f'{k} : {v}')
                all_performance[k].append(v)
    print('Finish 10 times 5 fold :')
    for k, v in all_performance.items():
        print(f'{k} : {np.mean(v)} +- {np.std(v)}')

def esm2_RNN(data_file_path, device):
    # HyperParameter
    hyper_args = {
            'model': 'RNN',
            'max_epoch' : 100,
            'batch_size' : 32,
            'lr' : 0.001,
            }
    all_performance = {'Acc':[], 'Mcc':[], 'Sensitivity':[], 'Specificity':[], 'AUC':[]}
    # load data
    embs = get_esm_embs(data_file_path, 40, device)
    _, labels = generate_features(data_file_path)

    # start training
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(embs)):
            print(f'Time {time} Fold {k}')
            tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
            tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
            tr_dataset = ADPDataset(tr_embs, tr_labels)
            tt_dataset = ADPDataset(tt_embs, tt_labels)
            tr_set = DataLoader(tr_dataset, batch_size=hyper_args['batch_size'], shuffle=True)
            tt_set = DataLoader(tt_dataset, batch_size=hyper_args['batch_size'], shuffle=False)
            model = RNN(embs.shape[-1])
            optimizer = torch.optim.Adam(model.parameters(), lr=hyper_args['lr'])
            train_perform, best_model_name, best_epoch = evaluate(tr_set, tt_set, model, optimizer, hyper_args, time, k, device, 'train')
            print(f'Finish Training at epoch {best_epoch}')
            for k, v in train_perform.items():
                print(f'{k} : {v}')
            print(f'Load {best_model_name}.pth')
            model.load_state_dict(torch.load(f'{best_model_name}.pth'))
            test_perform = evaluate(tr_set, tt_set, model, '', '', 0, 0, device, 'test')
            print('Test Result:')
            for k, v in test_perform.items():
                print(f'{k} : {v}')
                all_performance[k].append(v)
    print('Finish 10 times 5 fold :')
    for k, v in all_performance.items():
        print(f'{k} : {np.mean(v)} +- {np.std(v)}')

def esm2_LSTM(data_file_path, device):
    # HyperParameter
    hyper_args = {
            'model': 'LSTM',
            'max_epoch' : 100,
            'batch_size' : 32,
            'lr' : 0.001,
            }
    all_performance = {'Acc':[], 'Mcc':[], 'Sensitivity':[], 'Specificity':[], 'AUC':[]}
    # load data
    embs = get_esm_embs(data_file_path, 40, device)
    _, labels = generate_features(data_file_path)

    # start training
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(embs)):
            print(f'Time {time} Fold {k}')
            tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
            tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
            tr_dataset = ADPDataset(tr_embs, tr_labels)
            tt_dataset = ADPDataset(tt_embs, tt_labels)
            tr_set = DataLoader(tr_dataset, batch_size=hyper_args['batch_size'], shuffle=True)
            tt_set = DataLoader(tt_dataset, batch_size=hyper_args['batch_size'], shuffle=False)
            model = LSTM(embs.shape[-1])
            optimizer = torch.optim.Adam(model.parameters(), lr=hyper_args['lr'])
            train_perform, best_model_name, best_epoch = evaluate(tr_set, tt_set, model, optimizer, hyper_args, time, k, device, 'train')
            print(f'Finish Training at epoch {best_epoch}')
            for k, v in train_perform.items():
                print(f'{k} : {v}')
            print(f'Load {best_model_name}.pth')
            model.load_state_dict(torch.load(f'{best_model_name}.pth'))
            test_perform = evaluate(tr_set, tt_set, model, '', '', 0, 0, device, 'test')
            print('Test Result:')
            for k, v in test_perform.items():
                print(f'{k} : {v}')
                all_performance[k].append(v)
    print('Finish 10 times 5 fold :')
    for k, v in all_performance.items():
        print(f'{k} : {np.mean(v)} +- {np.std(v)}')

def esm2_hand_RF(data_file_path, device):
    # HyperParameter
    hyper_args = {
            'model': 'RF',
            'batch_size' : 32,
            }
    all_performance = {'Acc':[], 'Mcc':[], 'Sensitivity':[], 'Specificity':[], 'AUC':[]}
    # load data
    esm_embs = get_esm_embs(data_file_path, 40, device)
    # flat
    flat_esm_embs = np.array(esm_embs).reshape(len(esm_embs), -1)
    # avg
    avg_esm_embs = np.array(esm_embs).mean(axis=1)
    seqs, labels = generate_features(data_file_path)
    AAC_embs = AAC_DPC_TPC(seqs, 1)
    DPC_embs = AAC_DPC_TPC(seqs, 2)
    PAAC_embs = PseAAC(seqs, AAC_embs, 5, 0.05)
    CKS_embs = CKSAAGP(seqs, gap = 2)
    embs = np.concatenate((avg_esm_embs, AAC_embs, DPC_embs, PAAC_embs, CKS_embs), axis=1)
    # start training
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(embs)):
            print(f'Time {time} Fold {k}')
            tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
            tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
            clf = RandomForestClassifier(n_estimators=500, random_state=42)
            clf.fit(tr_embs, tr_labels)
            train_pred = clf.predict(tr_embs)
            train_pred_prob = clf.predict_proba(tr_embs)[:, 1]
            acc, mcc, sen, spe, auc = performance(train_pred, train_pred_prob, tr_labels)
            train_perform = {'Acc':acc, 'Mcc':mcc, 'Sensitivity':sen, 'Specificity':spe, 'AUC':auc}
            for k, v in train_perform.items():
                print(f'{k} : {v}')
            pred = clf.predict(tt_embs)
            pred_prob = clf.predict_proba(tt_embs)[:, 1]
            acc, mcc, sen, spe, auc = performance(pred, pred_prob, tt_labels)
            test_perform = {'Acc':acc, 'Mcc':mcc, 'Sensitivity':sen, 'Specificity':spe, 'AUC':auc}
            print('Test Result:')
            for k, v in test_perform.items():
                print(f'{k} : {v}')
                all_performance[k].append(v)
    print('Finish 10 times 5 fold :')
    for k, v in all_performance.items():
        print(f'{k} : {np.mean(v)} +- {np.std(v)}')

def mean_of_10times_5fold_M(embs, labels, model_name, device):
    '''
    For manchine learning.
    '''
    all_performance = {'Acc':[], 'Mcc':[], 'Sensitivity':[], 'Specificity':[], 'AUC':[]}
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(embs)):
            print(f'Time {time} Fold {k}')
            tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
            tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
            clf = get_model(model_name, embs.shape[-1])
            clf.fit(tr_embs, tr_labels)
            train_pred = clf.predict(tr_embs)
            train_pred_prob = clf.predict_proba(tr_embs)[:, 1]
            acc, mcc, sen, spe, auc = performance(train_pred, train_pred_prob, tr_labels)
            train_perform = {'Acc':acc, 'Mcc':mcc, 'Sensitivity':sen, 'Specificity':spe, 'AUC':auc}
            for k, v in train_perform.items():
                print(f'{k} : {v}')
            pred = clf.predict(tt_embs)
            pred_prob = clf.predict_proba(tt_embs)[:, 1]
            acc, mcc, sen, spe, auc = performance(pred, pred_prob, tt_labels)
            test_perform = {'Acc':acc, 'Mcc':mcc, 'Sensitivity':sen, 'Specificity':spe, 'AUC':auc}
            print('Test Result:')
            for k, v in test_perform.items():
                print(f'{k} : {v}')
                all_performance[k].append(v)
    print('Finish 10 times 5 fold :')
    for k, v in all_performance.items():
        print(f'{k} : {np.mean(v)} +- {np.std(v)}')
    return train_perform, test_perform, all_performance

def mean_of_10times_5fold_D(embs, labels, model_name, hyper_args, device):
    # start training
    all_performance = {'Acc':[], 'Mcc':[], 'Sensitivity':[], 'Specificity':[], 'AUC':[]}
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(embs)):
            print(f'Time {time} Fold {k}')
            tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
            tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
            tr_dataset = ADPDataset(tr_embs, tr_labels)
            tt_dataset = ADPDataset(tt_embs, tt_labels)
            tr_set = DataLoader(tr_dataset, batch_size=hyper_args['batch_size'], shuffle=True)
            tt_set = DataLoader(tt_dataset, batch_size=hyper_args['batch_size'], shuffle=False)
            model = get_model(model_name, embs.shape[-1])
            optimizer = torch.optim.Adam(model.parameters(), lr=hyper_args['lr'])
            train_perform, best_model_name, best_epoch = evaluate(tr_set, tt_set, model, optimizer, hyper_args, time, k, device, 'train')
            print(f'Finish Training at epoch {best_epoch}')
            for k, v in train_perform.items():
                print(f'{k} : {v}')
            print(f'Load {best_model_name}.pth')
            model.load_state_dict(torch.load(f'models/{best_model_name}.pth'))
            test_perform = evaluate(tr_set, tt_set, model, '', '', 0, 0, device, 'test')
            print('Test Result:')
            for k, v in test_perform.items():
                print(f'{k} : {v}')
                all_performance[k].append(v)
    print('Finish 10 times 5 fold :')
    for k, v in all_performance.items():
        print(f'{k} : {np.mean(v)} +- {np.std(v)}')
    return train_perform, test_perform, all_performance

def multipeptide_on_LLM(device):
    data_dir = 'data/many_peptide/'
    # peptide_list = ['ADP/benchmark', 'AAP/NT15', 'AAP/benchmark', 'CPP/benchmark', 'THP/main', 'THP/small']
    peptide_list = ['ADP/benchmark']
    cls_model = ['CNN', 'RNN', 'LSTM', 'RF', 'LGBM', 'XGB', 'CatB']
    emb_way = ['esm2_seq', 'esm2_cls', 'prott5', 'protbert_seq', 'protbert_cls']
    all_hyper_args = {
                'CNN': {
                    'model': 'CNN',
                    'max_epoch' : 50,
                    'batch_size' : 32,
                    'lr' : 0.001,
                },
                'RNN': {
                    'model': 'RNN',
                    'max_epoch' : 100,
                    'batch_size' : 32,
                    'lr' : 0.001,
                },
                'LSTM': {
                    'model': 'LSTM',
                    'max_epoch' : 100,
                    'batch_size' : 32,
                    'lr' : 0.001,
                },
            }
    for pep in peptide_list:
        # load dataset
        data_path = f'{data_dir}{pep}.fasta'
        pad_size = find_max_len(data_path)
        _, labels = generate_features(data_path, pad_size)
        for ew in emb_way:
            all_embs = []
            if 'esm2_seq' in ew:
                all_embs.append(get_esm_embs(data_path, pad_size, 'seq',  device))
            if 'esm2_cls' in ew:
                all_embs.append(get_esm_embs(data_path, pad_size, 'cls', device))
            if 'prott5' in ew:
                all_embs.append(get_prott5_embs(data_path, pad_size, device))
            if 'protbert_seq' in ew:
                all_embs.append(get_proteinbert_embs(data_path, pad_size, 'seq', device))
            if 'protbert_cls' in ew:
                all_embs.append(get_proteinbert_embs(data_path, pad_size, 'cls', device))
            all_embs = np.concatenate(all_embs, axis=-1)
            for cm in cls_model:
                model = get_model(cm, all_embs.shape[-1])
                # if machine learning, average length dim
                if cm in ['RF', 'LGBM', 'XGB', 'CatB']:
                    if '_cls' in ew:
                        avg_embs = all_embs
                    else:
                        avg_embs = all_embs.mean(axis=1)
                    train_perform, test_perform, all_performance = mean_of_10times_5fold_M(avg_embs, labels, cm, device)
                else:
                    if '_cls' in ew:
                        continue
                    hyper_args = all_hyper_args[cm]
                    train_perform, test_perform, all_performance = mean_of_10times_5fold_D(all_embs, labels, cm, hyper_args, device)                
                whole_model_name = '_'.join(pep.split('/')) + '_' + ew + '_' + cm
                save_performance_to_csv(train_perform, test_perform, all_performance, whole_model_name)

def mean_of_10times_5fold_M_return_prob(embs, labels, model_name, device):
    '''
    For manchine learning.
    '''
    all_performance = {'Acc':[], 'Mcc':[], 'Sens':[], 'Spec':[], 'AUC':[]}
    pred_prob_train_list = []
    pred_prob_test_list = []
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(embs)):
            print(f'Time {time} Fold {k}')
            tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
            tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
            clf = get_model(model_name, embs.shape[-1])
            clf.fit(tr_embs, tr_labels)
            train_pred = clf.predict(tr_embs)
            train_pred_prob = clf.predict_proba(tr_embs)[:, 1]
            pred_prob_train_list.append(train_pred_prob)
            pred = clf.predict(tt_embs)
            pred_prob = clf.predict_proba(tt_embs)[:, 1]
            pred_prob_test_list.append(pred_prob)
            acc, mcc, sen, spe, auc = performance(pred, pred_prob, tt_labels)
            test_perform = {'Acc':acc, 'Mcc':mcc, 'Sens':sen, 'Spec':spe, 'AUC':auc}
            print('Test Result:')
            for k, v in test_perform.items():
                print(f'{k} : {v}')
                all_performance[k].append(v)
    print('Finish 10 times 5 fold :')
    for k, v in all_performance.items():
        print(f'{k} : {np.mean(v)} +- {np.std(v)}')
    return all_performance, pred_prob_train_list, pred_prob_test_list

# def try_stack(data_file_name, device): # wrong!!! correct version is at line 714
    # esm2_embs = get_esm_embs(data_file_name, 41, '', device, True)
    # prott5_embs = get_prott5_embs(data_file_name, 41, device, True)
    # seqs, labels = generate_features(data_file_name, 41)
    # AAC_embs = AAC_DPC_TPC(seqs, 1)
    # PAAC_embs = PseAAC(seqs, AAC_embs, 3, 0.05)
    # APAAC_embs = Am_PseAAC(seqs, AAC_embs, 3, 0.05)
    # CTD_embs = calculate_ctd_features(seqs)
    # rdkit_embs = get_rdkit_embs(data_file_name, 41)
    # # no.42 is too big, standard scaling it
    # scaler = StandardScaler()
    # rdkit_embs[:, 42] = scaler.fit_transform(rdkit_embs[:, 42].reshape(-1, 1)).flatten()
    # embs_list = []
    # emb_ways = []
    # pad_seqs = pad_seq(seqs, 41)
    # for i in range(1, 11):
        # args = {
            # 'w2v_path' : "./models/w2v_model/",
            # 'emb_dim' : 128,
            # 'kmer' : i,
            # 'window' : 20,
            # 'epochs' : 10,
            # 'sg' : 1,
            # 'force_train' : False
        # }
        # w2v_model = train_w2v(pad_seqs, **args)
        # # embs_list.append(emb_seq_w2v(pad_seqs, w2v_model, args['kmer'], flatten=True))
        # embs_list.append(emb_seq_w2v(pad_seqs, w2v_model, args['kmer'], flatten=False).mean(1))
        # emb_ways.append(f'w2v{i}mer-avg')
    # # w2v_embs = emb_seq_w2v(pad_seqs, w2v_model, args['kmer'], flatten=False)
    # # w2v_embs = w2v_embs.mean(1)
    # # embs_list = [esm2_embs, prott5_embs, AAC_embs, PAAC_embs, APAAC_embs, CTD_embs, rdkit_embs]
    # # emb_ways = ['esm2', 'prott5', 'AAC', 'PAAC', 'APACC', 'CTD', 'rdkit']
    # # embs_list = [w2v_embs]
    # # emb_ways = ['w2v']
    # model_list = ['LGBM', 'XGB', 'RF', 'EXT']
    # record = {'embed way':[], 'model':[]}
    # for i, embs in enumerate(embs_list):
        # for model_name in model_list:
            # record['embed way'].append(emb_ways[i])
            # record['model'].append(model_name)
            # all_performance, pred_prob_train_list, pred_prob_test_list = mean_of_10times_5fold_M_return_prob(embs, labels, model_name, device)
            # for key, value in all_performance.items():
                # if f'{key} mean' not in record:
                    # record[f'{key} mean'] = []
                    # record[f'{key} std'] = []
                # record[f'{key} mean'].append(np.mean(value))
                # record[f'{key} std'].append(np.std(value))
            # save_txt(f'results/probs/{emb_ways[i]}_{model_name}_train_probs.pkl', pred_prob_train_list)
            # save_txt(f'results/probs/{emb_ways[i]}_{model_name}_test_probs.pkl', pred_prob_test_list)
    # df = pd.DataFrame(record)
    # df.to_csv(f'results/all_w2v-avg_stack.csv', index=False)

# def try_stack2():
    # '''
    # Load probs -> logestic reg
    # '''
    # # emb_ways = ['esm2', 'prott5', 'AAC', 'PAAC', 'APACC', 'CTD', 'rdkit', 'w2v']
    # emb_ways = [f'w2v{i}mer-avg' for i in range(1, 11)]
    # model_list = ['LGBM', 'XGB', 'RF', 'EXT']
    # train_data = [[] for i in range(50)]
    # test_data = [[] for i in range(50)]
    # # get labels
    # all_train_labels, all_test_labels = [], []
    # seqs, labels = generate_features(data_file_name, 41)
    # for time in range(10):
        # kf = KFold(n_splits=5, shuffle=True, random_state=time)
        # for k, (tr_idx, tt_idx) in enumerate(kf.split(seqs)):
            # all_train_labels.append(labels[tr_idx])
            # all_test_labels.append(labels[tt_idx])
    # for e in emb_ways:
        # for model_name in model_list:
            # train_prob_list = load_txt(f'results/probs/{e}_{model_name}_train_probs.pkl')
            # test_prob_list = load_txt(f'results/probs/{e}_{model_name}_test_probs.pkl')
            # for tt in range(len(train_prob_list)):
                # train_data[tt].append(train_prob_list[tt])
                # test_data[tt].append(test_prob_list[tt])
    # for i in range(len(train_data)):
        # train_data[i] = np.array(train_data[i]).T
        # test_data[i] = np.array(test_data[i]).T
    # all_train_performance = {'Acc':[], 'Mcc':[], 'Sens':[], 'Spec':[], 'AUC':[]}
    # all_test_performance = {'Acc':[], 'Mcc':[], 'Sens':[], 'Spec':[], 'AUC':[]}
    # for i in range(len(train_data)):
        # model = LogisticRegression()
        # model.fit(train_data[i], all_train_labels[i])
        # train_y_pred = model.predict(train_data[i])
        # train_y_pred_prob = model.predict_proba(train_data[i])[:, 1]
        # acc, mcc, sen, spe, auc = performance(train_y_pred, train_y_pred_prob, all_train_labels[i])
        # train_performance = {'Acc':acc, 'Mcc':mcc, 'Sens':sen, 'Spec':spe, 'AUC':auc}
        # for k, v in train_performance.items():
            # all_train_performance[k].append(v)
        # test_y_pred = model.predict(test_data[i])
        # test_y_pred_prob = model.predict_proba(test_data[i])[:, 1]
        # acc, mcc, sen, spe, auc = performance(test_y_pred, test_y_pred_prob, all_test_labels[i])
        # test_performance = {'Acc':acc, 'Mcc':mcc, 'Sens':sen, 'Spec':spe, 'AUC':auc}
        # for k, v in test_performance.items():
            # all_test_performance[k].append(v)
    # print(emb_ways)
    # print(model_list)
    # print('Train:')
    # for k, v in train_performance.items():
        # print(f'{k} : {np.mean(v)} +- {np.std(v)}')
    # print('Test:')
    # for k, v in test_performance.items():
        # print(f'{k} : {np.mean(v)} +- {np.std(v)}')

def test_w2v(data_file_name, device):
    args = {
        'w2v_path' : "./models/w2v_model/",
        'emb_dim' : 128,
        'kmer' : 9,
        'window' : 20,
        'epochs' : 10,
        'sg' : 1,
        'force_train' : False
    }
    seqs, labels = generate_features(data_file_name, 41)
    pad_seqs = pad_seq(seqs, 41)
    w2v_model = train_w2v(pad_seqs, **args)
    embs = emb_seq_w2v(pad_seqs, w2v_model, args['kmer'], flatten=False)
    embs = embs.mean(1)
    model_name = 'RF'
    mean_of_10times_5fold_M(embs, labels, model_name, device) 

def ensemble_w2v(data_file_name):
    seqs, labels = generate_features(data_file_name, 41)
    pad_seqs = pad_seq(seqs, 41)
    pad_seqs = np.array(pad_seqs)
    all_test_performance = {'Acc':[], 'Mcc':[], 'Sens':[], 'Spec':[], 'AUC':[]}
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(pad_seqs)):
            print(f'Time {time} Fold {k}')
            tr_seqs, tt_seqs = pad_seqs[tr_idx], pad_seqs[tt_idx]
            tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
            each_mer_probs = []
            for i in range(1, 11):
                args = {
                    'w2v_path' : "./models/w2v_model/",
                    'emb_dim' : 128,
                    'kmer' : i,
                    'window' : 20,
                    'epochs' : 10,
                    'sg' : 1,
                    'force_train' : False
                }
                w2v_model = train_w2v(pad_seqs, **args)
                tr_embs = emb_seq_w2v(tr_seqs, w2v_model, args['kmer'], flatten=True)
                tt_embs = emb_seq_w2v(tt_seqs, w2v_model, args['kmer'], flatten=True)
                clf = get_model('RF', tr_embs.shape[-1])
                clf.fit(tr_embs, tr_labels)
                each_mer_probs.append(clf.predict_proba(tt_embs)[:, 1])
            pred_probs = np.array(each_mer_probs).mean(0)
            preds = [1 if i > 0.5 else 0 for i in pred_probs]
            acc, mcc, sen, spe, auc = performance(preds, pred_probs, tt_labels)
            test_performance = {'Acc':acc, 'Mcc':mcc, 'Sens':sen, 'Spec':spe, 'AUC':auc}
            for k, v in test_performance.items():
                all_test_performance[k].append(v)
    for k, v in all_test_performance.items():
        print(f'{k} : {np.mean(v):.4f} ({np.std(v):.4f})')

def concat_w2v(data_file_name):
    seqs, labels = generate_features(data_file_name, 41)
    pad_seqs = pad_seq(seqs, 41)
    pad_seqs = np.array(pad_seqs)
    all_test_performance = {'Acc':[], 'Mcc':[], 'Sens':[], 'Spec':[], 'AUC':[]}
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(pad_seqs)):
            print(f'Time {time} Fold {k}')
            tr_seqs, tt_seqs = pad_seqs[tr_idx], pad_seqs[tt_idx]
            tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
            each_fold_train_embs = []
            each_fold_test_embs = []
            for i in range(1, 11):
                args = {
                    'w2v_path' : "./models/w2v_model/",
                    'emb_dim' : 128,
                    'kmer' : i,
                    'window' : 20,
                    'epochs' : 10,
                    'sg' : 1,
                    'force_train' : False
                }
                w2v_model = train_w2v(pad_seqs, **args)
                tr_embs = emb_seq_w2v(tr_seqs, w2v_model, args['kmer'], flatten=True)
                tt_embs = emb_seq_w2v(tt_seqs, w2v_model, args['kmer'], flatten=True)
                each_fold_train_embs.append(tr_embs)
                each_fold_test_embs.append(tt_embs)
            concat_train_embs = np.concatenate(each_fold_train_embs, axis=1)
            clf = get_model('RF', concat_train_embs.shape[-1])
            clf.fit(concat_train_embs, tr_labels)
            concat_test_embs = np.concatenate(each_fold_test_embs, axis=1)
            preds = clf.predict(concat_test_embs)
            pred_probs = clf.predict_proba(concat_test_embs)[:, 1]
            acc, mcc, sen, spe, auc = performance(preds, pred_probs, tt_labels)
            test_performance = {'Acc':acc, 'Mcc':mcc, 'Sens':sen, 'Spec':spe, 'AUC':auc}
            for k, v in test_performance.items():
                all_test_performance[k].append(v)
    for k, v in all_test_performance.items():
        print(f'{k} : {np.mean(v):.4f} ({np.std(v):.4f})')

def ensemble_probs_RF(embs_way, embs_list, labels, save_file_path, hc_fs = False):
    all_test_performance = {'Acc':[], 'Mcc':[], 'Sens':[], 'Spec':[], 'AUC':[]}
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(labels)):
            print(f'Time {time} Fold {k}')
            each_embs_probs = []
            for ei, embs in enumerate(embs_list):
                tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
                tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
                # if hc_fs, feature selection for hand-crafted features
                if hc_fs and embs_way[ei] == 'hand-crafted':
                    select = SelectKBest(r_regression, k=500)
                    tr_embs = select.fit_transform(tr_embs, tr_labels)
                    tt_embs = select.transform(tt_embs)
                clf = get_model('RF', tr_embs.shape[-1])
                clf.fit(tr_embs, tr_labels)
                each_embs_probs.append(clf.predict_proba(tt_embs)[:, 1])
            pred_probs = np.array(each_embs_probs).mean(0)
            preds = [1 if i > 0.5 else 0 for i in pred_probs]
            acc, mcc, sen, spe, auc = performance(preds, pred_probs, tt_labels)
            test_performance = {'Acc':acc, 'Mcc':mcc, 'Sens':sen, 'Spec':spe, 'AUC':auc}
            for k, v in test_performance.items():
                all_test_performance[k].append(v)
    record = {'Embed way': [' '.join(embs_way)]}
    for k in test_performance.keys():
        print(f'{k} : {np.mean(all_test_performance[k]):.4f} ({np.std(all_test_performance[k]):.4f})')
        record[k + ' mean'] = [np.mean(all_test_performance[k])]
        record[k + ' std'] = [np.std(all_test_performance[k])]
    record = pd.DataFrame(record)
    if save_file_path in os.listdir('results/'):
        record.to_csv(f'results/{save_file_path}', mode='a', header=False, index=False)
    else:
        record.to_csv(f'results/{save_file_path}', index=False)
    return record

def ensemble_vote_RF(embs_way, embs_list, labels, save_file_path):
    all_test_performance = {'Acc':[], 'Mcc':[], 'Sens':[], 'Spec':[], 'AUC':[]}
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(labels)):
            print(f'Time {time} Fold {k}')
            each_embs_preds = []
            for embs in embs_list:
                tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
                tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
                clf = get_model('RF', tr_embs.shape[-1])
                clf.fit(tr_embs, tr_labels)
                each_embs_preds.append(clf.predict(tt_embs))
            pred_probs = np.array(each_embs_preds).mean(0)
            preds = [1 if i > 0.5 else 0 for i in pred_probs]
            acc, mcc, sen, spe, auc = performance(preds, pred_probs, tt_labels)
            test_performance = {'Acc':acc, 'Mcc':mcc, 'Sens':sen, 'Spec':spe, 'AUC':auc}
            for k, v in test_performance.items():
                all_test_performance[k].append(v)
    record = {'Embed way': [' '.join(embs_way)]}
    for k in test_performance.keys():
        print(f'{k} : {np.mean(all_test_performance[k]):.4f} ({np.std(all_test_performance[k]):.4f})')
        record[k + ' mean'] = [np.mean(all_test_performance[k])]
        record[k + ' std'] = [np.std(all_test_performance[k])]
    record = pd.DataFrame(record)
    if save_file_path in os.listdir('results/'):
        record.to_csv(f'results/{save_file_path}', mode='a', header=False, index=False)
    else:
        record.to_csv(f'results/{save_file_path}', index=False)
    return record

def ensemble_probs_RF_in_val(embs_way, embs_list, labels, save_file_path, hc_fs = False):
    '''
    forward or backward selection according to validation set
    '''
    all_test_performance = {'Acc':[], 'Mcc':[], 'Sens':[], 'Spec':[], 'AUC':[]}
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(labels)):
            print(f'Time {time} Fold {k}')
            each_embs_probs = []
            for ei, embs in enumerate(embs_list):
                all_tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
                all_tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
                tr_embs, val_embs, tr_labels, val_labels = train_test_split(all_tr_embs, all_tr_labels, test_size = 0.2, random_state = 42)
                # if hc_fs, feature selection for hand-crafted features
                if hc_fs and embs_way[ei] == 'hand-crafted':
                    select = SelectKBest(r_regression, k=500)
                    tr_embs = select.fit_transform(tr_embs, tr_labels)
                    val_embs = select.transform(val_embs)
                clf = get_model('RF', tr_embs.shape[-1])
                clf.fit(tr_embs, tr_labels)
                each_embs_probs.append(clf.predict_proba(val_embs)[:, 1])
            pred_probs = np.array(each_embs_probs).mean(0)
            preds = [1 if i > 0.5 else 0 for i in pred_probs]
            acc, mcc, sen, spe, auc = performance(preds, pred_probs, val_labels)
            test_performance = {'Acc':acc, 'Mcc':mcc, 'Sens':sen, 'Spec':spe, 'AUC':auc}
            for k, v in test_performance.items():
                all_test_performance[k].append(v)
    record = {'Embed way': [' '.join(embs_way)]}
    for k in test_performance.keys():
        print(f'{k} : {np.mean(all_test_performance[k]):.4f} ({np.std(all_test_performance[k]):.4f})')
        record[k + ' mean'] = [np.mean(all_test_performance[k])]
        record[k + ' std'] = [np.std(all_test_performance[k])]
    record = pd.DataFrame(record)
    if save_file_path in os.listdir('results/'):
        record.to_csv(f'results/{save_file_path}', mode='a', header=False, index=False)
    else:
        record.to_csv(f'results/{save_file_path}', index=False)
    return record

def concat_embs_RF(embs_way, embs_list, labels, save_file_path):
    concat_embs = np.concatenate(embs_list, axis=1)
    all_test_performance = {'Acc':[], 'Mcc':[], 'Sens':[], 'Spec':[], 'AUC':[]}
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(labels)):
            print(f'Time {time} Fold {k}')
            tr_embs, tt_embs = concat_embs[tr_idx], concat_embs[tt_idx]
            tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
            clf = get_model('RF', tr_embs.shape[-1])
            clf.fit(tr_embs, tr_labels)
            preds = clf.predict(tt_embs)
            pred_probs = clf.predict_proba(tt_embs)[:, 1]
            acc, mcc, sen, spe, auc = performance(preds, pred_probs, tt_labels)
            test_performance = {'Acc':acc, 'Mcc':mcc, 'Sens':sen, 'Spec':spe, 'AUC':auc}
            for k, v in test_performance.items():
                all_test_performance[k].append(v)
    record = {'Embed way': [' '.join(embs_way)]}
    for k in test_performance.keys():
        print(f'{k} : {np.mean(all_test_performance[k]):.4f} ({np.std(all_test_performance[k]):.4f})')
        record[k + ' mean'] = [np.mean(all_test_performance[k])]
        record[k + ' std'] = [np.std(all_test_performance[k])]
    record = pd.DataFrame(record)
    if save_file_path in os.listdir('results/'):
        record.to_csv(f'results/{save_file_path}', mode='a', header=False, index=False)
    else:
        record.to_csv(f'results/{save_file_path}', index=False)

def ensemble_w2v_with_other(data_file_name):
    seqs, labels = generate_features(data_file_name, 41)
    pad_seqs = pad_seq(seqs, 41)
    each_embs_list = []
    # 1-10mer w2v
    each_mer_w2v_embs = []
    for i in range(1, 11):
        args = {
            'w2v_path' : "./models/w2v_model/",
            'emb_dim' : 128,
            'kmer' : i,
            'window' : 20,
            'epochs' : 10,
            'sg' : 1,
            'force_train' : False
        }
        w2v_model = train_w2v(pad_seqs, **args)
        each_mer_w2v_embs.append(emb_seq_w2v(pad_seqs, w2v_model, args['kmer'], flatten=True))
    each_mer_w2v_embs = np.concatenate(each_mer_w2v_embs, axis=1)
    esm2_embs = get_esm_embs(data_file_name, 41, '', device, True)
    prott5_embs = get_prott5_embs(data_file_name, 41, device, True)
    AAC_embs = AAC_DPC_TPC(seqs, 1)
    PAAC_embs = PseAAC(seqs, AAC_embs, 3, 0.05)
    APAAC_embs = Am_PseAAC(seqs, AAC_embs, 3, 0.05)
    CTD_embs = calculate_ctd_features(seqs)
    rdkit_embs = get_rdkit_embs(data_file_name, 41)
    # no.42 is too big, standard scaling it
    scaler = StandardScaler()
    rdkit_embs[:, 42] = scaler.fit_transform(rdkit_embs[:, 42].reshape(-1, 1)).flatten()
    # embs_ways = ['esm2', 'prott5', 'AAC', 'PAAC', 'APAAC', 'CTD', 'rdkit']
    # add_embs_list = [esm2_embs, prott5_embs, AAC_embs, PAAC_embs, APAAC_embs, CTD_embs, rdkit_embs]
    embs_ways = ['prott5', 'AAC', 'PAAC', 'APAAC', 'CTD', 'rdkit']
    add_embs_list = [prott5_embs, AAC_embs, PAAC_embs, APAAC_embs, CTD_embs, rdkit_embs]
    for i, ew in enumerate(embs_ways):
        _ = ensemble_probs_RF(['1-10mer w2v', ew], [each_mer_w2v_embs, esm2_embs, add_embs_list[i]], labels, 'ensemble_probs_with_1_10_mer_w2v.csv')
        # concat_embs_RF(['1-10mer w2v', ew], [each_mer_w2v_embs, add_embs_list[i]], labels, 'concat_with_1_10_mer_w2v.csv')

def SFS_with_w2v(data_file_name):
    # load embeddings
    seqs, labels = generate_features(data_file_name, 41)
    # LLM base part
    esm2_embs = get_esm_embs(data_file_name, 41, '', device, True)
    prott5_embs = get_prott5_embs(data_file_name, 41, device, True)
    protbert_embs = get_proteinbert_embs(data_file_name, 41, '', device, True)
    # Hand-crafted part (include Molecular) (472, 730)
    AAC_embs = AAC_DPC_TPC(seqs, 1)
    DPC_embs = AAC_DPC_TPC(seqs, 2)
    PAAC_embs = PseAAC(seqs, AAC_embs, 5, 0.05)
    CKS_embs = CKSAAGP(seqs, gap=2)
    CTD_embs = calculate_ctd_features(seqs)
    # Molecular part
    rdkit_embs = get_rdkit_embs(data_file_name, 41) # no.42 is too big, standard scaling it
    scaler = StandardScaler()
    rdkit_embs[:, 42] = scaler.fit_transform(rdkit_embs[:, 42].reshape(-1, 1)).flatten()
    # w2v base part
    pad_seqs = pad_seq(seqs, 41)
    w2v_embs = []
    for i in range(1, 11):
        args = {
            'w2v_path' : "./models/w2v_model/",
            'emb_dim' : 128,
            'kmer' : i,
            'window' : 20,
            'epochs' : 10,
            'sg' : 1,
            'force_train' : False
        }
        w2v_model = train_w2v(pad_seqs, **args)
        w2v_embs.append(emb_seq_w2v(pad_seqs, w2v_model, args['kmer'], flatten=True))
    w2v_embs = np.concatenate(w2v_embs, axis=1)
    other_embs_ways = ['CTD', 'rdkit', 'CKS', 'DPC', 'prott5', 'PAAC', 'AAC', 'esm2']
    other_embs_list = [CTD_embs, rdkit_embs, CKS_embs, DPC_embs, prott5_embs, PAAC_embs, AAC_embs, esm2_embs]
    selected_embs_ways = ['1-10mer w2v']
    selected_embs_list = [w2v_embs]
    best_acc = 0
    for i, oe in enumerate(other_embs_list):
        selected_embs_ways.append(other_embs_ways[i])
        selected_embs_list.append(oe)
        record = ensemble_probs_RF_in_val(selected_embs_ways, selected_embs_list, labels, 'SFS_cos_w2v_in_val.csv')
        cur_acc = record.at[0, "Acc mean"]
        if cur_acc > best_acc:
            best_acc = cur_acc
        else:
            print(f'Stop at {selected_embs_ways}')
            break
    # while len(other_embs_ways) > 0:
        # temp_record = []
        # for i, ew in enumerate(other_embs_ways):
            # temp_record.append(ensemble_probs_RF(selected_embs_ways + [ew], selected_embs_list + [other_embs_list[i]], labels, 'flatten_LLM_SFS_with_1-10mer_w2v.csv'))
            # temp_record.append(ensemble_vote_RF(selected_embs_ways + [ew], selected_embs_list + [other_embs_list[i]], labels, 'SFS_with_1-10mer_w2v_voted.csv'))
            # temp_record.append(ensemble_probs_RF_in_val(selected_embs_ways + [ew], selected_embs_list + [other_embs_list[i]], labels, 'SFS_with_1-10mer_w2v_in_val_10times.csv'))
        # record = pd.concat(temp_record)
        # last_max_acc = record.iloc[-(len(other_embs_ways)):]['Acc mean'].to_numpy().argmax()
        # if last_max_acc <= best_acc:
            # break
        # else:
            # best_acc = last_max_acc
        # selected_embs_ways.append(other_embs_ways.pop(last_max_acc))
        # selected_embs_list.append(other_embs_list.pop(last_max_acc))

def stacking(embs_way, embs_list, labels, save_file_path):
    all_test_performance = {'Acc':[], 'Mcc':[], 'Sens':[], 'Spec':[], 'AUC':[]}
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(labels)):
            print(f'Time {time} Fold {k}')
            tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
            for hs in range(10):
                print(f'---- Split {hs}')
                val_embs_probs, test_embs_probs = [], []
                for embs in embs_list:
                    # Spilt 80% into half
                    tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
                    half_train_embs, half_val_embs, half_train_labels, half_val_labels = train_test_split(tr_embs, tr_labels, test_size=0.1, random_state=hs)
                    for clf_model in ['RF', 'LGBM', 'XGB', 'EXT']:
                        print(f'-------- {clf_model}')
                        clf = get_model(clf_model, half_train_embs.shape[-1])
                        clf.fit(half_train_embs, half_train_labels)
                        val_embs_probs.append(clf.predict_proba(half_val_embs)[:, 1])
                        test_embs_probs.append(clf.predict_proba(tt_embs)[:, 1])
                val_pred_probs = np.array(val_embs_probs).transpose(1, 0)
                test_pred_probs = np.array(test_embs_probs).transpose(1, 0)
                # training meta model by val data
                meta_model = LogisticRegression()
                meta_model.fit(val_pred_probs, half_val_labels)
                preds = meta_model.predict(test_pred_probs)
                pred_probs = meta_model.predict_proba(test_pred_probs)[:, 1]
                acc, mcc, sen, spe, auc = performance(preds, pred_probs, tt_labels)
                test_performance = {'Acc':acc, 'Mcc':mcc, 'Sens':sen, 'Spec':spe, 'AUC':auc}
                for k, v in test_performance.items():
                    all_test_performance[k].append(v)
    record = {'Embed way': [' '.join(embs_way)]}
    for k in test_performance.keys():
        print(f'{k} : {np.mean(all_test_performance[k]):.4f} ({np.std(all_test_performance[k]):.4f})')
        record[k + ' mean'] = [np.mean(all_test_performance[k])]
        record[k + ' std'] = [np.std(all_test_performance[k])]
    record = pd.DataFrame(record)
    if save_file_path in os.listdir('results/'):
        record.to_csv(f'results/{save_file_path}', mode='a', header=False, index=False)
    else:
        record.to_csv(f'results/{save_file_path}', index=False)
    return record

def SFS_stacking_features_with_w2vs(data_file_name):
    '''
    Split data into 5 fold, and split 80% training part into half 10 times for training two parts in 
    stacking structure.
    '''
    # load embeddings
    seqs, labels = generate_features(data_file_name, 41)
    esm2_embs = get_esm_embs(data_file_name, 41, '', device, True)
    prott5_embs = get_prott5_embs(data_file_name, 41, device, True)
    AAC_embs = AAC_DPC_TPC(seqs, 1)
    PAAC_embs = PseAAC(seqs, AAC_embs, 3, 0.05)
    APAAC_embs = Am_PseAAC(seqs, AAC_embs, 3, 0.05)
    CTD_embs = calculate_ctd_features(seqs)
    rdkit_embs = get_rdkit_embs(data_file_name, 41)
    # no.42 is too big, standard scaling it
    scaler = StandardScaler()
    rdkit_embs[:, 42] = scaler.fit_transform(rdkit_embs[:, 42].reshape(-1, 1)).flatten()
    # load 1-10mer w2v
    pad_seqs = pad_seq(seqs, 41)
    w2v_embs = []
    for i in range(1, 11):
        args = {
            'w2v_path' : "./models/w2v_model/",
            'emb_dim' : 128,
            'kmer' : i,
            'window' : 20,
            'epochs' : 10,
            'sg' : 1,
            'force_train' : False
        }
        w2v_model = train_w2v(pad_seqs, **args)
        w2v_embs.append(emb_seq_w2v(pad_seqs, w2v_model, args['kmer'], flatten=True))
    w2v_embs = np.concatenate(w2v_embs, axis=1)
    # stacking(['1-10mer w2v', 'esm2', 'prott5', 'AAC', 'PAAC', 'APAAC', 'CTD', 'rdkit'], [w2v_embs, esm2_embs, prott5_embs, AAC_embs, PAAC_embs, APAAC_embs, CTD_embs, rdkit_embs], labels, 'only_all_7_3_stacking_with_1-10mer_w2v.csv')
    stacking(['1-10mer w2v', 'esm2', 'prott5', 'AAC', 'PAAC', 'APAAC', 'CTD', 'rdkit'], [w2v_embs, esm2_embs, prott5_embs, AAC_embs, PAAC_embs, APAAC_embs, CTD_embs, rdkit_embs], labels, 'only_all_9_1_stacking_with_1-10mer_w2v.csv')
    # # SFS
    # other_embs_ways = ['rdkit']
    # other_embs_list = [rdkit_embs]
    # selected_embs_ways = ['1-10mer w2v', 'esm2', 'prott5', 'AAC', 'PAAC', 'APAAC', 'CTD']
    # selected_embs_list = [w2v_embs, esm2_embs, prott5_embs, AAC_embs, PAAC_embs, APAAC_embs, CTD_embs]
    # # other_embs_ways = ['esm2', 'AAC', 'PAAC', 'APAAC', 'CTD', 'rdkit']
    # # other_embs_list = [esm2_embs, AAC_embs, PAAC_embs, APAAC_embs, CTD_embs, rdkit_embs]
    # # selected_embs_ways = ['1-10mer w2v', 'prott5']
    # # selected_embs_list = [w2v_embs, prott5_embs]
    # while len(other_embs_ways) > 0:
        # temp_record = []
        # for i, ew in enumerate(other_embs_ways):
            # # if selected_embs_ways + [ew] == ['1-10mer w2v', 'esm2']:
                # # print('Pass !!')
                # # continue
            # temp_record.append(stacking(selected_embs_ways + [ew], selected_embs_list + [other_embs_list[i]], labels, 'SFS_8_2_stacking_with_1-10mer_w2v.csv'))
        # record = pd.concat(temp_record)
        # last_max_acc = record.iloc[-(len(other_embs_ways)):]['Acc mean'].to_numpy().argmax()
        # selected_embs_ways.append(other_embs_ways.pop(last_max_acc))
        # selected_embs_list.append(other_embs_list.pop(last_max_acc))

def same_nature_combine(data_file_name):
    '''
    Combine features have similar nature.
    w2v | LLM | hand-crafted | molecular
    '''
    # load embeddings
    seqs, labels = generate_features(data_file_name, 41)
    # LLM base part
    avg_esm2_embs = get_esm_embs(data_file_name, 41, '', device, True)
    avg_prott5_embs = get_prott5_embs(data_file_name, 41, device, True)
    fla_esm2_embs = get_esm_embs(data_file_name, 41, 'seq', device, False)
    fla_esm2_embs = fla_esm2_embs.reshape(fla_esm2_embs.shape[0], -1)
    fla_prott5_embs = get_prott5_embs(data_file_name, 41, device, False) # flatten
    fla_prott5_embs = fla_prott5_embs.reshape(fla_prott5_embs.shape[0], -1)
    # llm_embs = np.concatenate([esm2_embs, prott5_embs], axis=1)
    # Hand-crafted part
    AAC_embs = AAC_DPC_TPC(seqs, 1)
    PAAC_embs = PseAAC(seqs, AAC_embs, 3, 0.05)
    APAAC_embs = Am_PseAAC(seqs, AAC_embs, 3, 0.05)
    CTD_embs = calculate_ctd_features(seqs)
    hc_embs = np.concatenate([AAC_embs, PAAC_embs, APAAC_embs, CTD_embs], axis=1)
    # Molecular part
    rdkit_embs = get_rdkit_embs(data_file_name, 41) # no.42 is too big, standard scaling it
    scaler = StandardScaler()
    rdkit_embs[:, 42] = scaler.fit_transform(rdkit_embs[:, 42].reshape(-1, 1)).flatten()
    # w2v base part
    pad_seqs = pad_seq(seqs, 41)
    w2v_embs = []
    for i in range(1, 11):
        args = {
            'w2v_path' : "./models/w2v_model/",
            'emb_dim' : 128,
            'kmer' : i,
            'window' : 20,
            'epochs' : 10,
            'sg' : 1,
            'force_train' : False
        }
        w2v_model = train_w2v(pad_seqs, **args)
        w2v_embs.append(emb_seq_w2v(pad_seqs, w2v_model, args['kmer'], flatten=True))
    w2v_embs = np.concatenate(w2v_embs, axis=1)
    w2v_hc_embs = np.concatenate([w2v_embs, AAC_embs, PAAC_embs, APAAC_embs, CTD_embs], axis=1)
    other_embs_ways = ['1-10mer w2v', 'avg_esm2', 'fla_esm2', 'avg_prott5', 'fla_prott5', 'AAC', 'PAAC', 'APAAC', 'CTD', 'rdkit']
    other_embs_list = [w2v_embs, avg_esm2_embs, fla_esm2_embs, avg_prott5_embs, fla_prott5_embs, AAC_embs, PAAC_embs, APAAC_embs, CTD_embs, rdkit_embs]
    selected_embs_ways = []
    selected_embs_list = []
    # other_embs_ways = ['LLM', 'hand-crafted', 'molecular']
    # other_embs_list = [llm_embs, hc_embs, rdkit_embs]
    # selected_embs_ways = ['1-10mer w2v']
    # selected_embs_list = [w2v_embs]
    # other_embs_ways = ['1-10mer w2v', 'LLM', 'hand-crafted', 'molecular']
    # other_embs_list = [w2v_embs, llm_embs, hc_embs, rdkit_embs]
    # selected_embs_ways = []
    # selected_embs_list = []
    while len(other_embs_ways) > 0:
        temp_record = []
        for i, ew in enumerate(other_embs_ways):
            temp_record.append(ensemble_probs_RF(selected_embs_ways + [ew], selected_embs_list + [other_embs_list[i]], labels, 'all_single_performance.csv'))
            # temp_record.append(ensemble_vote_RF(selected_embs_ways + [ew], selected_embs_list + [other_embs_list[i]], labels, 'SFS_same_nature_combine_voted.csv'))
            # temp_record.append(ensemble_probs_RF(selected_embs_ways + [ew], selected_embs_list + [other_embs_list[i]], labels, 'SFS_same_nature_combine_2.csv'))
        record = pd.concat(temp_record)
        last_max_acc = record.iloc[-(len(other_embs_ways)):]['Acc mean'].to_numpy().argmax()
        selected_embs_ways.append(other_embs_ways.pop(last_max_acc))
        selected_embs_list.append(other_embs_list.pop(last_max_acc))
        break

def binary_search(embs_list, embs_name, labels):
    combination = []
    for i in range(2**len(embs_list)):
        combination.append([(i >> j) & 1 for j in range(len(embs_list)-1, -1, -1)])
    for i in range(1, len(combination)):
        emb_ways = [embs_name[n] for n in range(len(combination[i])) if combination[i][n] == 1]
        selected_embs = [embs_list[n] for n in range(len(combination[i])) if combination[i][n] == 1]
        selected_embs = np.concatenate(selected_embs, axis=1)
        _ = ensemble_probs_RF(['+'.join(emb_ways)], [selected_embs], labels, 'combine_BS_' + '_'.join(embs_name) + '.csv', True)

def FS_each_combination_2n(data_file_name):
    # load embeddings
    seqs, labels = generate_features(data_file_name, 41)
    # LLM base part
    esm2_embs = get_esm_embs(data_file_name, 41, '', device, True)
    prott5_embs = get_prott5_embs(data_file_name, 41, device, True)
    protbert_embs = get_proteinbert_embs(data_file_name, 41, '', device, True)
    # fla_esm2_embs = get_esm_embs(data_file_name, 41, 'seq', device, False)
    # fla_esm2_embs = fla_esm2_embs.reshape(fla_esm2_embs.shape[0], -1)
    # fla_prott5_embs = get_prott5_embs(data_file_name, 41, device, False)
    # fla_prott5_embs = fla_prott5_embs.reshape(fla_prott5_embs.shape[0], -1)
    # fla_protbert_embs = get_proteinbert_embs(data_file_name, 41, 'seq', device, False)
    # fla_protbert_embs = fla_protbert_embs.reshape(fla_protbert_embs.shape[0], -1)
    # cls_esm2_embs = get_esm_embs(data_file_name, 41, 'cls', device, True)
    # cls_protbert_embs = get_proteinbert_embs(data_file_name, 41, 'cls', device, True)
    # binary_search([esm2_embs, prott5_embs, protbert_embs], ['esm2', 'prott5', 'protbert'], labels)
    # binary_search([fla_esm2_embs, fla_prott5_embs, fla_protbert_embs], ['fla_esm2', 'fla_prott5', 'fla_protbert'], labels)
    # binary_search([cls_esm2_embs, cls_protbert_embs], ['cls_esm2', 'cls_protbert'], labels)
    # from 'BS_esm2_prott5_protbert.csv' we can find out concat three model is better than any combination
    llm_embs = np.concatenate([esm2_embs, prott5_embs, protbert_embs], axis=1)
    # Hand-crafted part (include Molecular) (472, 730)
    AAC_embs = AAC_DPC_TPC(seqs, 1)
    DPC_embs = AAC_DPC_TPC(seqs, 2)
    PAAC_embs = PseAAC(seqs, AAC_embs, 5, 0.05)
    CKS_embs = CKSAAGP(seqs, gap=2)
    # Molecular part
    rdkit_embs = get_rdkit_embs(data_file_name, 41) # no.42 is too big, standard scaling it
    scaler = StandardScaler()
    rdkit_embs[:, 42] = scaler.fit_transform(rdkit_embs[:, 42].reshape(-1, 1)).flatten()
    hc_embs = np.concatenate([AAC_embs, DPC_embs, PAAC_embs, CKS_embs, rdkit_embs], axis=1)
    all_other_embs = np.concatenate([llm_embs, hc_embs], axis=1)
    # w2v base part
    pad_seqs = pad_seq(seqs, 41)
    w2v_embs = []
    for i in range(1, 11):
        args = {
            'w2v_path' : "./models/w2v_model/",
            'emb_dim' : 128,
            'kmer' : i,
            'window' : 20,
            'epochs' : 10,
            'sg' : 1,
            'force_train' : False
        }
        w2v_model = train_w2v(pad_seqs, **args)
        w2v_embs.append(emb_seq_w2v(pad_seqs, w2v_model, args['kmer'], flatten=True))
    w2v_embs = np.concatenate(w2v_embs, axis=1)
    # binary_search([w2v_embs, llm_embs, hc_embs], ['1-10mer w2v', 'LLMs', 'hand-crafted'], labels)
    binary_search([w2v_embs, all_other_embs], ['1-10mer w2v', 'LLMs and hand-crafted'], labels)

def predict_correlation(data_file_name):
    # load embeddings
    seqs, labels = generate_features(data_file_name, 41)
    # LLM base part
    esm2_embs = get_esm_embs(data_file_name, 41, '', device, True)
    prott5_embs = get_prott5_embs(data_file_name, 41, device, True)
    protbert_embs = get_proteinbert_embs(data_file_name, 41, '', device, True)
    # Hand-crafted part (include Molecular) (472, 730)
    AAC_embs = AAC_DPC_TPC(seqs, 1)
    DPC_embs = AAC_DPC_TPC(seqs, 2)
    PAAC_embs = PseAAC(seqs, AAC_embs, 5, 0.05)
    CKS_embs = CKSAAGP(seqs, gap=2)
    CTD_embs = calculate_ctd_features(seqs)
    # Molecular part
    rdkit_embs = get_rdkit_embs(data_file_name, 41) # no.42 is too big, standard scaling it
    scaler = StandardScaler()
    rdkit_embs[:, 42] = scaler.fit_transform(rdkit_embs[:, 42].reshape(-1, 1)).flatten()
    # w2v base part
    pad_seqs = pad_seq(seqs, 41)
    each_w2v_embs = []
    for i in range(1, 11):
        args = {
            'w2v_path' : "./models/w2v_model/",
            'emb_dim' : 128,
            'kmer' : i,
            'window' : 20,
            'epochs' : 10,
            'sg' : 1,
            'force_train' : False
        }
        w2v_model = train_w2v(pad_seqs, **args)
        each_w2v_embs.append(emb_seq_w2v(pad_seqs, w2v_model, args['kmer'], flatten=True))
    w2v_embs = np.concatenate(each_w2v_embs, axis=1)
    # all_embs_list =  [esm2_embs, prott5_embs, protbert_embs, AAC_embs, DPC_embs, PAAC_embs, CKS_embs, CTD_embs, rdkit_embs, w2v_embs]
    # all_embs_way = ['esm2', 'prott5', 'protbert', 'AAC', 'DPC', 'PAAC', 'CKS', 'CTD', 'rdkit', '1-10mer_w2v']
    all_embs_list = each_w2v_embs
    all_embs_way = [f'{i}mer w2v' for i in range(1, 11)]
    # get all embeddings' predict probability
    cos_matrix = []
    for time in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=time)
        for k, (tr_idx, tt_idx) in enumerate(kf.split(labels)):
            print(f'Time {time} Fold {k}')
            each_embs_probs = []
            for ei, embs in enumerate(all_embs_list):
                all_tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
                all_tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
                tr_embs, val_embs, tr_labels, val_labels = train_test_split(all_tr_embs, all_tr_labels, test_size = 0.2, random_state = 42)
                clf = get_model('RF', tr_embs.shape[-1])
                clf.fit(tr_embs, tr_labels)
                each_embs_probs.append(clf.predict_proba(val_embs)[:, 1])
            each_embs_probs = np.stack(each_embs_probs)
            cos_matrix.append(pd.DataFrame(cosine_similarity(each_embs_probs), index=all_embs_way, columns=all_embs_way))
    cos_matrix = sum(cos_matrix) / len(cos_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cos_matrix, annot=True, cmap="coolwarm", fmt=".4f")
    plt.title("Cosine Similarity of Model Predictions")
    plt.savefig("./cosine_similarity_predictions_w2v.png", dpi=300, bbox_inches='tight')
    cos_matrix.to_csv(f'results/cosine_similarity_predictions_w2v.csv')

def random_w2v_test(data_file_name):
    # load embeddings
    seqs, labels = generate_features(data_file_name, 41)
    # w2v base part
    pad_seqs = pad_seq(seqs, 41)
    w2v_embs = []
    for i in range(1, 11):
        args = {
            'w2v_path' : "./models/w2v_model/",
            'emb_dim' : 128,
            'kmer' : i,
            'window' : 20,
            'epochs' : 10,
            'sg' : 1,
            'force_train' : False
        }
        w2v_model = train_w2v(pad_seqs, **args)
        w2v_embs.append(emb_seq_w2v(pad_seqs, w2v_model, args['kmer'], flatten=True))
    some_combination = [[[0, 1, 2], [3, 4, 5, 6, 7, 8], [9]]]
    # for i, w in enumerate(w2v_embs):
        # record = ensemble_probs_RF([f'{i+1}mer w2v'], [w], labels, 'all_mer_w2v.csv')
    for sc in some_combination:
        embs_way = []
        embs_list = []
        for c in sc:
            embs_way.append('+'.join([f'{i+1}' for i in c]) + 'mer w2v')
            each_embs_list = []
            for i in c:
                each_embs_list.append(w2v_embs[i])
            embs_list.append(np.concatenate(each_embs_list, axis = 1))
        record = ensemble_probs_RF(embs_way, embs_list, labels, 'test_some_combination_w2v.csv')

def concat_similar_w2v(data_file_name):
    '''
    1~10 mer w2v, in while loop, take top 2 similarity mer out and concat, and keep going until 
    conforming some condition. 
    '''
    curr_w2v_embs, curr_w2v_mer_list, labels = get_embs_set(data_file_name, device)
    curr_w2v_embs = curr_w2v_embs[:10]
    curr_w2v_mer_list = curr_w2v_mer_list[:10]
    all_probs = []
    while len(curr_w2v_embs) > 1:
        print(curr_w2v_mer_list)
        cos_matrix = []
        entropy_list = []
        corr_dict = {'pearson':[], 'spearman':[], 'kendall':[], 'cosine':[], 'euclidean':[], 'manhattan':[]}
        for time in range(10):
            kf = KFold(n_splits=5, shuffle=True, random_state=time)
            for k, (tr_idx, tt_idx) in enumerate(kf.split(labels)):
                print(f'Time {time} Fold {k}')
                each_embs_probs = []
                for ei, embs in enumerate(curr_w2v_embs):
                    all_tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
                    all_tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
                    tr_embs, val_embs, tr_labels, val_labels = train_test_split(all_tr_embs, all_tr_labels, test_size = 0.2, random_state = 42)
                    print(tr_embs.shape)
                    print(tr_labels.shape)
                    print(dkdkkdkdkdk)
                    clf = get_model('RF', tr_embs.shape[-1])
                    clf.fit(tr_embs, tr_labels)
                    each_embs_probs.append(clf.predict_proba(val_embs)[:, 1])
                each_embs_probs = np.stack(each_embs_probs)
                all_probs.append(each_embs_probs)
                entropy_list.append(cal_entropy(each_embs_probs))
                for cor_key, cor_values in cal_correlation(each_embs_probs).items():
                    corr_dict[cor_key].append(cor_values)
                cos_matrix.append(pd.DataFrame(cosine_similarity(each_embs_probs), index=curr_w2v_mer_list, columns=curr_w2v_mer_list))
        stats = {f"{key} mean": pd.Series(values).mean() for key, values in corr_dict.items()}
        stats.update({f"{key} std": pd.Series(values).std(ddof=0) for key, values in corr_dict.items()})
        corr_df = pd.DataFrame([stats])
        corr_df.to_csv(f'results/each_pred_corr_of_split_w2v.csv', mode='a', index=False, header=False)
        cos_matrix = sum(cos_matrix) / len(cos_matrix)
        # cos_matrix.to_csv(f'results/split_by_cos_sim_len{len(curr_w2v_embs)}.csv')
        # entropy_df = pd.DataFrame([{
            # 'w2vs': '/'.join(curr_w2v_mer_list), 
            # 'entropy mean': np.mean(entropy_list),
            # 'entropy std': np.std(entropy_list),
            # }])
        # entropy_df.to_csv(f'results/entropy_of_split_w2v.csv', mode='a', index=False, header=False)
        mean_sim = cos_matrix.mean()
        # _ = ensemble_probs_RF(curr_w2v_mer_list, curr_w2v_embs, labels, 'split_by_cos_sim.csv')
        # pick top 2 high and concat
        top2_sim_idx = np.argsort(mean_sim)[-2:][::-1]
        temp_embs_list, temp_way_list = [], []
        for i in top2_sim_idx:
            temp_embs_list.append(curr_w2v_embs.pop(i))
            temp_way_list.append(curr_w2v_mer_list.pop(i))
        curr_w2v_embs.append(np.concatenate(temp_embs_list, axis=1))
        curr_w2v_mer_list.append('+'.join(temp_way_list))
    save_txt('results/probs_txt/split_by_cos_sim_w2v.txt', all_probs)

def heatmap_csv():
    for l in range(2, 20):
        df = pd.read_csv(f'results/cos_sim/all_split_by_cos_sim_len{l}.csv', index_col = 0)
        plt.figure(figsize=(8, 6))
        sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".4f")
        plt.title("Cosine Similarity of Model Predictions")
        plt.savefig(f"./imgs/all_cos_sim_and_concat_len{l}_w2v.png", dpi=300, bbox_inches='tight')
        mask = ~np.eye(len(df), dtype=bool)
        mean_values = (df.where(mask).sum() / (l - 1)).to_frame().T
        mean_values.index = ['mean']
        df = pd.concat([df, mean_values])
        df.to_csv(f'results/cos_sim/all_mean_split_by_cos_sim_len{l}.csv')
        print(l)
        print(df.loc['mean'].mean())

def concat_similar_all(data_file_name):
    curr_w2v_embs, curr_w2v_mer_list, labels = get_embs_set(data_file_name, device)
    all_probs = []
    while len(curr_w2v_embs) > 1:
        print(curr_w2v_mer_list)
        cos_matrix = []
        entropy_list = []
        corr_dict = {'pearson':[], 'spearman':[], 'kendall':[], 'cosine':[], 'euclidean':[], 'manhattan':[]}
        for time in range(10):
            kf = KFold(n_splits=5, shuffle=True, random_state=time)
            for k, (tr_idx, tt_idx) in enumerate(kf.split(labels)):
                print(f'Time {time} Fold {k}')
                each_embs_probs = []
                for ei, embs in enumerate(curr_w2v_embs):
                    all_tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
                    all_tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
                    tr_embs, val_embs, tr_labels, val_labels = train_test_split(all_tr_embs, all_tr_labels, test_size = 0.2, random_state = 42)
                    clf = get_model('RF', tr_embs.shape[-1])
                    clf.fit(tr_embs, tr_labels)
                    each_embs_probs.append(clf.predict_proba(val_embs)[:, 1])
                each_embs_probs = np.stack(each_embs_probs)
                all_probs.append(each_embs_probs)
                entropy_list.append(cal_entropy(each_embs_probs))
                for cor_key, cor_values in cal_correlation(each_embs_probs).items():
                    corr_dict[cor_key].append(cor_values)
                cos_matrix.append(pd.DataFrame(cosine_similarity(each_embs_probs), index=curr_w2v_mer_list, columns=curr_w2v_mer_list))
        stats = {f"{key} mean": pd.Series(values).mean() for key, values in corr_dict.items()}
        stats.update({f"{key} std": pd.Series(values).std(ddof=0) for key, values in corr_dict.items()})
        corr_df = pd.DataFrame([stats])
        corr_df.to_csv(f'results/each_pred_corr_of_split_all.csv', mode='a', index=False, header=False)
        cos_matrix = sum(cos_matrix) / len(cos_matrix)
        # cos_matrix.to_csv(f'results/cos_sim/all_split_by_cos_sim_len{len(curr_w2v_embs)}.csv')
        # entropy_df = pd.DataFrame([{
            # 'w2vs': '/'.join(curr_w2v_mer_list), 
            # 'entropy mean': np.mean(entropy_list),
            # 'entropy std': np.std(entropy_list),
            # }])
        # entropy_df.to_csv(f'results/entropy_of_split_all.csv', mode='a', index=False, header=False)
        mean_sim = cos_matrix.mean()
        # _ = ensemble_probs_RF(curr_w2v_mer_list, curr_w2v_embs, labels, 'all_split_by_cos_sim.csv')
        # pick top 2 high and concat
        top2_sim_idx = np.argsort(mean_sim)[-2:][::-1]
        temp_embs_list, temp_way_list = [], []
        for i in top2_sim_idx:
            temp_embs_list.append(curr_w2v_embs.pop(i))
            temp_way_list.append(curr_w2v_mer_list.pop(i))
        curr_w2v_embs.append(np.concatenate(temp_embs_list, axis=1))
        curr_w2v_mer_list.append('+'.join(temp_way_list))
    save_txt('results/probs_txt/all_split_by_cos_sim.txt', all_probs)

def backward_split(data_file_name):
    w2v_embs, w2v_mer_list, labels = get_embs_set(data_file_name, device)
    # only w2v
    # w2v_embs = w2v_embs[:10]
    # w2v_mer_list = w2v_mer_list[:10]
    w2v_embs = w2v_embs
    w2v_mer_list = w2v_mer_list
    all_probs = []
    selected_embs, selected_mer_list = [], []
    # df = pd.read_csv(f'results/cos_sim/split_by_cos_sim_len10.csv', index_col = 0)
    df = pd.read_csv(f'results/cos_sim/all_split_by_cos_sim_len19.csv', index_col = 0)
    mask = ~np.eye(len(df), dtype=bool)
    mean_values = (df.where(mask).sum() / (len(w2v_mer_list)-1)).to_frame().T.to_numpy().squeeze()
    # record_probs = load_txt('results/probs_txt/backward_split_by_cos_sim_w2v.txt')
    record_probs = load_txt('results/probs_txt/backward_split_by_cos_sim_all.txt')
    while len(w2v_mer_list) > 1:
        # pop least similar feature into selected list
        selected_mer_list.append(w2v_mer_list.pop(np.argmin(mean_values)))
        selected_embs.append(w2v_embs.pop(np.argmin(mean_values)))
        mean_values = np.delete(mean_values, np.argmin(mean_values))
        print(' '.join(selected_mer_list) + ' ' + '+'.join(w2v_mer_list))
        curr_w2v_embs = selected_embs + [np.concatenate(w2v_embs, axis=1)]
        curr_w2v_mer_list = selected_mer_list + ['+'.join(w2v_mer_list)]
        cos_matrix = []
        entropy_list, std_list = [], []
        corr_dict = {'pearson':[], 'spearman':[], 'kendall':[], 'cosine':[], 'euclidean':[], 'manhattan':[]}
        for time in range(10):
            kf = KFold(n_splits=5, shuffle=True, random_state=time)
            for k, (tr_idx, tt_idx) in enumerate(kf.split(labels)):
                print(f'Time {time} Fold {k}')
                # each_embs_probs = []
                # for ei, embs in enumerate(curr_w2v_embs):
                    # all_tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
                    # all_tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
                    # tr_embs, val_embs, tr_labels, val_labels = train_test_split(all_tr_embs, all_tr_labels, test_size = 0.2, random_state = 42)
                    # clf = get_model('RF', tr_embs.shape[-1])
                    # clf.fit(tr_embs, tr_labels)
                    # each_embs_probs.append(clf.predict_proba(val_embs)[:, 1])
                # each_embs_probs = np.stack(each_embs_probs)
                # all_probs.append(each_embs_probs)
                each_embs_probs = record_probs.pop(0)
                entropy_list.append(cal_entropy(each_embs_probs))
                std_list.append(cal_each_model_std(each_embs_probs))
                for cor_key, cor_values in cal_correlation(each_embs_probs).items():
                    corr_dict[cor_key].append(cor_values)
                cos_matrix.append(pd.DataFrame(cosine_similarity(each_embs_probs), index=curr_w2v_mer_list, columns=curr_w2v_mer_list))
        stats = {f"{key} mean": pd.Series(values).mean() for key, values in corr_dict.items()}
        stats.update({f"{key} std": pd.Series(values).std(ddof=0) for key, values in corr_dict.items()})
        corr_df = pd.DataFrame([stats])
        # corr_df.to_csv(f'results/each_pred_corr_of_split_w2v_backward.csv', mode='a', index=False, header=False)
        corr_df.to_csv(f'results/each_pred_corr_of_split_all_backward.csv', mode='a', index=False)
        # cos_matrix = sum(cos_matrix) / len(cos_matrix)
        # cos_matrix.to_csv(f'results/cos_sim/backward_split_by_cos_sim_len{len(curr_w2v_embs)}.csv')
        # entropy_df = pd.DataFrame([{
            # 'w2vs': '/'.join(curr_w2v_mer_list), 
            # 'entropy mean': np.mean(entropy_list),
            # 'entropy std': np.std(entropy_list),
            # }])
        # entropy_df.to_csv(f'results/entropy_of_split_w2v_backward.csv', mode='a', index=False, header=False)
        # std_df = pd.DataFrame([{
            # 'w2vs': '/'.join(curr_w2v_mer_list), 
            # 'std mean': np.mean(std_list),
            # 'std std': np.std(std_list),
            # }])
        # std_df.to_csv(f'results/std_of_split_all_backward.csv', mode='a', index=False, header=False)
        # _ = ensemble_probs_RF(curr_w2v_mer_list, curr_w2v_embs, labels, 'backward_split_by_cos_sim.csv')
    # save_txt('results/probs_txt/backward_split_by_cos_sim_w2v.txt', all_probs)

def backward_split_in_group(data_file_name):
    embs, emb_ways, labels = get_embs_set(data_file_name, device)
    # only w2v
    all_embs = embs[:10]
    all_emb_ways = emb_ways[:10]
    # all_embs = embs
    # all_emb_ways = emb_ways
    all_probs = []
    selected_embs, selected_ways = [], []
    while len(all_emb_ways) > 1:
        # select feature that has least similarity with features in 'selected_embs' and the concat group except it
        least_sim = 99
        for i in range(len(all_emb_ways)):
            candidate_embs = all_embs[i]
            group_embs = [e for ei, e in enumerate(all_embs) if ei != i]
            curr_embs = [candidate_embs] + selected_embs + [np.concatenate(group_embs, axis=1)]
            curr_emb_ways = [all_emb_ways[i]] + selected_ways + ['+'.join([e for ei, e in enumerate(all_emb_ways) if ei != i])]
            cos_matrix = []
            entropy_list, std_list = [], []
            corr_dict = {'pearson':[], 'spearman':[], 'kendall':[], 'cosine':[], 'euclidean':[], 'manhattan':[]}
            for time in range(10):
                kf = KFold(n_splits=5, shuffle=True, random_state=time)
                for k, (tr_idx, tt_idx) in enumerate(kf.split(labels)):
                    if f'each_embs_probs_len{len(all_emb_ways)}_candidate{i}_time{time}_fold{k}.txt' not in os.listdir('results/probs_txt/backward_in_group/'):
                    # if f'all_each_embs_probs_len{len(all_emb_ways)}_candidate{i}_time{time}_fold{k}.txt' not in os.listdir('results/probs_txt/backward_in_group/'):
                        print(f'Time {time} Fold {k}')
                        each_embs_probs = []
                        for c_embs in curr_embs:
                            all_tr_embs, tt_embs = c_embs[tr_idx], c_embs[tt_idx]
                            all_tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
                            tr_embs, val_embs, tr_labels, val_labels = train_test_split(all_tr_embs, all_tr_labels, test_size = 0.2, random_state = 42)
                            clf = get_model('RF', tr_embs.shape[-1])
                            clf.fit(tr_embs, tr_labels)
                            each_embs_probs.append(clf.predict_proba(val_embs)[:, 1])
                        each_embs_probs = np.stack(each_embs_probs)
                        # save_txt(f'results/probs_txt/backward_in_group/each_embs_probs_len{len(all_emb_ways)}_candidate{i}_time{time}_fold{k}.txt', each_embs_probs)
                        # save_txt(f'results/probs_txt/backward_in_group/all_each_embs_probs_len{len(all_emb_ways)}_candidate{i}_time{time}_fold{k}.txt', each_embs_probs)
                    else:
                        # if want load record
                        print('Has inferenced !!!')
                        each_embs_probs = load_txt(f'results/probs_txt/backward_in_group/each_embs_probs_len{len(all_emb_ways)}_candidate{i}_time{time}_fold{k}.txt')
                        # each_embs_probs = load_txt(f'results/probs_txt/backward_in_group/all_each_embs_probs_len{len(all_emb_ways)}_candidate{i}_time{time}_fold{k}.txt')
                    entropy_list.append(cal_entropy(each_embs_probs))
                    std_list.append(cal_each_model_std(each_embs_probs))
                    for cor_key, cor_values in cal_correlation(each_embs_probs).items():
                        corr_dict[cor_key].append(cor_values)
                    cos_matrix.append(pd.DataFrame(cosine_similarity(each_embs_probs), index=curr_emb_ways, columns=curr_emb_ways))
            cos_matrix = sum(cos_matrix) / len(cos_matrix)
            if np.mean(cos_matrix.iloc[0]) < least_sim:
                least_sim = np.mean(cos_matrix.iloc[0])
                least_sim_corr_dict = corr_dict
                least_sim_entropy_list = entropy_list
                least_sim_cos_matrix = cos_matrix
                least_sim_index = i
        selected_embs.append(all_embs.pop(least_sim_index))
        selected_ways.append(all_emb_ways.pop(least_sim_index))
        stats = {f"{key} mean": pd.Series(values).mean() for key, values in least_sim_corr_dict.items()}
        stats.update({f"{key} std": pd.Series(values).std(ddof=0) for key, values in least_sim_corr_dict.items()})
        corr_df = pd.DataFrame([stats])
        corr_df.to_csv(f'results/each_pred_corr_of_split_w2v_backward_in_group.csv', mode='a', index=False, header=False)
        # corr_df.to_csv(f'results/each_pred_corr_of_split_all_backward_in_group.csv', mode='a', index=False, header=False)
        # least_sim_cos_matrix.to_csv(f'results/cos_sim/backward_in_group_split_by_cos_sim_len{len(selected_ways) + 1}.csv')
        # least_sim_cos_matrix.to_csv(f'results/cos_sim/backward_in_group_all_split_by_cos_sim_len{len(selected_ways) + 1}.csv')
        # entropy_df = pd.DataFrame([{
            # 'w2vs': '/'.join(selected_ways + ['+'.join(all_emb_ways)]), 
            # 'entropy mean': np.mean(least_sim_entropy_list),
            # 'entropy std': np.std(least_sim_entropy_list),
            # }])
        # entropy_df.to_csv(f'results/entropy_of_split_backward_in_group.csv', mode='a', index=False, header=False)
        # entropy_df.to_csv(f'results/entropy_of_split_all_backward_in_group.csv', mode='a', index=False, header=False)
        # std_df = pd.DataFrame([{
            # 'w2vs': '/'.join(selected_ways + ['+'.join(all_emb_ways)]), 
            # 'std mean': np.mean(std_list),
            # 'std std': np.std(std_list),
            # }])
        # std_df.to_csv(f'results/std_of_split_w2v_backward_in_group.csv', mode='a', index=False, header=False)
        # std_df.to_csv(f'results/std_of_split_all_backward_in_group.csv', mode='a', index=False, header=False)
        # _ = ensemble_probs_RF(selected_ways + ['+'.join(all_emb_ways)], selected_embs + [np.concatenate(all_embs, axis=1)], labels, 'backward_in_group_split_by_cos_sim.csv')
        # _ = ensemble_probs_RF(selected_ways + ['+'.join(all_emb_ways)], selected_embs + [np.concatenate(all_embs, axis=1)], labels, 'backward_in_group_all_split_by_cos_sim.csv')
   
def BS_only_w2v(data_file_name):
    embs, emb_ways, labels = get_embs_set(data_file_name, device)
    # only w2v
    # all_embs = embs[:10]
    # all_emb_ways = emb_ways[:10]
    all_embs = embs
    all_emb_ways = emb_ways
    # backward selection
    best_embs = all_embs
    best_emb_ways = all_emb_ways
    all_max_acc = 0
    while len(best_emb_ways) > 1:
        print(best_emb_ways)
        temp_record = []
        for drop_i in range(len(best_emb_ways)):
            curr_embs = [best_embs[i] for i in range(len(best_emb_ways)) if i != drop_i]
            curr_emb_ways = [best_emb_ways[i] for i in range(len(best_emb_ways)) if i != drop_i]
            # temp_record.append(ensemble_probs_RF_in_val(curr_emb_ways, curr_embs, labels, 'SBS_w2vs.csv'))
            temp_record.append(ensemble_probs_RF_in_val(curr_emb_ways, curr_embs, labels, 'SBS_all.csv'))
        record = pd.concat(temp_record)
        max_acc_id = record['Acc mean'].to_numpy().argmax()
        max_acc = record['Acc mean'].to_numpy().max()
        if max_acc < all_max_acc:
            print(f'End at {best_emb_ways} !!!')
            _ = ensemble_probs_RF(best_emb_ways, best_embs, labels, 'BS_all.csv')
            break
        else:
            all_max_acc = max_acc
            best_embs.pop(max_acc_id)
            best_emb_ways.pop(max_acc_id)

def FS_only_w2v(data_file_name):
    embs, emb_ways, labels = get_embs_set(data_file_name, device)
    # only w2v
    # all_embs = embs[:10]
    # all_emb_ways = emb_ways[:10]
    all_embs = embs
    all_emb_ways = emb_ways
    # backward selection
    selected_embs, selected_emb_ways = [], []
    all_max_acc = 0
    while len(all_emb_ways) > 1:
        print(selected_emb_ways)
        temp_record = []
        for add_i in range(len(all_embs)):
            curr_embs = selected_embs + [all_embs[add_i]]
            curr_emb_ways = selected_emb_ways + [all_emb_ways[add_i]]
            # temp_record.append(ensemble_probs_RF(curr_emb_ways, curr_embs, labels, 'SFS_watch_test_label_w2vs.csv'))
            temp_record.append(ensemble_probs_RF_in_val(curr_emb_ways, curr_embs, labels, 'SFS_watch_test_label_all.csv'))
        record = pd.concat(temp_record)
        max_acc_id = record['Acc mean'].to_numpy().argmax()
        max_acc = record['Acc mean'].to_numpy().max()
        if max_acc < all_max_acc:
            print(f'End at {selected_emb_ways} !!!')
            break
        else:
            all_max_acc = max_acc
            selected_embs.append(all_embs.pop(max_acc_id))
            selected_emb_ways.append(all_emb_ways.pop(max_acc_id))

def fast_ensemble_w2v(data_file_name):
    embs, emb_ways, labels = get_embs_set(data_file_name, device)
    # only w2v
    all_embs = embs[:10]
    all_emb_ways = emb_ways[:10]
    # all_embs = embs
    # all_emb_ways = emb_ways
    # ['2mer', '5mer', '9mer', '10mer']
    # FS_embs = [all_embs[1], all_embs[4], all_embs[8], all_embs[9]]
    # FS_emb_ways = [all_emb_ways[1], all_emb_ways[4], all_emb_ways[8], all_emb_ways[9]]
    # ['9mer', 'prott5', '10mer', '5mer', 'DPC', 'rdkit', 'CTD', '6mer']
    # FS_embs = [all_embs[4], all_embs[5], all_embs[8], all_embs[9], all_embs[11], all_embs[14], all_embs[17], all_embs[18]]
    # FS_emb_ways = [all_emb_ways[4], all_emb_ways[5], all_emb_ways[8], all_emb_ways[9], all_emb_ways[11], all_emb_ways[14], all_emb_ways[17], all_emb_ways[18]]
    FS_embs = [all_embs[0], all_embs[1], all_embs[4], all_embs[8], all_embs[9]]
    FS_emb_ways = [all_emb_ways[0], all_emb_ways[1], all_emb_ways[4], all_emb_ways[8], all_emb_ways[9]]
    _ = ensemble_probs_RF(FS_emb_ways, FS_embs, labels, 'FS_watch_test_label_w2vs.csv')

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
    # esm2_CNN(data_file_name, device)
    # esm2_RNN(data_file_name, device)
    # esm2_LSTM(data_file_name, device)
    # esm2_hand_RF(data_file_name, device)
    # multipeptide_on_LLM(device)
    # try_stack(data_file_name, device)
    # try_stack2()
    # print(load_txt('results/probs/esm2_LGBM_train_probs.pkl'))
    # test_w2v(data_file_name, device)
    # ensemble_w2v(data_file_name)
    # concat_w2v(data_file_name)
    # ensemble_w2v_with_other(data_file_name)
    # SFS_with_w2v(data_file_name)
    # SFS_stacking_features_with_w2vs(data_file_name)
    # same_nature_combine(data_file_name)
    # FS_each_combination_2n(data_file_name)
    # predict_correlation(data_file_name)
    # random_w2v_test(data_file_name)
    # concat_similar_w2v(data_file_name)
    # heatmap_csv()
    # concat_similar_all(data_file_name)
    # backward_split(data_file_name)
    # backward_split_in_group(data_file_name)
    # BS_only_w2v(data_file_name)
    FS_only_w2v(data_file_name)
    # fast_ensemble_w2v(data_file_name)
