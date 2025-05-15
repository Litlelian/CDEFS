import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier, RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import r_regression, SelectKBest
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from utils import *
from feature import *
from model import *

def plain_ert_rf_gridsearch(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(
        train_data_file_name, test_data_file_name, pad_size, device, dataset_name
    )
    train_embs = [tr_embs[19], tr_embs[14], tr_embs[18], tr_embs[21], tr_embs[20]]
    test_embs = [tt_embs[19], tt_embs[14], tt_embs[18], tt_embs[21], tt_embs[20]]

    X_train = np.concatenate(train_embs, axis=1)
    X_test = np.concatenate(test_embs, axis=1)
    tr_labels = np.array(tr_labels)
    tt_labels = np.array(tt_labels)

    param_space = [
        {'n_estimators': ne, 'max_depth': md, 'max_features': mf}
        for ne, md, mf in product([100, 200, 300, 400, 500, 600, 700, 800 ,900, 1000], [None, 10, 20], ['sqrt', 'log2'])
    ]

    all_results = []

    for i, params in enumerate(param_space):
        for model_name, Model in [('ERT', ExtraTreesClassifier), ('RF', RandomForestClassifier)]:
            model = Model(**params, random_state=42)
            model.fit(X_train, tr_labels)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(tt_labels, y_pred)
            mcc = matthews_corrcoef(tt_labels, y_pred)
            auc = roc_auc_score(tt_labels, y_prob)
            tn, fp, fn, tp = confusion_matrix(tt_labels, y_pred).ravel()
            sens = tp / (tp + fn)
            spec = tn / (tn + fp)

            result = {
                'Index': i,
                'Model': model_name,
                'Params': params,
                'Accuracy': acc,
                'MCC': mcc,
                'AUC': auc,
                'Sensitivity': sens,
                'Specificity': spec
            }

            all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'results/{dataset_name}_Plain_ERT_RF_all_results.csv', index=False)

    # 儲存各模型最佳結果
    for model_name in ['ERT', 'RF']:
        best_row = results_df[results_df['Model'] == model_name].sort_values('Accuracy', ascending=False).iloc[0]
        pd.DataFrame([best_row]).to_csv(f'results/{dataset_name}_Plain_{model_name}_best_result.csv', index=False)

    print(f"完成：{dataset_name} ERT & RF 單模結果")

def select_features(all_features, dataset_name):
    print('Feature selection...')
    feature_index = pd.read_csv(f"feature_index_select_{dataset_name}.csv")
    new_feature = []
    original_data = pd.DataFrame(all_features)
    for i in list(feature_index.iloc[0, 1:]):
        new_feature.append(original_data[int(i)])  # 选择特征
    features = np.array(new_feature).T  # 在转化为矩阵，feature即为最终选出的最优特征子集
    return features

def test_if_select_same_feature(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    train_embs = [tr_embs[19], tr_embs[22], tr_embs[18], tr_embs[21], tr_embs[20]]

    if dataset_name == 'MD': n_select = 393
    elif dataset_name == 'RD': n_select = 389

    X_train = np.concatenate(train_embs, axis=1)
    estimator = DecisionTreeClassifier(random_state=42)
    selector = RFE(estimator, n_features_to_select=n_select)
    selector.fit(X_train, tr_labels)
    X_train_rfe = selector.transform(X_train)
    X_test_rfe = selector.transform(X_test)

def repro(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    train_embs = [tr_embs[19], tr_embs[22], tr_embs[18], tr_embs[21], tr_embs[20]]
    test_embs = [tt_embs[19], tt_embs[22], tt_embs[18], tt_embs[21], tt_embs[20]]

    if dataset_name == 'MD': n_select = 393
    elif dataset_name == 'RD': n_select = 389

    X_train = np.concatenate(train_embs, axis=1)
    X_test = np.concatenate(test_embs, axis=1)
    X_all = np.concatenate([X_train, X_test], axis=0)
    all_labels = np.concatenate([tr_labels, tt_labels], axis=0)

    # estimator = DecisionTreeClassifier(random_state=42)
    # selector = RFE(estimator, n_features_to_select=n_select)
    # selector.fit(X_train, tr_labels)
    # X_train_rfe = selector.transform(X_train)
    # X_test_rfe = selector.transform(X_test)

    # X_train = np.load(f'{dataset_name}_X_train.npy')
    # X_test = np.load(f'{dataset_name}_X_test.npy')
    # tr_labels = np.load(f'{dataset_name}_tr_labels.npy')
    # tt_labels = np.load(f'{dataset_name}_tt_labels.npy')
    # X_all = np.concatenate([X_train, X_test], axis=0)
    # all_labels = np.concatenate([tr_labels, tt_labels], axis=0)

    X_train_rfe = select_features(X_train, dataset_name)
    X_test_rfe = select_features(X_test, dataset_name)
    X_all_rfe = select_features(X_all, dataset_name)

    scaler = MinMaxScaler()
    scaler.fit(X_train_rfe)
    X_train_rfe = scaler.transform(X_train_rfe)
    # scaler2 = MinMaxScaler()
    # scaler2.fit(X_test_rfe)
    X_test_rfe = scaler.transform(X_test_rfe)

    # if dataset_name == 'MD':
    #     base_learners = [
    #         ('ert', ExtraTreesClassifier(min_samples_split=3, n_estimators=850, random_state=1214)),
    #         ('rf', RandomForestClassifier(max_depth=9, n_estimators=75, max_features='sqrt', random_state=1412))
    #     ]
    # elif dataset_name == 'RD':
    #     base_learners = [
    #         ('ert', ExtraTreesClassifier(min_samples_split=5, n_estimators=1650, max_features='log2', random_state=1214)),
    #         ('rf', RandomForestClassifier(n_estimators=1825, max_depth= 9, max_features='log2', random_state=1412))
    #     ]
    if dataset_name == 'MD':
        base_learners = [
            ExtraTreesClassifier(min_samples_split=3, n_estimators=850, random_state=1214),
            RandomForestClassifier(max_depth=9, n_estimators=75, max_features='sqrt', random_state=1412)
        ]
    elif dataset_name == 'RD':
        base_learners = [
            ExtraTreesClassifier(min_samples_split=5, n_estimators=1650, max_features='log2', random_state=1214),
            RandomForestClassifier(n_estimators=1825, max_depth= 9, max_features='log2', random_state=1412)
        ]
    meta_model = LogisticRegressionCV()
    stack = StackingCVClassifier(
        classifiers=base_learners,
        meta_classifier=meta_model,
        cv=10,  # 在訓練 base learners 時用的 cross-validation
        n_jobs=-1,
        random_state=1412, 
        use_probas=True
    )
    stack.fit(X_train_rfe, tr_labels)
    # stack.fit(X_all_rfe, all_labels)
    y_pred = stack.predict(X_test_rfe)
    y_prob = stack.predict_proba(X_test_rfe)[:, 1]

    acc = accuracy_score(tt_labels, y_pred)
    mcc = matthews_corrcoef(tt_labels, y_pred)
    auc = roc_auc_score(tt_labels, y_prob)
    tn, fp, fn, tp = confusion_matrix(tt_labels, y_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)

    # 輸出 CSV
    results_df = pd.DataFrame([{
        'Accuracy': acc,
        'MCC': mcc,
        'AUC': auc,
        'Sensitivity': sens,
        'Specificity': spec
    }])

    results_df.to_csv(f'results/{dataset_name}_correct_Hyb_Senc_results.csv', index=False)

def repro_RD():
    train_data_file_name = f'data/ATPdataset/MD_train.csv'
    test_data_file_name = f'data/ATPdataset/MD_test.csv'
    tr_embs, tt_embs, emb_ways, MD_tr_labels, MD_tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, 61, device, 'MD')
    MD_train_embs = [tr_embs[19], tr_embs[22], tr_embs[18], tr_embs[21], tr_embs[20]]
    MD_test_embs = [tt_embs[19], tt_embs[22], tt_embs[18], tt_embs[21], tt_embs[20]]
    train_data_file_name = f'data/ATPdataset/RD_train.csv'
    test_data_file_name = f'data/ATPdataset/RD_test.csv'
    tr_embs, tt_embs, emb_ways, RD_tr_labels, RD_tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, 61, device, 'RD')
    RD_train_embs = [tr_embs[19], tr_embs[22], tr_embs[18], tr_embs[21], tr_embs[20]]
    RD_test_embs = [tt_embs[19], tt_embs[22], tt_embs[18], tt_embs[21], tt_embs[20]]
    MD_train_embs = np.concatenate(MD_train_embs, axis=1)
    RD_train_embs = np.concatenate(RD_train_embs, axis=1)
    MD_test_embs = np.concatenate(MD_test_embs, axis=1)
    RD_test_embs = np.concatenate(RD_test_embs, axis=1)
    X_train = np.concatenate([MD_train_embs, RD_train_embs], axis=0)
    X_test = np.concatenate([MD_test_embs, RD_test_embs], axis=0)
    X_all = np.concatenate([X_train, X_test], axis=0)
    all_labels = np.concatenate([MD_tr_labels, RD_tr_labels, MD_tt_labels, RD_tt_labels], axis=0)
    X_test_only_MD = MD_test_embs
    X_test_only_RD = RD_test_embs

    X_train_rfe = select_features(X_train, 'MD')
    X_test_rfe = select_features(X_test, 'MD')
    X_test_MD_rfe = select_features(X_test_only_MD, 'MD')
    X_all_rfe = select_features(X_all, 'MD')
    # X_train_rfe = select_features(X_train, 'RD')
    # X_test_rfe = select_features(X_test, 'RD')
    # X_test_RD_rfe = select_features(X_test_only_RD, 'RD')
    # X_all_rfe = select_features(X_all, 'RD')

    # scaler = MinMaxScaler()
    # scaler.fit(X_all_rfe)
    # X_all_rfe = scaler.transform(X_all_rfe)
    scaler2 = MinMaxScaler()
    scaler2.fit(X_test_MD_rfe)
    X_test_MD_rfe = scaler2.transform(X_test_MD_rfe)

    base_learners = [
        ExtraTreesClassifier(min_samples_split=3, n_estimators=850, random_state=1214),
        RandomForestClassifier(max_depth=9, n_estimators=75, max_features='sqrt', random_state=1412)
    ]
    # base_learners = [
    #     ExtraTreesClassifier(min_samples_split=5, n_estimators=1650, max_features='log2', random_state=1214),
    #     RandomForestClassifier(n_estimators=1825, max_depth= 9, max_features='log2', random_state=1412)
    # ]
    meta_model = LogisticRegressionCV()
    stack = StackingCVClassifier(
        classifiers=base_learners,
        meta_classifier=meta_model,
        cv=10,  # 在訓練 base learners 時用的 cross-validation
        n_jobs=-1,
        random_state=1412, 
        use_probas=True
    )
    stack.fit(X_all_rfe, all_labels)
    y_pred = stack.predict(X_test_MD_rfe)
    y_prob = stack.predict_proba(X_test_MD_rfe)[:, 1]

    acc = accuracy_score(RD_tt_labels, y_pred)
    mcc = matthews_corrcoef(RD_tt_labels, y_pred)
    auc = roc_auc_score(RD_tt_labels, y_prob)
    tn, fp, fn, tp = confusion_matrix(RD_tt_labels, y_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)

    # 輸出 CSV
    results_df = pd.DataFrame([{
        'Accuracy': acc,
        'MCC': mcc,
        'AUC': auc,
        'Sensitivity': sens,
        'Specificity': spec
    }])

    results_df.to_csv(f'results/MD_train-with-RD-all_nonorm_Hyb_Senc_results.csv', index=False)

def repro_each_feature(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    feature_indices = [19, 22, 18, 21, 20]
    models = {
        "ExtraTrees_MD": ExtraTreesClassifier(min_samples_split=3, n_estimators=850, random_state=1214),
        "RandomForest_MD": RandomForestClassifier(max_depth=9, n_estimators=75, max_features='sqrt', random_state=1412),
        "ExtraTrees_RD": ExtraTreesClassifier(min_samples_split=5, n_estimators=1650, max_features='log2', random_state=1214),
        "RandomForest_RD": RandomForestClassifier(n_estimators=1825, max_depth=9, max_features='log2', random_state=1412)
    }

    results = []
    
    for idx in feature_indices:
        X_train = tr_embs[idx]
        X_test = tt_embs[idx]
        
        for model_name, model in models.items():
            if ("MD" in model_name and dataset_name == "MD") or ("RD" in model_name and dataset_name == "RD"):
                model.fit(X_train, tr_labels)
                y_pred = model.predict(X_test)
                acc = accuracy_score(tt_labels, y_pred)
                
                results.append({
                    "Feature": emb_ways[idx],
                    "Model": model_name,
                    "Accuracy": acc
                })
    
    df = pd.DataFrame(results)
    df.to_csv(f"results/feature_model_results_{dataset_name}.csv", index=False)

def repro_model(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    train_embs = [tr_embs[19], tr_embs[22], tr_embs[18], tr_embs[21], tr_embs[20]]
    test_embs = [tt_embs[19], tt_embs[22], tt_embs[18], tt_embs[21], tt_embs[20]]
    X_train = np.concatenate(train_embs, axis=1)
    X_test = np.concatenate(test_embs, axis=1)
    X_train_rfe = select_features(X_train, dataset_name)
    X_test_rfe = select_features(X_test, dataset_name)
    model = joblib.load(f'./Hyb_model/sclf_{dataset_name}.pkl')
    scaler = MinMaxScaler()
    scaler.fit(X_train_rfe)
    X_train_rfe = scaler.transform(X_train_rfe)
    X_test_rfe = scaler.transform(X_test_rfe)
    y_prob = model.predict_proba(X_test_rfe)[:, 1]
    y_pred = model.predict(X_test_rfe)
    acc = accuracy_score(tt_labels, y_pred)
    mcc = matthews_corrcoef(tt_labels, y_pred)
    auc = roc_auc_score(tt_labels, y_prob)
    tn, fp, fn, tp = confusion_matrix(tt_labels, y_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    results_df = pd.DataFrame([{
        'Accuracy': acc,
        'MCC': mcc,
        'AUC': auc,
        'Sensitivity': sens,
        'Specificity': spec
    }])
    print(results_df)

def get_ten_fold(embs, labels, model, seed):
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    mean_acc = []
    for k, (tr_idx, tt_idx) in enumerate(kf.split(labels)):
        tr_embs, tt_embs = embs[tr_idx], embs[tt_idx]
        tr_labels, tt_labels = labels[tr_idx], labels[tt_idx]
        model.fit(tr_embs, tr_labels)
        y_pred = model.predict(tt_embs)
        mean_acc.append(accuracy_score(tt_labels, y_pred))
    return sum(mean_acc) / len(mean_acc)

def repro_each_model(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    feature_indices = [19, 22, 18, 21, 20]
    param_space = [
        {'n_estimators': ne, 'max_depth': md, 'max_features': mf}
        for ne, md, mf in product([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], [None, 10, 20], ['sqrt', 'log2'])
    ]
    all_results = []
    for feature_idx in feature_indices:
        X_train = np.array(tr_embs[feature_idx])  # 只取單獨的 feature
        X_test = np.array(tt_embs[feature_idx])
        for i, params in enumerate(param_space):
            for model_name, Model in [('ERT', ExtraTreesClassifier), ('RF', RandomForestClassifier)]:
                mean_acc = []
                for model_seed in range(1):
                    model = Model(**params, random_state=model_seed)
                    model.fit(X_train, tr_labels)
                    y_pred = model.predict(X_test)
                    mean_acc.append(accuracy_score(tt_labels, y_pred))
                acc = sum(mean_acc) / len(mean_acc)
                result = {
                    'Feature': emb_ways[feature_idx],
                    'Seed': model_seed,
                    'Model': model_name,
                    'Params': params,
                    'Accuracy': acc,
                }
                all_results.append(result)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'results/repro/{dataset_name}_samefeature_results_seed{model_seed}.csv', index=False)

    print(f"完成：{dataset_name} ERT & RF 單獨 feature 表現")
    split_csv_by_model(dataset_name)

def repro_each_model_10fold(train_data_file_name, test_data_file_name, pad_size, dataset_name):
    tr_embs, tt_embs, emb_ways, tr_labels, tt_labels = get_ind_embs_set(train_data_file_name, test_data_file_name, pad_size, device, dataset_name)
    feature_indices = [19, 22, 18, 21, 20]
    param_space = [
        {'n_estimators': ne, 'max_depth': md, 'max_features': mf}
        for ne, md, mf in product([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], [None, 10, 20], ['sqrt', 'log2'])
    ]
    all_results = []
    model_seed = 0
    for feature_idx in feature_indices:
        X_train = np.array(tr_embs[feature_idx])  # 只取單獨的 feature
        X_test = np.array(tt_embs[feature_idx])
        for i, params in enumerate(param_space):
            for model_name, Model in [('ERT', ExtraTreesClassifier), ('RF', RandomForestClassifier)]:
                mean_acc_in_seed = []
                for model_seed in range(1):
                    model = Model(**params, random_state=model_seed)
                    mean_acc_in_seed.append(get_ten_fold(X_train, tr_labels, model, model_seed))
                acc = sum(mean_acc_in_seed) / len(mean_acc_in_seed)
                result = {
                    'Feature': emb_ways[feature_idx],
                    'Seed': model_seed,
                    'Model': model_name,
                    'Params': params,
                    'Accuracy': acc,
                }
                all_results.append(result)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'results/repro/{dataset_name}_{model_seed}times_10fold_results.csv', index=False)

    print(f"完成：{dataset_name} ERT & RF 單獨 feature 表現")
    split_csv_by_model(dataset_name)

def split_csv_by_model(dataset_name):
    # 讀取 CSV 檔案
    df = pd.read_csv(f'results/repro/{dataset_name}_10fold_results_seed0.csv')

    # 根據 Model 欄位分割成兩個 DataFrame
    rf_df = df[df['Model'] == 'RF']
    ert_df = df[df['Model'] == 'ERT']

    # 存成兩個 CSV 檔案
    rf_df.to_csv(f'results/repro/{dataset_name}_RF_10fold_results.csv', index=False)
    ert_df.to_csv(f'results/repro/{dataset_name}_ERT_10fold_results.csv', index=False)

    print(f"已成功分割為 {dataset_name}_RF_10fold_results.csv 和 {dataset_name}_ERT_10fold_results.csv")

def get_best_param(dataset_name):
    results_df = pd.read_csv(f'results/repro/{dataset_name}_0times_10fold_results.csv')

    # 依據 Params 分組並計算平均 Accuracy
    avg_results = results_df.groupby(['Model', results_df['Params'].apply(str)])['Accuracy'].mean().reset_index()

    # 找出每個 Model 平均 Accuracy 最高的參數組合
    best_params_per_model = avg_results.loc[avg_results.groupby('Model')['Accuracy'].idxmax()]

    for _, row in best_params_per_model.iterrows():
        print(f"模型: {row['Model']}")
        print(f"最佳參數: {row['Params']}")
        print(f"最高平均準確率: {row['Accuracy']}\n")

def check_model(dataset_name):
    X_train = np.load(f'{dataset_name}_X_train.npy')
    X_test = np.load(f'{dataset_name}_X_test.npy')
    tr_labels = np.load(f'{dataset_name}_tr_labels.npy')
    tt_labels = np.load(f'{dataset_name}_tt_labels.npy')
    X_train_rfe = select_features(X_train, dataset_name)
    X_test_rfe = select_features(X_test, dataset_name)

    scaler = MinMaxScaler()
    scaler.fit(X_test_rfe)
    # X_train_rfe = scaler.transform(X_train_rfe)
    X_test_rfe = scaler.transform(X_test_rfe)

    model = joblib.load(f'./Hyb_model/sclf_{dataset_name}.pkl')
    print(model)
    print(dkkdkdkdkd)
    y_pred = model.predict(X_test_rfe)
    y_prob = model.predict_proba(X_test_rfe)[:, 1]
    acc = accuracy_score(tt_labels, y_pred)
    mcc = matthews_corrcoef(tt_labels, y_pred)
    auc = roc_auc_score(tt_labels, y_prob)
    tn, fp, fn, tp = confusion_matrix(tt_labels, y_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    print({
        'Accuracy': acc,
        'MCC': mcc,
        'AUC': auc,
        'Sensitivity': sens,
        'Specificity': spec
    })

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
    # dataset_names = ['ATP', 'MD', 'RD']
    # pad_sizes = [75, 61, 61]
    dataset_names = ['MD', 'RD']
    pad_sizes = [61, 61]
    # dataset_names = ['RD']
    # pad_sizes = [61]
    for i, dataset_name in enumerate(dataset_names):
        train_data_file_name = f'data/ATPdataset/{dataset_name}_train.csv'
        test_data_file_name = f'data/ATPdataset/{dataset_name}_test.csv'
        pad_size = pad_sizes[i]
        repro(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        # repro_each_feature(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        # repro_model(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        # plain_ert_rf_gridsearch(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        # repro_each_model(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        # repro_each_model_10fold(train_data_file_name, test_data_file_name, pad_size, dataset_name)
        # split_csv_by_model(dataset_name)
        # get_best_param(dataset_name)
    # check_model('RD')
    # repro_RD()