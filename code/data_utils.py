import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from collections import deque


def read_rsna(MAIN_PATH, NUM_FEAT):
    rsna_path = os.path.join(MAIN_PATH, "data", f"RSNA_{NUM_FEAT}")

    train_folds = []
    test_folds = []
    for i in range(1,6):
        train_file = f"{rsna_path}/train/RSNA_train_{NUM_FEAT}_{i}.csv"
        test_file = f"{rsna_path}/test/RSNA_test_{NUM_FEAT}_{i}.csv"
        train_ds = pd.read_csv(train_file)
        test_ds = pd.read_csv(test_file)

        train_groups = train_ds.groupby("bag_name")
        train_bags_data = []
        train_bags_labels = []
        train_bags_y_labels = []
        for _, group in train_groups:
            train_bags_data.append(group.iloc[:, 0:NUM_FEAT].values)
            train_bags_labels.append(group.iloc[:, NUM_FEAT+4].values[0])
            train_bags_y_labels.append(group.iloc[:, NUM_FEAT+1].values)
        train_folds.append((train_bags_data, train_bags_labels, train_bags_y_labels))

        test_groups = test_ds.groupby("bag_name")
        test_bags_data = []
        test_bags_labels = []
        test_bags_y_labels = []
        for _, group in test_groups:
            test_bags_data.append(group.iloc[:, 0:NUM_FEAT].values)
            test_bags_labels.append(group.iloc[:, NUM_FEAT+4].values[0])
            test_bags_y_labels.append(group.iloc[:, NUM_FEAT+1].values)
        test_folds.append((test_bags_data, test_bags_labels, test_bags_y_labels))

    return train_folds, test_folds

def read_cq500(MAIN_PATH, NUM_FEAT):
    cq500_path = os.path.join(MAIN_PATH, "data", f"CQ500_{NUM_FEAT}")
    folds = []
    for i in range(1,6):
        file = f"{cq500_path}/CQ500_{NUM_FEAT}_{i}.csv"
        ds = pd.read_csv(file)

        groups = ds.groupby("bag_name")
        bags_data = []
        bags_labels = []
        for _, group in groups:
            bags_data.append(group.iloc[:, 0:NUM_FEAT].values)
            bags_labels.append(group.iloc[:, NUM_FEAT].values[0])
        folds.append((bags_data, bags_labels))

    return folds


def read_mnist(MAIN_PATH):
    mnist_path = os.path.join(MAIN_PATH, "data", "mnist")
    train_file = f"{mnist_path}/mnist_train.csv"
    test_file = f"{mnist_path}/mnist_test.csv"

    train_ds = pd.read_csv(train_file)
    train_labels = train_ds["label"].to_numpy()
    train_data = train_ds.drop(["label"], axis=1).to_numpy()

    test_ds = pd.read_csv(test_file)
    test_labels = test_ds["label"].to_numpy()
    test_data = test_ds.drop(["label"], axis=1).to_numpy()

    return train_data, train_labels, test_data, test_labels

def read_musk(MAIN_PATH, type="musk1"):
    musk_path = os.path.join(MAIN_PATH, "data", "musk", f"{type}.csv")
    df = pd.read_csv(musk_path, header=None)
    bags_ids = df[0].unique()
    bags = [df[df[0]==id][df.columns.values[2:-1]].values.tolist() for id in bags_ids]
    bags = [np.array(bag) for bag in bags]
    bags_y_labels = [df[df[0]==id][df.columns.values[-1]].values.astype(int).tolist() for id in bags_ids]
    bags_labels = [np.max(bag) for bag in bags_y_labels]

    return bags, bags_labels, bags_y_labels


def create_bags(data, labels, obj_labels=[2,9], bag_size=10, num_bags=7000):
    np.random.seed(0)

    pos_idx = np.where(np.isin(labels, obj_labels))[0]
    np.random.shuffle(pos_idx)
    neg_idx = np.where(~np.isin(labels, obj_labels))[0]
    np.random.shuffle(neg_idx)

    num_pos_bags = num_bags // 2
    num_neg_bags = num_bags - num_pos_bags

    pos_idx_queue = deque(pos_idx)
    neg_idx_queue = deque(neg_idx)

    bags = []
    bags_labels = []
    bags_y_labels = []
    for i in range(num_pos_bags):
        bag = []
        y_labels = []
        num_positives = np.random.randint(1, bag_size//2)
        num_negatives = bag_size - num_positives
        for _ in range(num_positives):
            a = pos_idx_queue.pop()
            bag.append(data[a])
            y_labels.append(labels[a])
            pos_idx_queue.appendleft(a)
        for _ in range(num_negatives):
            a = neg_idx_queue.pop()
            bag.append(data[a])
            y_labels.append(labels[a])
            neg_idx_queue.appendleft(a)

        idx_sort = np.argsort(y_labels)
        bag = np.stack(bag)[idx_sort]
        y_labels = np.array(y_labels)[idx_sort]
        y_labels = np.where(np.isin(y_labels, obj_labels), 1, 0)
        bag_label = np.max(y_labels)

        bags.append(bag)
        bags_labels.append(bag_label)
        bags_y_labels.append(y_labels)

    for i in range(num_neg_bags):
        bag = []
        y_labels = []
        for _ in range(bag_size):
            a = neg_idx_queue.pop()
            bag.append(data[a])
            y_labels.append(labels[a])
            neg_idx_queue.appendleft(a)

        idx_sort = np.argsort(y_labels)
        bag = np.stack(bag)[idx_sort]
        y_labels = np.array(y_labels)[idx_sort]
        y_labels = np.zeros_like(y_labels)
        bag_label = 0

        bags.append(bag)
        bags_labels.append(bag_label)
        bags_y_labels.append(y_labels)
    
    idx_list = np.arange(len(bags))
    np.random.shuffle(idx_list)
    bags = [bags[i] for i in idx_list]
    bags_labels = [bags_labels[i] for i in idx_list]
    bags_y_labels = [bags_y_labels[i] for i in idx_list]

    return bags, bags_labels, bags_y_labels

def generate_random_bags(img_list_len, low, up):
    bags_idx = []
    j = 0
    lim = 0
    while lim < img_list_len:
        num_inst = np.random.randint(low, up)
        if lim+num_inst >= img_list_len:
            num_inst = img_list_len - lim
        bags_idx = bags_idx + [j for _ in range(num_inst)]
        lim = lim + num_inst
        j = j+1
    return bags_idx
    

def prepare_data_mil(data, labels, obj_label, mode, num_inst_per_bag=9):
    
    new_data, inst_labels, bags_idx = create_bags(data, labels, obj_label, mode)

    bag_labels = np.array( [1 if np.any(inst_labels[bags_idx==bag_id]==1) else 0 for bag_id in np.unique(bags_idx)] )
    
    inst_bag_labels = np.array( [ bag_labels[bag_id] for bag_id in bags_idx ] )

    return new_data, inst_labels, bags_idx, inst_bag_labels, bag_labels

def predict_bags(preds, bag_index):
    bags_preds = np.zeros(len(np.unique(bag_index)))
    for i, bag_id in enumerate(np.unique(bag_index)):
        bags_preds[i] = np.max(preds[bag_index==bag_id])
    return bags_preds


def compute_attcnn_metrics(MAIN_PATH, NUM_FEAT):
    instlogloss_mean = {8: 0.33467 , 32:0.356112, 128: 0.350143}
    instlogloss_std = {8: 0.006, 32: 0.001, 128: 0.003}
    
    inst_acc_vec = []
    inst_f1_vec = []
    inst_auc_vec = []

    bag_acc_vec = []
    bag_f1_vec = []
    bag_auc_vec = []
    bag_log_loss_vec = []
    for i in range(1, 6):
        file_path = f"{MAIN_PATH}/data/RSNA_{NUM_FEAT}/test/RSNA_test_{NUM_FEAT}_{i}.csv"
        df = pd.read_csv(file_path)
        y_pred = df["cnn_prediction"].to_numpy()
        y_true = df["instance_label"].to_numpy()
        acc = np.mean(y_true == y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        inst_acc_vec.append(acc)
        inst_f1_vec.append(f1)
        inst_auc_vec.append(auc)

        df_g = df.groupby("bag_name")
        T_pred = []
        T_prob_pred = []
        T_true = []
        for _, group in df_g:
            y_pred = group["cnn_prediction"].to_numpy()
            #T_pred.append(np.max(y_pred))
            if "bag_cnn_prediction" in group.columns:
                T_pred.append(group["bag_cnn_prediction"].to_numpy()[0])
            else:
                T_pred.append(np.max(y_pred))
            if "bag_cnn_probability" in group.columns:
                T_prob_pred.append(group["bag_cnn_probability"].to_numpy()[0])
            else:
                T_prob_pred.append(np.max(y_pred))
            T_true.append(group["bag_label"].to_numpy()[0])
        T_pred = np.array(T_pred)
        T_prob_pred = np.array(T_prob_pred)
        T_true = np.array(T_true)
        acc = np.mean(T_true == T_pred)
        f1 = f1_score(T_true, T_pred)
        auc = roc_auc_score(T_true, T_pred)
        ll = log_loss(T_true, T_prob_pred)
        bag_acc_vec.append(acc)
        bag_f1_vec.append(f1)
        bag_auc_vec.append(auc)
        bag_log_loss_vec.append(ll)

    metrics_dic = { "rsna/inst/acc_mean": np.mean(inst_acc_vec), "rsna/inst/acc_std": np.std(inst_acc_vec),
                    "rsna/inst/f1_mean": np.mean(inst_f1_vec), "rsna/inst/f1_std": np.std(inst_f1_vec),
                    "rsna/inst/log_loss_mean": instlogloss_mean[NUM_FEAT], "rsna/inst/log_loss_std": instlogloss_std[NUM_FEAT],
                    "rsna/inst/auc_mean": np.mean(inst_auc_vec), "rsna/inst/auc_std": np.std(inst_auc_vec), 
                    "rsna/bag/acc_mean": np.mean(bag_acc_vec), "rsna/bag/acc_std": np.std(bag_acc_vec),
                    "rsna/bag/f1_mean": np.mean(bag_f1_vec), "rsna/bag/f1_std": np.std(bag_f1_vec),
                    "rsna/bag/log_loss_mean": np.mean(bag_log_loss_vec), "rsna/bag/log_loss_std": np.std(bag_log_loss_vec),
                    "rsna/bag/auc_mean": np.mean(bag_auc_vec), "rsna/bag/auc_std": np.std(bag_auc_vec) }

    bag_acc_vec = []
    bag_f1_vec = []
    bag_auc_vec = []
    bag_log_loss_vec = []
    for i in range(1, 6):
        file_path = f"{MAIN_PATH}/data/CQ500_{NUM_FEAT}/CQ500_{NUM_FEAT}_{i}.csv"
        df = pd.read_csv(file_path)

        df_g = df.groupby("bag_name")
        T_pred = []
        T_prob_pred = []
        T_true = []
        for _, group in df_g:
            T_pred.append(group["bag_cnn_prediction"].to_numpy()[0])
            T_prob_pred.append(group["bag_cnn_probability"].to_numpy()[0])
            T_true.append(group["bag_label"].to_numpy()[0])
        T_pred = np.array(T_pred)
        T_prob_pred = np.array(T_prob_pred)
        T_true = np.array(T_true)
        acc = np.mean(T_true == T_pred)
        f1 = f1_score(T_true, T_pred)
        auc = roc_auc_score(T_true, T_prob_pred)
        ll = log_loss(T_true, T_prob_pred)
        bag_acc_vec.append(acc)
        bag_f1_vec.append(f1)
        bag_auc_vec.append(auc)
        bag_log_loss_vec.append(ll)

    metrics_dic["cq500/bag/acc_mean"] = np.mean(bag_acc_vec)
    metrics_dic["cq500/bag/acc_std"] = np.std(bag_acc_vec)
    metrics_dic["cq500/bag/f1_mean"] = np.mean(bag_f1_vec)
    metrics_dic["cq500/bag/f1_std"] = np.std(bag_f1_vec)
    metrics_dic["cq500/bag/log_loss_mean"] = np.mean(bag_log_loss_vec)
    metrics_dic["cq500/bag/log_loss_std"] = np.std(bag_log_loss_vec)
    metrics_dic["cq500/bag/auc_mean"] = np.mean(bag_auc_vec)
    metrics_dic["cq500/bag/auc_std"] = np.std(bag_auc_vec)

    return metrics_dic

