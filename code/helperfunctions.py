import numpy as np
import sklearn.metrics as skmetrics
import cv2

def bags2index(bags):
    index = []
    for i in range(len(bags)):
        bag = bags[i]
        index = index + [i for _ in range(len(bag))]
    return np.array(index)

def bags2inst_bag_labels(bags, bag_labels):
    inst_bag_labels = []
    for x,y in zip(bags, bag_labels):
        inst_bag_labels = inst_bag_labels + [y for _ in range(len(x))]
    return np.array(inst_bag_labels)

def bags2instances(bags):
    return [instance for bag in bags for instance in bag]

def sigmoid_np(x):
    """
    Logistic Sigmoid function $\sigma(x) = (1 + exp(-x))^(-1)$
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))

def compute_metrics(y_true, y_prob, threshold=0.5):

    y_pred = (y_prob > threshold).astype(int)
 
    log_loss = skmetrics.log_loss(y_true, y_prob)
    acc = skmetrics.accuracy_score(y_true, y_pred)
    f1 = skmetrics.f1_score(y_true, y_pred)
    auc = skmetrics.roc_auc_score(y_true, y_prob)
    return acc, f1, auc, log_loss

def compute_inducing_points(bags, bags_labels, num_inducing, normalize=True):
    
    Xtrain = np.array(bags2instances(bags))
    if normalize:
        data_mean, data_std = np.mean(Xtrain, 0), np.std(Xtrain, 0)
        data_std[data_std == 0] = 1.0
        Xtrain = (Xtrain - data_mean) / data_std
    InstBagLabel = bags2inst_bag_labels(bags, bags_labels)
    Xzeros = Xtrain[InstBagLabel == 0].astype("float32")
    Xones = Xtrain[InstBagLabel == 1].astype("float32")
    num_ind_pos = np.uint32(np.floor(num_inducing * 0.5))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    nr_attempts = 10
    _, _, Z0_mat = cv2.kmeans(Xzeros, num_inducing - num_ind_pos, None, criteria, attempts=nr_attempts, flags=cv2.KMEANS_RANDOM_CENTERS)
    _, _, Z1_mat = cv2.kmeans(Xones, num_ind_pos, None, criteria, attempts=nr_attempts, flags=cv2.KMEANS_RANDOM_CENTERS)
    Z_mat = np.concatenate((Z0_mat, Z1_mat))
    return Z_mat

