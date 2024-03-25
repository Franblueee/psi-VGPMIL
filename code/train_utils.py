import numpy as np

import time

from helperfunctions import bags2instances

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from engine import Trainer, evaluate

def run_cross_val(bags, bags_labels, bags_y_labels, idx_splits, build_model_fn, args):
    metrics_dic = { 'stop_iter' : [], 'time_per_iter' : []}
    n = 1
    for train_idx, val_idx, test_idx in idx_splits:

        print(f"FOLD {n}")
        n = n+1
        print("Train bags:", len(train_idx), "Val bags:", len(val_idx), "Test bags:", len(test_idx))

        train_bags = [bags[i] for i in train_idx]
        train_bags_labels = [bags_labels[i] for i in train_idx]
        train_y_labels = [bags_y_labels[i] for i in train_idx]

        val_bags = [bags[i] for i in val_idx]
        val_bags_labels = [bags_labels[i] for i in val_idx]
        val_y_labels = [bags_y_labels[i] for i in val_idx]

        test_bags = [bags[i] for i in test_idx]
        test_bags_labels = [bags_labels[i] for i in test_idx]
        test_bags_y_labels = [bags_y_labels[i] for i in test_idx]
        test_y_labels = np.concatenate(test_bags_y_labels, axis=0)
        
        if args.use_pca:
            print("Computing PCA")
            Xtrain = np.array(bags2instances(bags))
            pca = PCA(n_components=30)
            pca.fit(Xtrain)
            train_bags = [pca.transform(bag) for bag in train_bags]
            val_bags = [pca.transform(bag) for bag in val_bags]
            test_bags = [pca.transform(bag) for bag in test_bags]

        model = build_model_fn()
        model.initialize(train_bags, train_bags_labels)

        print("Training model")
        trainer = Trainer(model, args.warmup_iters, args.early_stop_patience)
        start_time = time.time()
        trainer.train(args.max_iters, train_bags, train_bags_labels, test_bags, test_bags_labels)
        end_time = time.time()

        metrics_dic["stop_iter"].append(trainer.stop_iter)
        metrics_dic["time_per_iter"].append((end_time - start_time) / trainer.stop_iter)

        best_params = trainer.best_params
        model.set_params(best_params)
        
        print("Evaluating model")
        m_dic = evaluate(model, test_bags, test_bags_labels, test_y_labels)
        for k in m_dic.keys():
            if k not in metrics_dic:
                metrics_dic[k] = []
            metrics_dic[k].append(m_dic[k])
        for k in metrics_dic.keys():
            print(f"{k} = {metrics_dic[k][-1]}")
            
    results_dic = {}
    print("Mean metrics:")
    for k, v in metrics_dic.items():
        metric_mean = np.mean(v)
        metric_std = np.std(v)
        results_dic[f"{k}_mean"] = metric_mean
        results_dic[f"{k}_std"] = metric_std
        print(f"{k}: \tmean = {metric_mean}, \tstd = {metric_std}")
    return results_dic

def run_cross_val_ich(rsna_train_folds, rsna_test_folds, cq500_folds, build_model_fn, args):
    metrics_dic = { 'stop_iter' : [], 'time_per_iter' : [] }
    for n in range(0,5):

        (train_bags_data, train_bags_labels, train_y_labels) = rsna_train_folds[n]
        (test_bags_data, test_bags_labels, test_y_labels) = rsna_test_folds[n]
        (cq500_bags_data, cq500_bags_labels) = cq500_folds[n]

        idx = np.arange(len(train_bags_data))
        train_idx, val_idx = train_test_split(idx, train_size=0.9, stratify=train_bags_labels, random_state=42)

        val_bags_data = [train_bags_data[i] for i in val_idx]
        val_bags_labels = [train_bags_labels[i] for i in val_idx]
        val_y_labels = [train_y_labels[i] for i in val_idx]

        train_bags_data = [train_bags_data[i] for i in train_idx]
        train_bags_labels = [train_bags_labels[i] for i in train_idx]
        train_y_labels = [train_y_labels[i] for i in train_idx]

        train_y_labels = np.concatenate(train_y_labels, axis=0)
        test_y_labels = np.concatenate(test_y_labels, axis=0)

        print(f"FOLD {n+1}")
        print("RSNA Train bags:", len(train_bags_data), "RSNA Test bags:", len(test_bags_data))
        print("CQ500 bags:", len(cq500_bags_data))

        model = build_model_fn()
        model.initialize(train_bags_data, train_bags_labels)

        print("Training model")
        trainer = Trainer(model, args.warmup_iters, args.early_stop_patience)
        start_time = time.time()
        trainer.train(args.max_iters, train_bags_data, train_bags_labels, val_bags_data, val_bags_labels)
        end_time = time.time()
      
        metrics_dic["stop_iter"].append(trainer.stop_iter)
        metrics_dic["time_per_iter"].append((end_time - start_time) / trainer.stop_iter)

        best_params = trainer.best_params
        model.set_params(best_params)

        print("Evaluating model in RSNA")
        m_dic = evaluate(model, test_bags_data, test_bags_labels, test_y_labels)

        for k in m_dic.keys():
            key_name = f"rsna/{k}"
            if key_name not in metrics_dic:
                metrics_dic[key_name] = []
            metrics_dic[key_name].append(m_dic[k])

        print("Evaluating model in CQ500")
        m_dic = evaluate(model, cq500_bags_data, cq500_bags_labels)

        for k in m_dic.keys():
            key_name = f"cq500/{k}"
            if key_name not in metrics_dic:
                metrics_dic[key_name] = []
            metrics_dic[key_name].append(m_dic[k])
        
        for k in metrics_dic.keys():
            print(f"{k} = {metrics_dic[k][-1]}")
    
    results_dic = {}
    print("Mean metrics:")
    for k, v in metrics_dic.items():
        metric_mean = np.mean(v)
        metric_std = np.std(v)
        results_dic[f"{k}_mean"] = metric_mean
        results_dic[f"{k}_std"] = metric_std
        print(f"{k}: \tmean = {metric_mean}, \tstd = {metric_std}")
    return results_dic