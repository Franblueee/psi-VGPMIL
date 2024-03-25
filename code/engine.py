from tqdm import tqdm
import sklearn.metrics as skmetrics
import numpy as np

def balanced_log_loss(y_true, y_pred, normalize=True):
    """Computes the balanced log loss."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    balanced_weights = np.where(y_true == 1, 0.5, 0.5)
    balanced_weights /= balanced_weights.sum()
    return skmetrics.log_loss(y_true, y_pred, sample_weight=balanced_weights, normalize=normalize)

def evaluate(model, bags, T_labels, y_labels=None):

    metrics_dict = {}
    T_prob_pred, y_prob_pred = model.predict(bags)

    for t in [('bag', T_labels, T_prob_pred), ('inst', y_labels, y_prob_pred)]:
        true_labels = t[1]
        if true_labels is not None:
            pred_probs = t[2]
            pred_labels = (pred_probs > 0.5).astype(int)
            metrics_dict[t[0] + '/log_loss'] = skmetrics.log_loss(true_labels, pred_probs)
            metrics_dict[t[0] + '/acc'] = skmetrics.accuracy_score(true_labels, pred_labels)
            metrics_dict[t[0] + '/f1'] = skmetrics.f1_score(true_labels, pred_labels)
            metrics_dict[t[0] + '/auc'] = skmetrics.roc_auc_score(true_labels, pred_probs)

    return metrics_dict

class Trainer:
    def __init__(self, model, warmup_iters=10, early_stop_patience=10):
        self.model = model
        self.warmup_iters = warmup_iters
        self.early_stop_patience = early_stop_patience
        
        self.best_score = np.inf
        self.best_params = None
        self.stop_iter = None

        self.tol = 0.001
    
    def train_val_step(self, train_bags, train_T_labels, val_bags, val_T_labels):
        
        self.model.train_step()

        results_dict = {}

        # T_prob_pred, _ = self.model.predict(train_bags)
        # results_dict['train/bag/log_loss'] = balanced_log_loss(train_T_labels, T_prob_pred, normalize=True)
        # results_dict['train/bag/auc'] = skmetrics.roc_auc_score(train_T_labels, T_prob_pred)
        # results_dict['train/bag/acc'] = skmetrics.accuracy_score(train_T_labels, (T_prob_pred > 0.5).astype(int))

        T_prob_pred, _ = self.model.predict(val_bags)
        results_dict['val/bag/log_loss'] = balanced_log_loss(val_T_labels, T_prob_pred, normalize=True)
        results_dict['val/bag/auc'] = skmetrics.roc_auc_score(val_T_labels, T_prob_pred)
        results_dict['val/bag/acc'] = skmetrics.accuracy_score(val_T_labels, (T_prob_pred > 0.5).astype(int))

        return results_dict

    def train(self, num_iter, train_bags, train_T_labels, val_bags=None, val_T_labels=None):
        """
        Train the model
        num_iter: maximum number of iterations
        train_bags: list of bags
        train_T_labels: list of bag labels
        """

        if val_bags is None or val_T_labels is None:
            print("aaa")
            val_bags = train_bags
            val_T_labels = train_T_labels

        if not self.model.initialized:
            self.model.initialize(train_bags, train_T_labels)

        update_hyperparams = self.model.update_hyperparams

        # Warmup
        pbar = tqdm(range(self.warmup_iters))
        self.model.update_hyperparams = False
        for it in pbar:
            pbar.set_description(f"Warmup")
            
            results_dict = self.train_val_step(train_bags, train_T_labels, val_bags, val_T_labels)
            if results_dict['val/bag/log_loss'] < self.best_score - self.tol:
                self.best_score = results_dict['val/bag/log_loss']
                self.best_params = self.model.get_params()
            pbar.set_postfix(results_dict)
        
        self.model.update_hyperparams = update_hyperparams
        pbar = tqdm(range(num_iter))
        early_stop_count = 0
        self.stop_iter = num_iter
        for it in pbar:
            
            # print(f"Iteration {it}")
            pbar.set_description(f"Train")

            results_dict = self.train_val_step(train_bags, train_T_labels, val_bags, val_T_labels)
            
            if results_dict['val/bag/log_loss'] < self.best_score - self.tol:
                self.best_score = results_dict['val/bag/log_loss']
                self.best_params = self.model.get_params()
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= self.early_stop_patience:
                    self.stop_iter = it+1
                    break
            
            pbar.set_postfix(results_dict)
        
        self.best_score = self.best_score
        self.stop_iter = self.stop_iter + self.warmup_iters
        self.best_params = self.best_params