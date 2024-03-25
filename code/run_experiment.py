import pandas as pd
import numpy as np
import os
import argparse

import jax
jax.config.update("jax_enable_x64", True)

from data_utils import create_bags, read_musk, read_mnist, read_rsna, read_cq500
from train_utils import run_cross_val, run_cross_val_ich

from sklearn.model_selection import StratifiedKFold, train_test_split

from psi_functions import psi_vgpmil, psi_gvgpmil, dif_psi_vgpmil, dif_psi_gvgpmil

# from VGPMIL import VGPMIL 
# from G_VGPMIL import G_VGPMIL
from psiVGPMIL import psiVGPMIL

WORK_PATH = os.environ['WORK_PATH']
MAIN_PATH = f"{WORK_PATH}/psi-VGPMIL"

def lengthscale_type(value):
    if value in ['sqrt_d', 'd']:
        return value
    else:
        return float(value)

parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--save_results", action="store_true", help="Save results")
parser.add_argument("--update_hyperparams", action="store_true", help="Whether to update hyperparameters or not")

parser.add_argument("--model_name", type=str, default="VGPMIL", help="VGPMIL/G_VGPMIL")
parser.add_argument("--dataset_name", type=str, default="mnist", help="mnist/musk1/musk2/ich")
parser.add_argument("--use_pca", action="store_true", help="Use PCA (MNIST)")
parser.add_argument("--num_feat", type=int, default=8, help="Number of features (ICH problem) 98/32/128")

parser.add_argument("--warmup_iters", type=int, default=10, help="Number of warmup iterations")
parser.add_argument("--max_iters", type=int, default=40, help="Number of iterations")
parser.add_argument("--early_stop_patience", type=int, default=10, help="Early stop patience")
parser.add_argument("--num_folds", type=int, default=5, help="Number of cross-validation folds")
parser.add_argument("--num_inducing", type=int, default=50, help="Number of inducing points")

parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for hyperparameters optimization")
parser.add_argument("--max_opt_epochs", type=int, default=200, help="Max epochs for hyperparameters optimization")

parser.add_argument("--kernel_ls", type=lengthscale_type, default='d', help="Lenghtscale of the RBF kernel")
parser.add_argument("--kernel_var", type=float, default=0.5, help="Variance of the RBF kernel")
parser.add_argument("--alpha", type=float, default=1.0, help="alpha prior parameter of the Gamma distribution")
parser.add_argument("--beta", type=float, default=4.0, help="beta prior parameter of the Gamma distribution")

args = parser.parse_args()

# Check arguments
if args.model_name == "VGPMIL":
    args.alpha = args.beta = None

if args.dataset_name != 'mnist':
    args.use_pca = False

if args.dataset_name != 'ich':
    args.num_feat = None

if not args.update_hyperparams:
    args.max_iter = args.max_iter + args.warmup_iter
    args.warmup_iter = 0
    args.lr = None
    args.max_opt_epochs = None

print('Arguments:')
for arg in vars(args):
    print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))

# Read data, set D
if args.dataset_name == "musk1":
    bags, bags_labels, bags_y_labels = read_musk(MAIN_PATH, "musk1") 
    D = bags[0].shape[1]
    splits = StratifiedKFold(n_splits=args.num_folds, shuffle=False).split(bags, bags_labels)
    idx_splits = []
    for train_idx, test_idx in splits:
        val_idx = train_idx
        idx_splits.append((train_idx, test_idx, val_idx))
elif args.dataset_name == "musk2":
    bags, bags_labels, bags_y_labels = read_musk(MAIN_PATH, "musk2")
    D = bags[0].shape[1]
    splits = StratifiedKFold(n_splits=args.num_folds, shuffle=False).split(bags, bags_labels)
    idx_splits = []
    for train_idx, test_idx in splits:
        val_idx = train_idx
        idx_splits.append((train_idx, test_idx, val_idx))
elif args.dataset_name == "mnist":
    train_data, train_labels, test_data, test_labels = read_mnist(MAIN_PATH)
    data = np.vstack((train_data, test_data))
    labels = np.hstack((train_labels, test_labels))
    bags, bags_labels, bags_y_labels = create_bags(data, labels)
    splits = StratifiedKFold(n_splits=args.num_folds, shuffle=False).split(bags, bags_labels)
    idx_splits = []
    for train_idx, test_idx in splits:
        new_train_idx, val_idx = train_test_split(train_idx, train_size=0.9, stratify=[bags_labels[i] for i in train_idx], random_state=42)
        idx_splits.append((new_train_idx, test_idx, val_idx))
    if args.use_pca:
        D = 30
    else:
        D = bags[0].shape[1]
elif args.dataset_name == "ich":
    rsna_train_folds, rsna_test_folds = read_rsna(MAIN_PATH, args.num_feat)
    cq500_folds = read_cq500(MAIN_PATH, args.num_feat)
    D = args.num_feat

if args.kernel_ls == 'sqrt_d':
    args.kernel_ls = np.sqrt(D)
elif args.kernel_ls == 'd':
    args.kernel_ls = float(D)
else:
    args.kernel_ls = float(args.kernel_ls)

if args.model_name == "VGPMIL":
    psi_fn = psi_vgpmil
    dif_psi_fn = dif_psi_vgpmil
    psi_params = {}
elif args.model_name == "G_VGPMIL":
    psi_fn = psi_gvgpmil
    dif_psi_fn = dif_psi_gvgpmil
    psi_params = {'alpha': args.alpha, 'beta': args.beta}
else:
    raise ValueError("Invalid model name")

model_builder_fn = lambda: psiVGPMIL(
    num_inducing=args.num_inducing, 
    normalize=True, 
    lr=args.lr, 
    max_opt_epochs=args.max_opt_epochs, 
    kernel_ls=args.kernel_ls, kernel_var=args.kernel_var, 
    update_hyperparams=args.update_hyperparams, 
    psi_fn=psi_fn,
    dif_psi_fn=dif_psi_fn,
    psi_params=psi_params
)

if args.dataset_name == 'ich':
    metrics_dic = run_cross_val_ich(rsna_train_folds, rsna_test_folds, cq500_folds, model_builder_fn, args)
else:
    metrics_dic = run_cross_val(bags, bags_labels, bags_y_labels, idx_splits, model_builder_fn, args)

if args.save_results:

    results_dic = {}
    for k, v in vars(args).items():
        results_dic[k] = v
    for k, v in metrics_dic.items():
        results_dic[k] = v

    results_path = os.path.join(MAIN_PATH, "results", f"results_{args.dataset_name}.csv")
    res_df = pd.DataFrame(results_dic, index=[0])

    if(os.path.isfile(results_path)):
        res_df.to_csv(results_path, mode='a', index=False, header=False)
    else:
        res_df.to_csv(results_path, index=False)