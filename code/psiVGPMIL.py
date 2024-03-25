import numpy as np
import cv2

import jax
import jax.numpy as jnp
from jax import jit, vmap

from helperfunctions import bags2index, bags2inst_bag_labels, bags2instances

from variational import mm_rbf_kernel, minus_elbo_psivgpmil, compute_kernel_matrices, update_q_u, update_q_y, update_hyperparameters

class psiVGPMIL:
    def __init__(
            self,
            psi_fn,
            dif_psi_fn,
            num_inducing=50, 
            kernel_ls=1.0, kernel_var=0.5, 
            lr=0.001, max_opt_epochs=200,
            normalize=True, update_hyperparams=True, verbose=False,
            psi_params = {}
        ):
        """
            Init method:
                kernel_ls: kernel lengthscale
                kernel_var: kernel variance
                normalize: whether to normalize the data
                update_hyperparams: whether to update the hyperparameters
                verbose: whether to print verbose output
        """
        self.num_inducing = num_inducing
        self.log_kernel_ls = np.log(kernel_ls)
        self.log_kernel_var = np.log(kernel_var)

        self.psi_fn = psi_fn
        self.dif_psi_fn = dif_psi_fn
        self.psi_params = psi_params

        self.lr = lr
        self.max_opt_epochs = max_opt_epochs
        self.normalize = normalize
        self.update_hyperparams = update_hyperparams
        self.verbose = verbose

        self.lH = np.log(1e12)
        self.kernel_fn = mm_rbf_kernel

        self.initialized = False

        self.stop_iter = 0
        self.step_count = 0
        self.callbacks = []

    def set_callbacks(self, callbacks):
        """
        Set the stopping criterion
        :param stopping_criterion: one of "max_iter", "early_stopping"
        """
        self.callbacks = callbacks
        for callback in self.callbacks:
            callback.set_model(self)

    def get_params(self):
        """
        Get the parameters of the model
        :return: dictionary of parameters
        """
        params = {}
        params["m"] = np.copy(self.m)
        params["S"] = np.copy(self.S)
        params["pi"] = np.copy(self.pi)
        params["kernel_ls"] = np.exp(self.log_kernel_ls)
        params["kernel_var"] = np.exp(self.log_kernel_var)
        params['psi_params'] = self.psi_params
        return params
    
    def set_params(self, params):
        """
        Set the parameters of the model
        :param params: dictionary of parameters
        """
        self.m = params["m"]
        self.S = params["S"]
        self.pi = params["pi"]
        self.log_kernel_ls = np.log(params["kernel_ls"])
        self.log_kernel_var = np.log(params["kernel_var"])
        self.psi_params = params['psi_params']
        _, _, self.Kzzinv, self.KzziKzx, self.f_var = compute_kernel_matrices(self.Z, self.Xtrain, np.exp(self.log_kernel_ls), np.exp(self.log_kernel_var))

    def compute_inducing_points(self, Xtrain, InstBagLabel):
        """
        Compute the inducing points
            Xtrain: nxd array of n instances with d features each
            InstBagLabel:  n-dim vector with the bag label of each instance
        """

        Xzeros = Xtrain[InstBagLabel == 0].astype("float32")
        Xones = Xtrain[InstBagLabel == 1].astype("float32")
        num_ind_pos = np.uint32(np.floor(self.num_inducing * 0.5))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        nr_attempts = 10
        _, _, Z0 = cv2.kmeans(Xzeros, self.num_inducing - num_ind_pos, None, criteria, attempts=nr_attempts, flags=cv2.KMEANS_RANDOM_CENTERS)
        _, _, Z1 = cv2.kmeans(Xones, num_ind_pos, None, criteria, attempts=nr_attempts, flags=cv2.KMEANS_RANDOM_CENTERS)
        Z_mat = np.concatenate((Z0, Z1))
        return Z_mat

    def initialize(self, bags, bags_labels):
        """
        Initialize the model
            bags: list of bags
            bags_labels: list of bag labels
        """

        self.Xtrain = np.array(bags2instances(bags)) # N x D
        self.Bags = bags2index(bags) # N-dim vector with the bag index of each instance
        self.unique_Bags = np.unique(self.Bags) # Unique bag indices
        self.InstBagLabel = bags2inst_bag_labels(bags, bags_labels) # N-dim vector with the bag label of each instance

        self.Ntot = len(self.Bags)               # Nr of Training Instances
        self.B = len(self.unique_Bags)       # Nr of Training Bags

        if self.normalize:
            self.data_mean, self.data_std = np.mean(self.Xtrain, 0), np.std(self.Xtrain, 0)
            self.data_std[self.data_std == 0] = 1.0
            self.Xtrain = (self.Xtrain - self.data_mean) / self.data_std
        else:
            self.data_mean, self.data_std = np.zeros(self.Xtrain.shape[1]), np.ones(self.Xtrain.shape[1])

        # Compute Inducing points
        np.random.seed(0)
        if self.verbose:
            print("Computing inducing points")
        self.Z = self.compute_inducing_points(self.Xtrain, self.InstBagLabel)
        if self.verbose:
            print("Finished computing inducing points")
            
        # Compute Kernel matrices
        _, _, self.Kzzinv, self.KzziKzx, self.f_var = compute_kernel_matrices(self.Z, self.Xtrain, np.exp(self.log_kernel_ls), np.exp(self.log_kernel_var))

        # q(u)
        self.m = np.random.randn(self.num_inducing)
        self.S = np.identity(self.num_inducing) + np.random.randn(self.num_inducing, self.num_inducing) * 0.01

        # q(y)
        self.pi = np.random.uniform(0, 0.1, size=self.Ntot)

        self.Ef = np.random.randn(self.Ntot)
        self.Eff = np.absolute(np.random.randn(self.Ntot))

        self.initialized = True

    def train_step(self):
        """
        The logic for one training step. 
        """

        self.step_count += 1

        # theta_fn = lambda x : self.theta_fn(x, np.exp(self.log_alpha), np.exp(self.log_beta))
        # theta_fn = lambda x : self.theta_fn(x, self.psi_params)

        theta_fn = lambda x : - self.dif_psi_fn(x, **self.psi_params) / (x * self.psi_fn(x, **self.psi_params))

        m, S, Ef, Eff = update_q_u(theta_fn, self.m, self.S, self.pi, self.Eff, self.KzziKzx, self.Kzzinv, self.f_var)
        self.m = m
        self.S = S
        self.Ef = Ef
        self.Eff = Eff

        pi = update_q_y(self.pi, self.Ef, self.Bags, self.unique_Bags, self.InstBagLabel, self.lH)
        self.pi = pi        

        if self.update_hyperparams:
            kernel_params = { 'log_kernel_ls' : self.log_kernel_ls, 'log_kernel_var' : self.log_kernel_var }
            params = [kernel_params, self.psi_params]
            kernel_grad_mask_dict = { 'log_kernel_ls' : 1.0, 'log_kernel_var' : 1.0 }
            psi_grad_mask_dict = { key : 0.0 for key in self.psi_params.keys() }
            grad_mask = [kernel_grad_mask_dict, psi_grad_mask_dict]
            params = update_hyperparameters(params, minus_elbo_psivgpmil, self.psi_fn, self.Xtrain, self.Z, self.m, self.S, self.pi, self.lr, self.max_opt_epochs, grad_mask)
            self.log_kernel_ls = params[0]['log_kernel_ls']
            self.log_kernel_var = params[0]['log_kernel_var']
            self.psi_params = params[1]
            _, _, self.Kzzinv, self.KzziKzx, self.f_var = compute_kernel_matrices(self.Z, self.Xtrain, np.exp(self.log_kernel_ls), np.exp(self.log_kernel_var))

    def predict(self, bags):
        """ 
        Predict the labels of the bags
            bags: list of bags
        """

        if not self.initialized:
            raise Exception("Model not initialized")
        
        X = np.array(bags2instances(bags))
        Bags = bags2index(bags)
        unique_Bags = np.unique(Bags)
        T_prob, y_prob = predict_bags(self.kernel_fn, X, Bags, unique_Bags, self.Z, self.Kzzinv, self.m, self.S, self.data_mean, self.data_std, np.exp(self.log_kernel_ls), np.exp(self.log_kernel_var))

        return T_prob, y_prob
    
# @partial(jit, static_argnums=(0,))
def predict_bags(kernel_fn, X, Bags, unique_Bags, Z, Kzzinv, m, S, data_mean, data_std, kernel_ls, kernel_var):
    """
    Predict the label
        kernel_fn: kernel function
        X; (N,D): matrix of n instances with d features
        Bags: (N,): vector of bag indices
        Z; (M,D): matrix of M inducing points with D features
        Kzzinv; (M,D): inverse of the kernel matrix of the inducing points
        m; (M,): mean of the variational distribution of the inducing points
        kernel_ls: kernel lengthscale
        kernel_var: kernel variance
        data_mean: mean of the training data
        data_std: standard deviation of the training data
    """

    X = (X - data_mean) / data_std

    Kxz = kernel_fn(X, Z, kernel_ls, kernel_var)
    KxzKzzinv = Kxz @ Kzzinv
    mean = KxzKzzinv @ m

    KxzKzzinvS = KxzKzzinv @ S
    
    diag_Kxx = kernel_var * jnp.ones(len(X))
    # diag_KxzKzzinvSKzzinvKzx = jnp.diag( KxzKzzinv @ S @ KxzKzzinv.T )
    diag_KxzKzzinvSKzzinvKzx = jnp.einsum('ij,ji->i', KxzKzzinvS, KxzKzzinv.T)
    # diag_KxzKzzinvKzx = jnp.diag( KxzKzzinv @ Kxz.T )
    diag_KxzKzzinvKzx = jnp.einsum('ij,ji->i', KxzKzzinv, Kxz.T)

    mean = KxzKzzinv @ m
    std = jnp.sqrt( jnp.abs(diag_Kxx + diag_KxzKzzinvSKzzinvKzx - diag_KxzKzzinvKzx) )

    samples = mean + std * np.random.randn(100, len(mean))

    y_prob = jnp.mean(jax.nn.sigmoid(samples), axis=0)
    

    def predict_bag(b, Bags):
        mask = Bags == b
        y_prob_bag = jnp.where(mask, y_prob, 0.0)
        #T_prob_bag = 1.0 - jnp.prod(1.0 - y_prob_bag)
        T_prob_bag = jnp.max(y_prob_bag)
        return T_prob_bag
    
    T_prob = vmap(predict_bag, in_axes=(0, None))(unique_Bags, Bags)

    return T_prob, y_prob

