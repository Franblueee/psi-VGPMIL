import numpy as np

import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

import optax
import jax

from tqdm import tqdm


# RBF Kernel
@jit
def rbf_kernel(x, y, ls, var):
    return var * jnp.exp(-0.5 * jnp.sum((x - y)**2) / ls)
mv_rbf_kernel = vmap(rbf_kernel, (0, None, None, None), 0)
mm_rbf_kernel = vmap(mv_rbf_kernel, (None, 0, None, None), 1)

# Kernel matrices: Kzz_inv, Kzz_inv_Kzx, f_var
@jit
def compute_kernel_matrices(Z, X, kernel_ls, kernel_var):
    num_ind = Z.shape[0]
    Kzx = mm_rbf_kernel(Z, X, kernel_ls, kernel_var)
    Kzz = mm_rbf_kernel(Z, Z, kernel_ls, kernel_var)
    Kzzinv = jnp.linalg.inv(Kzz + jnp.identity(num_ind) * 1e-6)
    KzziKzx = Kzzinv @ Kzx
    f_var = 1 - jnp.einsum("ji,ji->i", Kzx, KzziKzx)
    return Kzx, Kzz, Kzzinv, KzziKzx, f_var


# Variational updates

@partial(jit, static_argnums=(0))
def update_q_u(theta_fn, m, S, pi, Eff, KzziKzx, Kzzinv, f_prior_var):

    num_ind = KzziKzx.shape[0]
    Theta = theta_fn(jnp.sqrt(Eff))
    Sinv = Kzzinv + (KzziKzx * Theta) @ KzziKzx.T
    S = jnp.linalg.inv(Sinv + np.identity(num_ind) * 1e-8)
    m = S @ KzziKzx @ (pi - 0.5)

    Ef = KzziKzx.T @ m
    mmTpS = jnp.outer(m, m) + S
    Eff = jnp.absolute(jnp.einsum("ij,ji->i", KzziKzx.T @ mmTpS, KzziKzx) + f_prior_var)
    
    return m, S, Ef, Eff

# @jit
def update_q_y(pi, Ef, Bags, unique_Bags, InstBagLabel, lH):
    """
    :param Bags: n-dim vector with the bag index of each instance
    :param pi: n-dim vector with the current value of q(y)
    """
    N = len(pi)
    @jit
    def bag_emax(b, pi):
        mask = Bags == b
        pisub = jnp.where(mask, pi, 0.0)
        max1 = jnp.max(pisub)
        max1_idx = jnp.argmax(pisub)
        tmp = jnp.full_like(pisub, max1)
        pisub = jnp.where(max1_idx == jnp.arange(N), -99.0, pisub)
        #pisub = jnp.where(max1_mask, -99.0, pisub)
        max2 = jnp.max(pisub)
        tmp = jnp.where(max1_idx == jnp.arange(N), max2, tmp)
        return jnp.where(mask, tmp, 0.0)

    Emax = vmap(bag_emax, in_axes=(0, None))(unique_Bags, pi)
    Emax = jnp.sum(Emax, axis=0)
    Emax = jnp.clip(Emax, 0, 1)

    pi = jax.nn.sigmoid(Ef + lH * (2 * InstBagLabel + Emax - 2 * InstBagLabel * Emax - 1))

    return pi

def update_hyperparameters(params, loss_fn, psi_fn, X, Z, m, S, pi, lr, max_epochs, grad_mask):

    tol = 1e-4

    optimizer = optax.sgd(learning_rate=lr, nesterov=True)
    opt_state = optimizer.init(params)

    patience = 10

    best_loss = np.inf
    best_params = params
    new_params = params
    loss = best_loss
    
    pbar = tqdm(range(max_epochs), leave=False)
    for epoch in pbar:
        lambda_val = max( epoch/max_epochs, 0.5)

        new_params, opt_state, loss_dict = gradient_update(opt_state, optimizer.update, loss_fn, psi_fn, params, X, Z, m, S, pi, grad_mask, lambda_val)
        loss = loss_dict['loss']

        if loss < best_loss - tol:
            best_loss = loss
            best_params = params
            patience = 10

        # loss_str = ', '.join([f'{k}: {v:.4f}' for k,v in loss_dict.items()])
        # params_str = ', '.join([f'{params_names[i]}: {new_params[i]:.4f}' for i in range(len(params_names))])
        # print(f'Epoch: {epoch+1}'),
        # print(loss_str)
        # print(params_str)
                
        params = new_params

        pbar.set_postfix({k : f'{v:.4f}' for k, v in loss_dict.items()})

    return best_params

@partial(jit, static_argnums=(0,1,2,3))
def gradient_update(opt_state, opt_update, loss_fn, psi_fn, params, X, Z, m, S, pi, grad_mask, lambda_val=1.0):
    loss_vec, grads = jax.value_and_grad(loss_fn, has_aux=True)(params, psi_fn, X, Z, m, S, pi, lambda_val=lambda_val)
    # grads = jnp.array([grad_mask[i] * grads[i] for i in range(len(grads))])
    grads = jax.tree_map(lambda g, m : m * g, grads, grad_mask)
    updates, opt_state = opt_update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_vec[1]

@partial(jit, static_argnums=(1))
def minus_elbo_psivgpmil(params, psi_fn, X, Z, m, S, pi, lambda_val=1.0, num_fourier_features=100):
    kernel_params = params[0]
    psi_params = params[1]

    log_lengthscale = kernel_params['log_kernel_ls']
    log_variance = kernel_params['log_kernel_var']
    
    M = Z.shape[0]
    N = X.shape[0]

    lengthscale = jnp.exp(log_lengthscale)
    variance = jnp.exp(log_variance)

    # Kxx = mm_rbf_kernel(X, X, lengthscale, variance) # N x N
    Kxz = mm_rbf_kernel(X, Z, lengthscale, variance) # N x M
    Kzz = mm_rbf_kernel(Z, Z, lengthscale, variance) + jnp.identity(M) * 1e-6 # M x M
    Kzzinv = jnp.linalg.inv(Kzz)
    Kxz_Kzzinv = Kxz @ Kzzinv # N x M

    Kxz_Kzzinv_S = Kxz_Kzzinv @ S
    
    diag_Kxx = variance * jnp.ones(len(X))

    # diag_KxzKzzinvSKzzinvKzx = jnp.diag( KxzKzzinv @ S @ KxzKzzinv.T )
    diag_Kxz_Kzzinv_S_Kzzinv_Kzx = jnp.einsum('ij,ji->i', Kxz_Kzzinv_S, Kxz_Kzzinv.T)
    # diag_KxzKzzinvKzx = jnp.diag( KxzKzzinv @ Kxz.T )
    diag_Kxz_Kzzinv_Kzx = jnp.einsum('ij,ji->i', Kxz_Kzzinv, Kxz.T)

    q_f_mean = Kxz_Kzzinv @ m # N 
    # diag_q_f_covar = jnp.diag(Kxx - Kxz_Kzz_inv @ (Kzz - S) @ Kxz_Kzz_inv.T) # N
    diag_q_f_covar = diag_Kxx + diag_Kxz_Kzzinv_S_Kzzinv_Kzx - diag_Kxz_Kzzinv_Kzx # N

    q_f_sample = q_f_mean + jnp.sqrt(jnp.abs(diag_q_f_covar)) * np.random.normal(size=N)

    log_psi_qf = jnp.mean( jnp.log( psi_fn(q_f_sample, **psi_params) ) )
    likelihood = ( (pi - 0.5).T @ q_f_mean) / N + log_psi_qf

    # rff = random_fourier_features(num_fourier_features, X, log_variance, log_lengthscale) # N x num_fourier_features
    # w = np.random.normal(size=(num_fourier_features,1)) # num_fourier_features x 1
    # p_f_sample = rff @ w # N x 1
    # log_psi_pf = jnp.log ( psi_fn(p_f_sample, **psi_params) ) # N x 1
    # log_phi_pf = - jnp.log( jnp.exp(0.5*p_f_sample) + jnp.exp(-0.5*p_f_sample) ) # N x 1
    # logZ = jnp.mean(log_psi_pf - log_phi_pf)
    logZ = 0.0

    kl = (0.5/M) * (jnp.trace(Kzzinv @ S) + m.T @ Kzzinv @ m - M + jnp.linalg.slogdet(Kzz)[1] - jnp.linalg.slogdet(S)[1])
    # print(kl)

    loss = - likelihood + lambda_val*( kl + logZ )
    return loss, {'loss': loss, 'likelihood': -likelihood, 'kl': kl, 'logZ': logZ}

@partial(jit, static_argnums=(0,1))
def random_fourier_features(n_features, X, log_variance, log_lengthscale):
    n_features = n_features//2
    D = X.shape[1]
    W = np.random.normal(size=(D, n_features))
    sqrt_lengthscale = jnp.exp(0.5*log_lengthscale)
    sqrt_variance = jnp.exp(0.5*log_variance)
    return sqrt_variance * jnp.sqrt( 0.5 / n_features ) * jnp.concatenate([jnp.cos(X @ W / sqrt_lengthscale), jnp.sin(X @ W / sqrt_lengthscale)], axis=1)

