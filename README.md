<h1 align='center'>
    Hyperbolic Secant Representation of the Logistic Function: Application to Probabilistic Multiple Instance Learning for CT Intracranial Hemorrhage Detection [<a href="https://doi.org/10.1016/j.artint.2024.104115">AIJ</a>] [<a href="https://arxiv.org/abs/2403.14829">arXiv</a>]<br>
</h1>

### TL;DR

This repository contains the code for the above paper, in which we propose a new model for Multiple Instance Learning based on Gaussian Processes. The model is called $\psi$-VGPMIL, and can be instantiated using any differentiable density $\psi$ that admits a Gaussian Scale Mixture representation. 

<!-- <p align="center">
<img align="middle" src="./img/BCD_net_reduced_ICIP_lateral.png" width="1000" />
</p> -->

### Abstract

Multiple Instance Learning (MIL) is a weakly supervised paradigm that has been successfully applied to many different scientific areas and is particularly well suited to medical imaging. Probabilistic MIL methods, and more specifically Gaussian Processes (GPs), have achieved excellent results due to their high expressiveness and uncertainty quantification capabilities. One of the most successful GP-based MIL methods, VGPMIL, resorts to a variational bound to handle the intractability of the logistic function. Here, we formulate VGPMIL using Pólya-Gamma random variables. This approach yields the same variational posterior approximations as the original VGPMIL, which is a consequence of the two representations that the Hyperbolic Secant distribution admits. This leads us to propose a general GP-based MIL method that takes different forms by simply leveraging distributions other than the Hyperbolic Secant one. Using the Gamma distribution we arrive at a new approach that obtains competitive or superior predictive performance and efficiency. This is validated in a comprehensive experimental study including one synthetic MIL dataset, two well-known MIL benchmarks, and a real-world medical problem. We expect that this work provides useful ideas beyond MIL that can foster further research in the field.

----

### Code organization

The proposed $\psi$-VGPMIL model paper is implemented in `psiVGPMIL.py`. The densities $\psi$ used in our paper are in `psi_functions.py`. The variational updates and the ELBO optimization procedure is implemented in `variational.py`. The training and evaluation procedures are implemented in `engine.py`. The scripts to run the experiments are placed in the `bash_scripts` folder. If you want to reproduce the results of the paper, you will need to download and process the MUSK1, MUKS2 and RSNA datasets. 

See `requirements.txt` for a list of required packages. 

**Note.** This code is highly based on the original VGPMIL implementation, which can be found [here](https://github.com/manuelhaussmann/vgpmil).

**Note.** We have observed that in some cases better results are obtained by removing the $\log Z$ term from the ELBO. This is the reason why we set `logZ=0.0` in `variational.py`.

### Define your own $\psi$-VGPMIL model

If you want to instantiate your own $\psi$-VGPMIL model, you can do so by defining a new density $\psi$ that admits a Gaussian Scale Mixture representation. You will also need to define its derivative. Please note that the JAX autodiff mechanism can be used to compute the derivative of the $\psi$ function. 

Once you have defined them, you can instantiate the model passing these functions as arguments to the psiVGPMIL class. 

```python
from jax import numpy as jnp
from psiVGPMIL import psiVGPMIL

# Define your psi function and its derivative
def psi(x):
    return jnp.exp(-0.5*x**2) / jnp.sqrt(2.0*jnp.pi)

def diff_psi(x):
    return -x * psi(x)

# Instantiate the psiVGPMIL model
model = psiVGPMIL(psi, diff_psi)
```

### Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{castro2024hyperbolic,
  title={Hyperbolic Secant Representation of the Logistic Function: Application to Probabilistic Multiple Instance Learning for CT Intracranial Hemorrhage Detection},
  author={Castro-Mac{\'\i}as, Francisco M and Morales-{\'A}lvarez, Pablo and Wu, Yunan and Molina, Rafael and Katsaggelos, Aggelos K},
  journal={Artificial Intelligence},
  pages={104115},
  year={2024},
  publisher={Elsevier}
}
```