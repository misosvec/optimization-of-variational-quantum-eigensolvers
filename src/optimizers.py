from typing import List
from qiskit_algorithms.optimizers import *

def init_optimizers(maxiter: int) -> List[Optimizer]:
    adam = ADAM(maxiter=maxiter)
    amsgrad = ADAM(maxiter=maxiter, amsgrad=True)
    cg = CG(maxiter=maxiter)
    cobyla = COBYLA(maxiter=maxiter)
    aqdg = AQGD(maxiter=maxiter)    
    l_bfgs_b = L_BFGS_B(maxiter=maxiter)
    gradient_descent = GradientDescent(maxiter=maxiter)
    nelder_mead = NELDER_MEAD(maxiter=maxiter)
    nft = NFT(maxiter=maxiter)
    powell = POWELL(maxiter=maxiter)
    slsqp = SLSQP(maxiter=maxiter)
    spsa = SPSA(maxiter=maxiter)
    qnspsa = None # this optimizer is dependent on a provided circuit, will be instantiated in the benchmark function
    tnc = TNC(maxiter=maxiter)
    umda = UMDA(maxiter=maxiter)

    return [(aqdg, type(aqdg).__name__),  # Analytical Quantum Gradient Descent with Epochs
            (nft, type(nft).__name__),  # Nakanishi-Fujii-Todo
            (qnspsa, 'QNSPSA'), # Quantum Natural SPSA 
            (spsa, type(spsa).__name__), # Simultaneous Perturbation Stochastic Approximation
            (cobyla, type(cobyla).__name__), # Constrained Optimization By Linear Approximation
            (nelder_mead, type(nelder_mead).__name__), # Nelder-Mead 
            (powell, type(powell).__name__), # Powell algorithm
            (umda, type(umda).__name__), # Continuous Univariate Marginal Distribution Algorithm 
            (gradient_descent, type(gradient_descent).__name__), # Gradient Descent
            (cg, type(cg).__name__), # Conjugate Gradient
            (adam, type(adam).__name__),  # Adaptive Moment Estimation
            (amsgrad, 'AMSGRAD'), # Variant of Adam uses a ‘long-term memory’ of past gradients
            (l_bfgs_b, type(l_bfgs_b).__name__), # Limited-memory BFGS Bound 
            (slsqp, type(slsqp).__name__), # Sequential Least SQuares Programming
            (tnc, type(tnc).__name__), # Truncated Newton 
            ]