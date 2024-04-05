import multiprocessing as mp
import pandas as pd
import numpy as np
from optimizers import init_optimizers
from typing import List, Tuple
from qiskit_algorithms.optimizers import Optimizer, QNSPSA, GSLS
from qiskit_algorithms import VQE
from ansatz import Ansatz, get_basic_ansatzes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator, Sampler
import time

c0 = -0.80718
c1 = 0.17374
c2 = -0.23047
c3 = 0.12149
c4 = 0.16940
c5 = -0.04509
c6 = 0.04509
c7 = 0.16658
c8 = 0.17511

hamiltonian = SparsePauliOp.from_list([
    ("IIII", c0),
    ("ZIII", c1),
    ("ZZII", c2),
    ("IIZI", c1),
    ("IZZZ", c2),
    ("IZII", c3),
    ("ZIZI", c4),
    ("XZXI", c5),
    ("XIXZ", c6),
    ("XIXI", c6),
    ("XZXZ", c5),
    ("ZZZZ", c7),
    ("ZZZI", c7),
    ("ZIZZ", c8),
    ("IZIZ", c3),
])

CPU_COUNT = mp.cpu_count()
SAMPLE_COUNT = 50
MAXITER = 100
seeds = [408741, 456529, 946492, 828685, 373666, 401855, 252083, 133852,
         688322,  43720, 923216, 840855, 110523, 995131, 539693, 833840,
         965173, 349809, 132510,  73543, 901210, 504157, 163085, 349753,
         483948, 119799, 213310,  39704, 764583, 667525, 519714, 837559,
         735186, 332061, 489257, 165523, 688646, 284265, 590176, 579704,
         773366, 207835, 845618, 668644, 847497, 541653, 897007, 893382,
         664063, 135086]

optimizers = init_optimizers(MAXITER)
ansatzes = get_basic_ansatzes(
    num_layers=1) + get_basic_ansatzes(num_layers=2) + get_basic_ansatzes(num_layers=3)


def benchmark_optimizer(optimizer: Tuple[Optimizer, str], ans: List[Ansatz]):
    data = []
    opt = optimizer[0]
    opt_name = optimizer[1]
    for i in range(SAMPLE_COUNT):
        print(f"\rBenchmarking optimizer: {opt_name}, i = {i}\n", flush=True)
        for ansatz in ans:
            estimator = Estimator()
            np.random.seed(seeds[i])
            initial_point = np.random.uniform(-2 *
                                              np.pi, 2*np.pi, ansatz.param_count)
            if opt_name == 'QNSPSA':
                opt = QNSPSA(QNSPSA.get_fidelity(
                    circuit=ansatz.circuit, sampler=Sampler()), maxiter=MAXITER)

            energies = []
            evals = []

            def store_intermediate_result(eval_count, parameters, mean, std):
                evals.append(eval_count)
                energies.append(mean)

            vqe = VQE(estimator, ansatz.circuit, opt,
                      callback=store_intermediate_result, initial_point=initial_point)
            result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
            data.append((i,
                         opt_name,
                         ansatz.entanglement,
                         ansatz.cnot_count,
                         ansatz.depth,
                         ansatz.param_count,
                         ansatz.num_layers,
                         result.optimal_value,
                         result.optimal_point,
                         initial_point,
                         result.cost_function_evals,
                         result.optimizer_evals,
                         result.optimizer_time,
                         opt.is_initial_point_ignored,
                         opt.is_gradient_ignored,
                         opt.is_bounds_ignored,
                         list(zip(evals, energies))))

    return data


def flatten(xss):
    return [x for xs in xss for x in xs]


if __name__ == '__main__':
    with mp.Pool(CPU_COUNT) as p:
        start = time.time()
        results = p.starmap(benchmark_optimizer, [
                            (optimizer, ansatzes) for optimizer in optimizers])
        df = pd.DataFrame(flatten(results),
                          columns=['sample_id',
                                   'optimizer',
                                   'entanglement',
                                   'cnot_count',
                                   'depth',
                                   'param_count',
                                   'num_layers',
                                   'optimal_value',
                                   'optimal_point',
                                   'initial_point',
                                   'cost_function_evals',
                                   'optimizer_evals',
                                   'optimizer_time',
                                   'is_initial_point_ignored',
                                   'is_gradient_ignored',
                                   'is_bounds_ignored',
                                   'energy_convergence'])
        df.to_csv('data.csv')
        end = time.time()
        print(f'elapsed time = {end-start}')
