import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from utils import (
    create_bootstrap_indices_and_Nbi,
    bagging_decision_trees,
    generate_data,
    inf_JK_bagged_variance,
    simulate_bagging_and_variance,
    save_results_png,
)


def main():

    # Constants
    CHUNK_SIZE = 250
    ijk = False
    fix_x_points = True
    
    # Simulation parameters
    n = 500
    n_sim = 1_000
    B = 1000 # Paper uses 10_000
    args = {"max_leaf_nodes": 5}
    seed = 45
    new_data = np.linspace(0, 1, n)

    # Arrays to store the predictions and estimated variances
    bagged_preds = np.zeros((n_sim, new_data.shape[0]))
    est_vars = np.zeros((n_sim, new_data.shape[0]))

    # Parallelize simulations with progress bar
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                simulate_bagging_and_variance,
                n=n,
                B=B,
                new_data=new_data,
                simulation_index=i,
                seed=seed,
                dt_args=args,
                fix_x_points=fix_x_points,
                ijk_calculation=ijk,
                chunk_size=CHUNK_SIZE
            )
            for i in range(n_sim)
        ]

        for i, future in enumerate(
            tqdm(futures, desc="Simulations", unit="simulation")
        ):
            bagged_prediction, est_variance = future.result()
            bagged_preds[i, :] = bagged_prediction
            est_vars[i, :] = est_variance

    # Calculate true variance of bagged predictions
    true_vars = bagged_preds.var(axis=0, ddof=1)
    est_vars_mean = est_vars.mean(axis=0)
    est_vars_std = est_vars.std(axis=0, ddof=1)

    print(f"Mean true variance: {round(np.mean(true_vars), 10)}")
    print(f"Mean estimated variance: {round(np.mean(est_vars_mean), 10)}")
    print(f"Min estimated variance: {round(np.min(est_vars), 10)}")

    save_results_png(
        new_data=new_data,
        true_variances=true_vars,
        est_variances_mean=est_vars_mean,
        est_variances_std=est_vars_std,
        n_data_points=n,
        n_simulations=n_sim,
        B=B,
        seed=seed,
        dt_args=args,
        fixed_x_points=fix_x_points,
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- runtime: %s minutes ---" % round((time.time() - start_time) / 60, 2))
