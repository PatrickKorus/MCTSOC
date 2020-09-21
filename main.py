from multiprocessing import Pool
from tqdm import tqdm
from mcts_general_experiments.all_experiments import get_all_experiments
from mcts_general_experiments.mcts_experiments_api import get_full_experiments_queue, run_experiment_and_store_results

NUM_THREADS = 4


def get_process_subset(list, process_id, number_of_processes):
    return list[process_id:-1:number_of_processes]


def worker(process_id):
    print("Process {} starting...", process_id)
    all_experiments = get_all_experiments()
    experiments_queue = get_full_experiments_queue(
        experiments=all_experiments,
        num_of_seeds=3,
        num_simulations_list=[10, 50, 100, 200, 400, 800, 1600, 3200]
    )
    experiments_queue_process_subset = get_process_subset(experiments_queue, process_id, NUM_THREADS)
    for experiment in tqdm(experiments_queue_process_subset):
        run_experiment_and_store_results(experiment)


def main(number_of_processes):

    # for proc_id in range(4):
    #     worker(proc_id)
    with Pool(number_of_processes) as pool:
        pool.map(func=worker, iterable=[0, 1, 2, 3])


if __name__ == '__main__':
    main(4)
