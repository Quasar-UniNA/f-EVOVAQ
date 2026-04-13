from tabulate import tabulate
from tqdm import tqdm
from typing import Union


def set_progress_bar(
        max_gen: int, max_nfev: Union[int, None], desc: Union[str, None] = None, unit: Union[str, None] = None) -> tqdm:
    """
    Set the progress bar based on the stopping criterion.

    Args:
        max_gen: Maximum number of generations.
        max_nfev: Maximum number of fitness evaluations.
        desc: Description related to the stopping criterion.
        unit: Unit describing the progress.

    Returns:
        The progress bar.
    """
    if max_nfev is not None:
        pbar = tqdm(total=max_nfev, desc='Fitness Evaluations', unit='nfev', dynamic_ncols=True)
    else:
        desc = desc if desc is not None else 'Generations'
        unit = unit if unit is not None else 'gen'
        pbar = tqdm(total=max_gen, desc=desc, unit=unit, dynamic_ncols=True)
    return pbar


def compute_statistics(fitness, xp) -> dict:
    """
    Compute the statistics of fitness values.

    Args:
        fitness: Fitness values as an array of real values with (`pop_size`, ) shape.
        xp: Numpy or Cupy backend.
     Returns:
        Dictionary containing the min, max, mean, and standard deviation value.
    """
    stats = {'min': xp.min(fitness),
             'max': xp.max(fitness),
             'mean': xp.mean(fitness),
             'std': xp.std(fitness)}
    return stats


def print_info(n_run: int, header: bool = False, **kwargs):
    """
    Print info.

    Args:
        n_run: Independent execution number of the algorithm.
        header: If True, the string indicating the independent execution number is printed.
        kwargs: Information to be printed defined in a dictionary.
    """
    if header:
        print(f'********** Execution #{n_run} **********')
        print(tabulate([kwargs], headers="keys", tablefmt="simple", numalign="left"))
    else:
        formatted_table = [f"{str(item):<{len(k) + 2}}" for k, item in kwargs.items()]
        print("\n" + tabulate([formatted_table], tablefmt="plain"))


class Logbook(dict):
    """
    Class used to store info during the evolution.
    """
    def record(self, **infos):
        """
        Record info.

        Args:
            infos: Info to be stored defined in a dictionary.
        """
        for k, v in infos.items():
            if k in self:
                self[k].append(v)
            else:
                self[k] = [v]

    def get_log(self):
        """
        Get the logbook.
        """
        return self


class BestIndividualTracker:
    """
    Class used to track the best solution ever found during the algorithm execution.

    Args:
        xp: Numpy or Cupy backend.
    """
    def __init__(self, xp):
        self.best_ind = None
        self.best_fit = float('inf')
        self.xp = xp

    def update(self, population, fitness):
        """
        Update the tracker.

        Args:
            population: A population of individuals as array of real parameters with (`pop_size`, `n_params`) shape.
            fitness: A set of fitness values associated to the population as array of real values with (`pop_size`, )
                    shape.
        """
        idx = self.xp.argsort(fitness)[0]
        current_best_fit = fitness[idx]

        # Compare with global best
        if current_best_fit < self.best_fit:
            self.best_fit = current_best_fit
            self.best_ind = population[idx].copy()


    def get_best(self):
        """
        Get the best individual ever found.
        """
        return self.best_ind

    def get_best_fit(self):
        """
        Get the best fitness value ever found.
        """
        return self.best_fit


class FinalResult(dict):
    """
    Class used to store the final result.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())
