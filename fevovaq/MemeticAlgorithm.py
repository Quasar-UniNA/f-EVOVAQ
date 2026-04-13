from functools import partial
from typing import Callable
from .problem import Problem
from .tools.operators import sel_random
from .tools.support import compute_statistics, print_info, Logbook, set_progress_bar, BestIndividualTracker, \
    FinalResult




class MA(object):
    """
    Memetic Algorithm (MA) [1] is an evolutionary approach that merge population- and local-based methods to improve
    exploration and exploitation capabilities in visiting the problem search space.
    In `evovaq`, MA workflow described in [2] is implemented.

    References:
        [1] P. Moscato, et al., "On evolution, search, optimization, genetic algorithms and martial arts: towards memetic
        algorithms", Caltech concurrent computation program, C3P Report, vol. 826, pp. 37, 1989.

        [2] Giovanni Acampora, Angela Chiatto, and Autilia Vitiello, "Training circuit-based quantum classifiers through
        memetic algorithms", Pattern Recognition Letters, Elsevier, 2023.

    Args:
        global_search: Population-based method used to evolve the population of individuals. The global search function
                       is defined as ``global_search(prob, pop, fitness, gen, *args) -> (offspring, fit_offspring, nfev)``, where
                       ``prob`` is :class:`~.Problem` to be solved; ``pop`` and ``fitness`` are two arrays of real parameters
                       with (`pop_size`,`n_params`) and (`n_params`, ) shape, respectively; ``gen`` is the current
                       generation number; ``args`` is a tuple of other fixed parameters needed to specify the function;
                       and the output is the resulting offspring and fitness values  with the same shape as ``pop`` and
                       ``fitness``, and number of fitness evaluations completed during the evolution. Here, it is
                       possible to use :class:`~.GA` or :class:`~.DE` as a population-based method via
                       :meth:`~DE.evolve_population` method.
        local_search: Local search method used to improve a subset of individuals. The local search function is defined
                      as ``local_search(prob, ind, fitness, *args) -> (ind, fitness, nfev)``, where ``prob`` is
                      :class:`~.Problem` to be solved; ``ind``  is an array of real parameters with (`n_params`,);
                      ``fitness`` is a float value; ``args`` is a tuple of other fixed parameters needed to specify the
                      function; and the output is the improved individual and fitness value with the same shape as
                      ``ind`` and ``fitness``, and number of fitness evaluations completed during the local research.
                      Here, it is possible to use :class:`~.HC` as a local search method via :meth:`~HC.stochastic_var`
                      method.
        sel_for_refinement: Selection operator used to choose the individuals undergoing local refinement.
        frequency: Individual learning frequency defined in the range (0 , 1) influencing the number of individuals that
                   is undergone to local refinement.
        intensity: Individual learning intensity representing the maximum computational budget allowable for individual
                   learning to expend on improving a single solution. Here, it corresponds to the maximum number
                   of iterations to be performed during the local search.
        elitism: If True, the best solution of current population is transferred directly into the next generation.
    """
    def __init__(
            self,
            global_search: Callable,
            local_search: Callable,
            sel_for_refinement: Callable,
            frequency: float,
            intensity: int,
            elitism: bool = True
    ):
        self.global_search = global_search
        self.local_search = local_search
        self.sel_for_refinement = sel_for_refinement
        self.frequency = frequency
        self.intensity = intensity
        self.elitism = elitism

        if self.sel_for_refinement == sel_random:
            self.sel_for_refinement = partial(sel_random, replace=False)

    def optimize(self, problem: Problem, pop_size: int, initial_pop=None,
                 max_nfev=None, max_gen=1000, num_run=1, seed=None, verbose=True) -> FinalResult:
        """
        Optimize the parameters of the problem to be solved.

        Args:
            problem: :class:`~.Problem` to be solved.
            pop_size: Population size.
            initial_pop: Initial population of possible solutions as array of real parameters with (`pop_size`, `n_params`)
                         shape. If None, the initial population is randomly generated from `param_bounds`.
             max_nfev: Maximum number of fitness evaluations. If not None, this is considered as stopping criterion.
            max_gen: Maximum number of generations. If `max_nfev` is None, this is considered as stopping criterion.
            num_run: Independent execution number of the algorithm.
            seed: Initialize the random number generator. If None, the current time is used.
            verbose: If True, the statistics of fitness values is printed during the evolution.

        Returns:
            A :class:`~.FinalResult` containing the optimization result.
        """

        xp = problem.xp

        if seed is not None:
            xp.random.seed(seed)

        gen = 0
        population = xp.array(initial_pop) if initial_pop is not None else problem.generate_random_pop(pop_size)
        fitness = problem.evaluate_fitness(population)

        tot_nfev = population.shape[0]

        best_tracker = BestIndividualTracker(xp)
        best_tracker.update(population, fitness)

        pbar = set_progress_bar(max_gen, max_nfev)
        if max_nfev is not None:
            pbar.update(tot_nfev)

        stats = compute_statistics(fitness, xp)
        lg = Logbook()
        lg.record(gen=gen, nfev=tot_nfev, **stats)

        if verbose: print_info(n_run=num_run, gen=gen, nfev=tot_nfev, **stats, header=True)

        for gen in range(1, max_gen + 1):
            if max_nfev is not None and tot_nfev >= max_nfev:
                break

            if self.elitism:
                idx_best = xp.argmin(fitness)
                elite_x = population[idx_best].copy()
                elite_f = fitness[idx_best].copy()

            offspring, fit_offspring, glob_nfev = self.global_search(problem, population, fitness, gen)
            tot_nfev += glob_nfev
            nfev = glob_nfev

            omega_idx = self.sel_for_refinement(offspring, fit_offspring, int(self.frequency * pop_size), xp)

            for ind_idx in omega_idx:
                ind = offspring[ind_idx]
                fit = fit_offspring[ind_idx]
                for _ in range(self.intensity):
                    ind, fit, loc_nfev = self.local_search(problem, ind, fit)
                    tot_nfev += loc_nfev
                    nfev += loc_nfev

            if self.elitism:
                idx_worst_offspring = xp.argmax(fit_offspring)
                offspring[idx_worst_offspring] = elite_x
                fit_offspring[idx_worst_offspring] = elite_f

            population[:] = offspring
            fitness[:] = fit_offspring

            best_tracker.update(population, fitness)
            stats = compute_statistics(fitness, xp)
            lg.record(gen=gen, nfev=nfev, **stats)

            if verbose: print_info(n_run=num_run, gen=gen, nfev=nfev, **stats)
            pbar.update(nfev if max_nfev else 1)

        pbar.close()
        return FinalResult(x=best_tracker.get_best(), fun=best_tracker.get_best_fit(),
                           nfev=tot_nfev, gen=gen, log=lg.get_log())
