from typing import Callable
from functools import partial
from .problem import Problem
from .tools.operators import sel_permutation, sel_best, CROSSOVER_ARGS
from .tools.support import compute_statistics, print_info, Logbook, set_progress_bar, BestIndividualTracker, \
    FinalResult




class CHC(object):
    """
    Cross generational elitist selection, Heterogeneous recombination, and Cataclysmic mutation (CHC) algorithm is a
    nontraditional genetic algorithm which combines a conservative selection strategy that always
    preserves the best individuals found so far with a radical (highly disruptive) crossover operator that produces
    offspring that are maximally different from both parents [1]. In detail, it is based on four main components:
    a elitist selection, a highly disruptive crossover, an incest prevention check to avoid the recombination of similar
    solutions, and a population reinitialization method when the population has converged.
    In `evovaq`, a real-coded CHC version based on [2-3] is implemented. However, the similarity is measured between
    fitness values and not between individuals, since different solutions can lead to equal or close cost values when
    training variational quantum circuits.

    References:
        [1] Larry J. Eshelman, "The CHC Adaptive Search Algorithm: How to Have Safe Search When Engaging in
        Nontraditional Genetic Recombination", Foundations of Genetic Algorithms, Elsevier, vol. 1, pp. 265-283, 1991.

        [2] O. Cordón, S. Damas, J. Santamaría, "Feature-based image registration by means of the CHC evolutionary algorithm",
        Image and Vision Computing, vol. 24, Issue 5, pp. 525-533, 2006.

        [3] Cuéllar, M. P., Gómez-Torrecillas, J., Lobillo, F. J., & Navarro, G., "Genetic algorithms with
        permutation-based representation for computing the distance of linear codes", Swarm and Evolutionary Computation,
        vol. 60, pp. 100797, 2021.

    Args:
        crossover: Crossover operator used to mate parents.
        distance: Distance metric used to evaluate the similarity between parents. Select a distance function from
        `~.tools.distances`or customize your distance function by considering as function input (population, fitness, xp).
        multiplier: Factor influencing the initial crossover threshold.
        dec_percentage:  Crossover threshold update rate.
        kwargs: Additional keyword arguments used to set hyperparameter values of crossover operator.
    """
    def __init__(
            self,
            crossover: Callable,
            distance: Callable,
            multiplier: float = 1,
            dec_percentage: float = 0.1, **kwargs
    ):
        self.distance = distance
        self.multiplier = multiplier
        self.dec_percentage = dec_percentage
        self.kwargs = kwargs
        self.thr =None
        self.dec = None

        if crossover in CROSSOVER_ARGS:
            args = {arg: self.kwargs.get(arg) for arg in CROSSOVER_ARGS[crossover] if arg in self.kwargs.keys()}
            self.crossover = partial(crossover, **args)
        else:
            self.crossover = crossover

    def incest_prevention_mask(self, population, fitness, xp):
        """
        Incest prevention check. Before mating, the similarity between the potential parents is calculated, and if this
        distance does not exceed the threshold `thr`, they are not mated.
        """
        dist = self.distance(population, fitness, xp)
        return dist > self.thr

    def initialize_cx_threshold(self, population, fitness, xp):
        """
        Initialize the crossover threshold and compute the decrement value.
        """
        n, dim = population.shape

        i, j = xp.triu_indices(n, k=1)

        paired_pop = xp.empty((2 * len(i), dim), dtype=population.dtype)
        paired_fit = xp.empty(2 * len(i), dtype=fitness.dtype)

        paired_pop[0::2] = population[i]
        paired_pop[1::2] = population[j]

        paired_fit[0::2] = fitness[i]
        paired_fit[1::2] = fitness[j]

        D = self.distance(paired_pop, paired_fit, xp)

        avg_d = xp.mean(D)
        max_d = xp.max(D)
        return avg_d * self.multiplier, max_d * self.dec_percentage

    def evolve_population(self, problem: Problem, population, fitness, gen):
        """
       Evolve the population by means of genetic operators.

       Args:
           problem : :class:`~.Problem` to be solved.
           population: A population of individuals as array of real parameters with (`pop_size`, `n_params`)
                       shape.
           fitness: A set of fitness values associated to the population as array of real values with (`pop_size`, )
                   shape.
           gen: Current generation number.

       Returns:
           The offspring and fitness values obtained after evolution, and number of fitness evaluations completed
           during the evolution.
       """
        xp = problem.xp
        pop_size = population.shape[0]
        nfev = 0

        if gen == 1:
            self.thr, self.dec = self.initialize_cx_threshold(population, fitness, xp)

        chosen_indices = sel_permutation(population, fitness, xp)
        parents = population[chosen_indices]
        fit_parents = fitness[chosen_indices]

        cx_mask = self.incest_prevention_mask(parents, fit_parents, xp)

        if xp.any(cx_mask):
            _offspring = self.crossover(parents, cx_mask, xp)
            valid_mask = xp.repeat(cx_mask, 2)

            _offspring = _offspring[valid_mask]
            problem.check_bounds(_offspring)
            _fit_offspring = problem.evaluate_fitness(_offspring)

            nfev += _offspring.shape[0]

            _joint = xp.concatenate((population, _offspring), axis=0)
            _joint_fitness = xp.concatenate((fitness, _fit_offspring), axis=0)

            sorted_idx = xp.argsort(_joint_fitness)
            improved = xp.any(sorted_idx[:pop_size] >= pop_size)

            offspring = _joint[sorted_idx[:pop_size]]
            fit_offspring = _joint_fitness[sorted_idx[:pop_size]]

        else:
            improved = False

        if not improved:
            self.thr -= self.dec
            if self.thr <= 0:
                best_idx = sel_best(population, fitness, 1, xp)
                best = population[best_idx]
                best_fit = fitness[best_idx]
                rand_pop = problem.generate_random_pop(pop_size - 1)
                rand_fitness = problem.evaluate_fitness(rand_pop)

                nfev += rand_pop.shape[0]

                offspring = xp.concatenate((best, rand_pop))
                fit_offspring = xp.concatenate((best_fit, rand_fitness))

                self.thr, self.dec = self.initialize_cx_threshold(offspring, fit_offspring, xp)
            else:
                offspring = population.copy()
                fit_offspring = fitness.copy()

        return offspring, fit_offspring, nfev

    def optimize(self, problem: Problem, pop_size: int, initial_pop=None,
            max_nfev=None, max_gen=1000, num_run=1, seed=None, verbose=True) -> FinalResult:
        """
        Optimize the parameters of the problem to be solved.

        Args:
            problem: :class:`~.Problem` to be solved.
            pop_size: Population size.
            initial_pop: Initial population of possible solutions as array of real parameters with (`pop_size`, `n_params`)
                         shape. If None, the initial population is randomly generated from `param_bounds`.
            max_nfev: Maximum number of fitness evaluations used as stopping criterion. If None, the maximum number of
                      generations `max_gen` is considered as stopping criterion.
            max_gen: Maximum number of generations used as stopping criterion. If `max_nfev` is not None, this is
                     considered as stopping criterion.
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

            offspring, fit_offspring, nfev = self.evolve_population(problem, population, fitness, gen)
            tot_nfev += nfev

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

