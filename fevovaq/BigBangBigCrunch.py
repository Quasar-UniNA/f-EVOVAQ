from .problem import Problem
from .tools.support import compute_statistics, print_info, Logbook, set_progress_bar, BestIndividualTracker, \
    FinalResult




class BBBC(object):
    """
    The Big Bang - Big Crunch (BBBC) method as developed by Erol and Eks in 2006 [1] consists of two alternating steps:
    1) a Big Bang phase, where the candidate solutions are randomly distributed over the search space; 2) a Big Crunch
    phase, where a contraction operation estimates a weighted average, denoted as Centre of Mass, of the randomly
    distributed candidate solutions. During the Big Bang phases, new candidate solutions are generated considering the
    Center of Mass and the best global solution, as introduced in [2].

    References:
        [1] O. K. Erol and I. Eksin, “A new optimization method: big bang–big crunch”, Advances in Engineering Software,
        vol. 37, no. 2, pp. 106– 111, 2006.

        [2] C. V. Camp, “Design of space trusses using big bang–big crunch optimization,” Journal of Structural
        Engineering, vol. 133, no. 7, pp. 999–1008, 2007.

    Args:
        elitism: If True, the best solution of current population is transferred directly into the next generation.
        alpha: Hyperparameter limiting the size of the search space.
        beta: Hyperparameter defined in range (0,1) controlling the influence of the best individual on the location of
              new candidate solutions.
    """

    def __init__(
            self,
            elitism: bool = True,
            alpha: float = 10.0,
            beta: float = 0.25
    ):
        self.elitism = elitism
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def _compute_centre_of_mass(population, fitness, xp):
        """
        Compute the centre of mass.
        """
        return xp.sum(population / fitness[:, None], axis=0) / xp.sum(1 / fitness)

    def evolve_population(self, problem, population, fitness, gen):
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
        pop_size, dim = population.shape

        best_idx = xp.argmin(fitness)
        best = population[best_idx]

        com = self._compute_centre_of_mass(population, fitness, xp)

        diff = problem.bounds_max - problem.bounds_min

        r = xp.random.normal(0, 1, size=(pop_size, dim))

        offspring = self.beta * com + (1 - self.beta) * best + diff * r * self.alpha / gen
        problem.check_bounds(offspring)
        fit_offspring = problem.evaluate_fitness(offspring)

        return offspring, fit_offspring, pop_size

    def optimize(self, problem: Problem, pop_size: int, initial_pop=None,
            max_nfev=None, max_gen=1000, num_run=1, seed=None, verbose=True) -> FinalResult:
        """
        Optimize the parameters of the problem to be solved.

        Args:
            problem: :class:`~.Problem` to be solved.
            pop_size: Population size.
            initial_pop: Initial population of possible solutions as array of real parameters with (`pop_size`, `n_params`)
                         shape. If None, the initial population is randomly generated from `param_bounds`.
            max_nfev: Maximum number of fitness evaluations. If not None, the maximum number this  is considered as stopping criterion.
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
        lg.record(gen=0, nfev=tot_nfev, **stats)

        if verbose: print_info(n_run=num_run, gen=0, nfev=tot_nfev, **stats, header=True)

        for gen in range(1, max_gen + 1):
            if max_nfev is not None and tot_nfev >= max_nfev:
                break

            if self.elitism:
                idx_best = xp.argmin(fitness)
                elite_x = population[idx_best].copy()
                elite_f = fitness[idx_best].copy()

            offspring, fit_offspring, nfev = self.evolve_population(problem, population, fitness, gen)
            tot_nfev += nfev

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
