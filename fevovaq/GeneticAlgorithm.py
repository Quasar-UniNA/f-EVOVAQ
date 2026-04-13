from typing import Callable
from functools import partial
from .problem import Problem
from .tools.operators import SELECTION_ARGS, CROSSOVER_ARGS, MUTATION_ARGS
from .tools.support import compute_statistics, print_info, Logbook, set_progress_bar, BestIndividualTracker, \
    FinalResult




class GA(object):
    """
    Genetic Algorithm (GA) is the simplest evolutionary algorithm inspired by Darwinian evolution principles: the evolution
    of a population of possible solutions to a given problem is linked to the concepts of randomness and survival of the
    fittest [1]. In detail, during an evolutionary cycle, the processes of selection, crossover and mutation, known as
    stochastic genetic operators, take place in order to produce the next generation of solutions. The evolution process
    stops when a maximum number of generations or fitness evaluations is reached. Typically, during the evolution
    process, the best individual of the current generation can be inserted into the next one in order to prevent its
    possible disappearance [2].

    References:
        [1] Giovanni Acampora, Angela Chiatto, and Autilia Vitiello, "Training variational quantum circuits through
        genetic algorithms", Proceedings of 2022 IEEE Congress on Evolutionary Computation (CEC), pp. 1–8, 2022.

        [2] Giovanni Acampora, Angela Chiatto, and Autilia Vitiello, "Genetic algorithms as classical optimizer for the
        quantum approximate optimization algorithm", Applied Soft Computing, pp. 110296, Elsevier, 2023.

    Args:
        selection: Selection operator used to select individuals to create the mating pool.
        crossover: Crossover operator used to mate parents.
        mutation: Mutation operator used to mutate individuals.
        elitism: If True, the best solution of current population is transferred directly into the next generation.
        cxpb: The probability of mating two individuals.
        mutpb: The probability of mutating an individual.
        kwargs: Additional keyword arguments used to set hyperparameter values of genetic operators.
        """
    def __init__(
        self,
        selection: Callable,
        crossover: Callable,
        mutation: Callable,
        elitism: bool = True,
        cxpb: float = 0.8,
        mutpb: float = 1.0, **kwargs
    ):
        self.elitism = elitism
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.kwargs = kwargs
        self.selection = self._wrap_operator(selection, SELECTION_ARGS)
        self.crossover = self._wrap_operator(crossover, CROSSOVER_ARGS)
        self.mutation = self._wrap_operator(mutation, MUTATION_ARGS)

    def _wrap_operator(self, op, args_registry):
        """
        Set the hyperparameter values for the genetic operators.
        """
        if op in args_registry:
            args = {k: self.kwargs[k] for k in args_registry[op] if k in self.kwargs}
            return partial(op, **args)
        return op

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

        chosen_indices = self.selection(population, fitness, pop_size, xp)

        mating_pool = xp.take(population, chosen_indices, axis=0)
        fit_offspring = xp.take(fitness, chosen_indices, axis=0)

        cx_mask = xp.random.random(pop_size // 2) < self.cxpb
        offspring = self.crossover(mating_pool, cx_mask, xp)

        mut_mask = xp.random.random(pop_size) < self.mutpb
        offspring = self.mutation(offspring, mut_mask, xp)

        problem.check_bounds(offspring)

        changed = mut_mask | xp.repeat(cx_mask, 2)

        nfev = int(xp.sum(changed).item())
        if nfev > 0:
            fit_offspring[changed] = problem.evaluate_fitness(offspring[changed])

        return offspring, fit_offspring, nfev

    def optimize(self, problem: Problem, pop_size: int, initial_pop=None,
                 max_nfev=None, max_gen=1000, num_run=1, seed=None, verbose=True) -> FinalResult:
        """
        Optimize the parameters of the problem to be solved.

        Args:
            problem: :class:`~.Problem` to be solved.
            pop_size: Population size.
            initial_pop: Initial population of possible solutions as array of real parameters with (`pop_size`, `n_params`)
                         shape. If None, the initial population is randomly generated from `init_range` in :class:`~.Problem`.
            max_nfev: Maximum number of fitness evaluations. If not None, this is considered as stopping criterion.
            max_gen: Maximum number of generations. If `max_nfev` is None, this is considered as stopping criterion.
            num_run: Independent execution number of the algorithm.
            seed: Initialize the random number generator. If None, the default method is used.
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
