from typing import Union
from .problem import Problem
from .tools.support import compute_statistics, print_info, Logbook,set_progress_bar, BestIndividualTracker, FinalResult


class PSO(object):
    """
    Particle Swarm Optimization (PSO) is an optimization method based on a swarm of candidate solutions, named particles,
    moving in the search space according to appropriate position and velocity equation [1]. Starting from a swarm of
    particles with random positions and velocities, each particle’s velocity is updated by combining its own best
    position (pbest) and the global best position (gbest) ever found in the search space with some random perturbations
    influenced by two hyperparameters, denoted as `phi1` and `phi2`. At this point, each particle’s position is updated
    by adding its resulting velocity to the current position. The final goal is to move the whole swarm close to an
    optimal position in the search space.
    In `fevovaq`, PSO with inertia weight (`inertia_weight`) described in [2] is implemented.

    References:
        [1] Giovanni Acampora, Angela Chiatto, and Autilia Vitiello, "A comparison of evolutionary algorithms for
        training variational quantum classifiers", Proceedings of 2023 IEEE Congress on Evolutionary Computation (CEC),
        pp. 1–8, 2023.

        [2] Poli R., Kennedy J., & Blackwell T., "Particle swarm optimization: An overview", Swarm intelligence, vol. 1,
        pp. 33-57, 2007.

    Args:
        vmin: Lower value(s) of the velocity. If None, no limits are considered.
        vmax: Upper value(s) of the velocity. If None, no limits are considered.
        inertia_weight: Inertia weight.
        phi1: Acceleration coefficient determining the magnitude of the random force in the direction of personal
              best solution.
        phi2: Acceleration coefficient determining the magnitude of the random force in the direction of global best
              solution.
        """
    def __init__(
            self,
            vmin: Union[float, None] = None,
            vmax: Union[float, None] = None,
            inertia_weight: float = 0.7298,
            phi1: float = 1.49618,
            phi2: float = 1.49618
    ):
        self.vmin = vmin
        self.vmax = vmax
        self.inertia_weight = inertia_weight
        self.phi1 = phi1
        self.phi2 = phi2

        self.velocity = None
        self.pbest_pos = None
        self.pbest_fit = None
        self.gbest_pos = None
        self.gbest_fit = None

    def evolve_population(self, problem, population, fitness, gen):
        """
        Evolve the population by iteratively updating individual positions and velocities using swarm equations inspired
        by social behavior.

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

        if gen == 1:
            self.velocity = xp.random.uniform(-1, 1, (pop_size, problem.n_params))
            self.pbest_pos = population.copy()
            self.pbest_fit = fitness.copy()

            best_idx = xp.argmin(fitness)
            self.gbest_pos = population[best_idx].copy()
            self.gbest_fit = fitness[best_idx].copy()

        r = xp.random.random((2, pop_size, dim))
        r1 = r[0] * self.phi1
        r2 = r[1] * self.phi2

        velocity = (
                self.inertia_weight * self.velocity
                + r1 * (self.pbest_pos - population)
                + r2 * (self.gbest_pos - population)
        )

        if self.vmin is not None or self.vmax is not None:
            velocity = xp.clip(velocity, self.vmin, self.vmax)

        offspring = population + velocity
        problem.check_bounds(offspring)
        fit_offspring = problem.evaluate_fitness(offspring)

        improved = fit_offspring < self.pbest_fit

        self.pbest_pos = xp.where(improved[:, None], offspring, self.pbest_pos)
        self.pbest_fit = xp.where(improved, fit_offspring, self.pbest_fit)

        best_idx = xp.argmin(self.pbest_fit)
        best_fit = self.pbest_fit[best_idx]
        best_pos = self.pbest_pos[best_idx]

        better = best_fit < self.gbest_fit

        self.gbest_fit = xp.where(better, best_fit, self.gbest_fit)
        self.gbest_pos = xp.where(better, best_pos, self.gbest_pos)

        self.velocity = velocity

        return offspring, fit_offspring, pop_size

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
