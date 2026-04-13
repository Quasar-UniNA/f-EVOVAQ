# ---------- SELECTION OPERATORS ------------
def sel_best(population, fitness, k, xp):
    """
    Selection of the first `k` best individuals (with the smallest objective values) in the population.

    Args:
        population: A population of individuals as array of real parameters with (`pop_size`, `n_params`) shape.
        fitness: A set of fitness values associated to the population as array of real values with (`pop_size`, )
                shape.
        k: Number of individuals to be selected.
        xp: Numpy or Cupy backend.

    Returns:
        Indices of selected individuals.
    """
    chosen_indices = xp.argsort(fitness)
    return chosen_indices[:k]


def sel_permutation(population, fitness, xp):
    """
    Selection of the individuals by permuting the population.

    Args:
        population: A population of individuals as array of real parameters with (`pop_size`, `n_params`) shape.
        fitness: A set of fitness values associated to the population as array of real values with (`pop_size`, )
                shape.
        xp: Numpy or Cupy backend.

    Returns:
        Indices of selected individuals.
    """
    chosen_indices = xp.random.permutation(len(population))
    return chosen_indices


def sel_tournament(population, fitness, k, xp, tournament_size=3):
    """
    Selection of the best individual among `tournament_size` randomly chosen individuals, `k` times.

    Args:
        population: A population of individuals as array of real parameters with (`pop_size`, `n_params`) shape.
        fitness: A set of fitness values associated to the population as array of real values with (`pop_size`, )
                shape.
        k: Number of individuals to be selected.
        xp: Numpy or Cupy backend.
        tournament_size: Size of the tournament.
    Returns:
        Indices of selected individuals.
    """
    pop_size = population.shape[0]

    aspirants = xp.random.randint(0, pop_size, size=(k, tournament_size))
    aspirant_fitness = fitness[aspirants]

    winners_rel = xp.argmin(aspirant_fitness, axis=1)
    chosen_indices = aspirants[xp.arange(k), winners_rel]
    return chosen_indices

def sel_random(population, fitness, k, xp, replace=True):
    """
    Selection of `k` individuals randomly.

    Args:
        population: A population of individuals as array of real parameters with (`pop_size`, `n_params`) shape.
        fitness: A set of fitness values associated to the population as array of real values with (`pop_size`, )
                shape.
        k: Number of individuals to be selected.
        xp: Numpy or Cupy backend.
        replace: Whether the sample is with or without replacement. Default is True, meaning that an element can be
                 selected multiple times.

    Returns:
        Indices of selected individuals.
    """
    chosen_indices = xp.random.choice(len(population), (k,), replace=replace)
    return chosen_indices


# ---------- CROSSOVER OPERATORS ------------
def cx_blx_alpha(parents, cx_mask, xp, alpha=0.5):
    """
    BLX-alpha crossover.

    Args:
        parents: Parents as arrays of real parameters with (`pop_size`,`n_params`) shape.
        cx_mask: Crossover mask.
        xp: Numpy or Cupy backend.
        alpha: Positive real hyperparameter.

    Returns:
        Resulting offspring.
    """
    p1 = parents[0::2]
    p2 = parents[1::2]
    offspring = xp.empty_like(parents)

    d = xp.abs(p1 - p2)
    low = xp.minimum(p1, p2) - alpha * d
    high = xp.maximum(p1, p2) + alpha * d

    r1 = xp.random.random(low.shape)
    r2 = xp.random.random(low.shape)
    c1 = low + r1 * (high - low)
    c2 = low + r2 * (high - low)

    offspring[0::2] = xp.where(cx_mask[:, None], c1, p1)
    offspring[1::2] = xp.where(cx_mask[:, None], c2, p2)
    return offspring


def cx_one_point(parents, cx_mask, xp):
    """
    One-point crossover.

    Args:
        parents: Parents as arrays of real parameters with (`pop_size`,`n_params`) shape.
        cx_mask: Crossover mask.
        xp: Numpy or Cupy backend.

    Returns:
        Resulting offspring.
    """
    N, D = parents.shape
    p1 = parents[0::2]
    p2 = parents[1::2]
    offspring = xp.empty_like(parents)

    points = xp.random.randint(1, D, size=p1.shape[0])

    idx = xp.arange(D)
    swap_mask = idx < points[:, None]
    final_mask = swap_mask & cx_mask[:, None]

    offspring[0::2] = xp.where(final_mask, p2, p1)
    offspring[1::2] = xp.where(final_mask, p1, p2)
    return offspring


def cx_two_point(parents, cx_mask, xp):
    """
    Two-point crossover.

    Args:
        parents: Parents as arrays of real parameters with (`pop_size`,`n_params`) shape.
        cx_mask: Crossover mask.
        xp: Numpy or Cupy backend.

    Returns:
        Resulting offspring.
    """
    N, D = parents.shape
    p1 = parents[0::2]
    p2 = parents[1::2]
    offspring = xp.empty_like(parents)

    pts = xp.sort(xp.random.randint(0, D, size=(len(p1), 2)), axis=1)
    low, high = pts[:, 0], pts[:, 1]

    idx = xp.arange(D)
    swap_mask = (idx >= low[:, None]) & (idx < high[:, None])
    final_mask = swap_mask & cx_mask[:, None]

    offspring[0::2] = xp.where(final_mask, p2, p1)
    offspring[1::2] = xp.where(final_mask, p1, p2)
    return offspring


def cx_uniform(parents, cx_mask, xp, cx_indpb=0.5):
    """
    Uniform crossover.

    Args:
        parents: Parents as arrays of real parameters with (`pop_size`,`n_params`) shape.
        cx_mask: Crossover mask.
        xp: Numpy or Cupy backend.
        cx_indpb: Independent probability for each parameter to be exchanged.

    Returns:
        Resulting offspring.
    """
    p1 = parents[0::2]
    p2 = parents[1::2]
    offspring = xp.empty_like(parents)

    gene_mask = xp.random.random(p1.shape) < cx_indpb
    final_mask = gene_mask & cx_mask[:, None]

    offspring[0::2] = xp.where(final_mask, p2, p1)
    offspring[1::2] = xp.where(final_mask, p1, p2)
    return offspring


# ---------- MUTATION OPERATORS ------------
def mut_gaussian(population, mut_mask, xp, mu=0, sigma=1, mut_indpb=0.1):
    """
    Gaussian mutation.

    Args:
        population: A population of individuals as array of real parameters with (`pop_size`, `n_params`) shape.
        mut_mask: Mask of mutation.
        xp: Numpy or Cupy backend.
        mu: Mean or sequence of means for the gaussian addition mutation.
        sigma: Standard deviation or sequence of standard deviations for the gaussian addition mutation.
        mut_indpb: Independent probability for each parameter to be mutated.

    Returns:
        Resulting mutated population.
    """
    gene_mask = (xp.random.random(population.shape) < mut_indpb) &  mut_mask[:, None]

    noise = xp.random.normal(mu, sigma, size=population.shape)
    population += noise * gene_mask

    return population


SELECTION_ARGS = {
    sel_tournament: ['tournament_size']
}

CROSSOVER_ARGS = {
    cx_blx_alpha: ['alpha'],
    cx_uniform: ['cx_indpb']
}

MUTATION_ARGS = {
    mut_gaussian: ['mu', 'sigma', 'mut_indpb']
}