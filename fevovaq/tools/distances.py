# --------------------- Distance functions for CHC Algorithm -----------------------------------------------------------

# Fitness-only distances
def fitness_l1(population, fitness, xp):
    p1 = fitness[0::2]
    p2 = fitness[1::2]
    return xp.abs(p1 - p2)

def fitness_l2(population, fitness, xp):
    p1 = fitness[0::2]
    p2 = fitness[1::2]
    return (p1 - p2) ** 2

def fitness_relative(population, fitness, xp, eps=1e-12):
    p1 = fitness[0::2]
    p2 = fitness[1::2]
    return xp.abs(p1 - p2) / (xp.abs(p1) + xp.abs(p2) + eps)


# Parameters-only distances
def param_l1(population, fitness, xp):
    p1 = population[0::2]
    p2 = population[1::2]
    return xp.sum(xp.abs(p1 - p2), axis=1)

def param_l2(population, fitness, xp):
    p1 = population[0::2]
    p2 = population[1::2]
    return xp.linalg.norm(p1 - p2, axis=1)


# Hybrid parameter-fitness distances
def hybrid_param_fit(population, fitness, xp, fact=0.5):
    p1, p2 = population[0::2], population[1::2]
    f1, f2 = fitness[0::2], fitness[1::2]

    dp = xp.linalg.norm(p1 - p2, axis=1)
    df = xp.abs(f1 - f2)
    return fact * dp + (1 - fact) * df


