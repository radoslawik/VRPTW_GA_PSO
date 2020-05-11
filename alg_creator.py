from core_funs import *
from deap import base, creator, tools, algorithms
import numpy


# printing the solution
def print_route(route):
    route_num = 0
    for sub_route in route:
        route_num += 1
        single_route = '0'
        for customer_id in sub_route:
            single_route = f'{single_route} - {customer_id}'
        single_route = f'{single_route} - 0'
        print(f' Route {route_num}: {single_route}')


# runs the pso and prints the solution
# https://deap.readthedocs.io/en/master/examples/pso_basic.html
# Once the operators are registered in the toolbox, we can fire up the algorithm by firstly creating a new population,
# and then apply the original PSO algorithm.
# The variable best contains the best particle ever found (it is known as gbest in the original algorithm).
def run_pso(instance_name, particle_size, pop_size, max_iteration,
            cognitive_coef, social_coef, speed_min=-3, speed_max=3):

    instance = load_problem_instance(instance_name)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
                   smin=None, smax=None, best=None)

    toolbox = base.Toolbox()
    toolbox.register("particle", generate_particle,
                     size=particle_size, val_min=1, val_max=particle_size, s_min=speed_min, s_max=speed_max)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", update_particle, phi1=cognitive_coef, phi2=social_coef)
    toolbox.register('evaluate', calculate_fitness, data=instance)

    pop = toolbox.population(n=pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    best = None

    print('Start of evolution')
    for g in range(max_iteration):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

    print('End of evolution')
    best_ind = tools.selBest(pop, 1)[0]
    print(f'Best individual: {best_ind}')
    route = create_route_from_ind(best_ind, instance)
    print_route(route)
    print(f'Fitness: {best_ind.fitness.values[0]}')
    print(f'Total cost: { calculate_fitness(best_ind, instance)[1]}')

    return route


# runs ga and prints the solution
# https://deap.readthedocs.io/en/master/examples/ga_onemax.html
def run_ga(instance_name, individual_size, pop_size, cx_pb, mut_pb, n_gen):

    instance = load_problem_instance(instance_name)

    if instance is None:
        return

    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register('indexes', random.sample, range(1, individual_size + 1), individual_size)
    # Structure initializers
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('evaluate', calculate_fitness, data=instance)
    toolbox.register('select', tools.selRoulette)
    toolbox.register('mate', crossover_two_points)
    toolbox.register('mutate', mutate_swap)
    pop = toolbox.population(n=pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    # Evaluate the entire population
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    print('Start of evolution')
    # Begin the evolution
    for gen in range(n_gen):
        # Keep the best individual
        elite = tools.selBest(pop, 1)

        # Roulette select the rest 90% of worst offsprings
        offspring = tools.selBest(pop, int(numpy.ceil(pop_size * 0.1)))
        offspring = list(map(toolbox.clone, offspring))
        offspring_roulette = toolbox.select(pop, int(numpy.floor(pop_size * 0.9)) - 1)
        offspring.extend(offspring_roulette)

        # Clone the selected individuals
        offspring = list(toolbox.map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Replace population by offspring
        offspring.extend(elite)
        pop[:] = offspring

        # Evaluate new population
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        logbook.record(gen=gen, evals=len(offspring), **stats.compile(offspring))
        print(logbook.stream)

    print('End of evolution')
    best_ind = tools.selBest(pop, 1)[0]
    print(f'Best individual: {best_ind}')
    route = create_route_from_ind(best_ind, instance)
    print_route(route)
    print(f'Fitness: {best_ind.fitness.values[0]}')
    print(f'Total cost: { calculate_fitness(best_ind, instance)[1]}')

    return route
