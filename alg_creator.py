
from core_funs import *
from deap import base, creator, tools
import matplotlib.pyplot as plt
import numpy
import time


# plot data from given problem with chosen customer amount
def plot_instance(instance_name, customer_number):

    instance = load_problem_instance(instance_name)
    depot = [instance[DEPART][COORDINATES][X_COORD], instance[DEPART][COORDINATES][Y_COORD]]
    dep, = plt.plot(depot[0], depot[1], 'kP', label='depot')

    for customer_id in range(1,customer_number):
        coordinates = [instance[F'C_{customer_id}'][COORDINATES][X_COORD],
                       instance[F'C_{customer_id}'][COORDINATES][Y_COORD]]
        custs, = plt.plot(coordinates[0], coordinates[1], 'ro', label='customers')
    plt.ylabel("y coordinate")
    plt.xlabel("x coordinate")
    plt.legend([dep, custs], ['depot', 'customers'], loc=1)
    plt.show()


# plot the result
def plot_route(route, instance_name):

    instance = load_problem_instance(instance_name)
    color_pack = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko',
                  'b*', 'g*', 'r*', 'c*', 'm*', 'y*', 'k*',
                  'bp', 'gp', 'rp', 'cp', 'mp', 'yp', 'kp']
    line_color_pack = ['b', 'g', 'r', 'c', 'm', 'y', 'k',
                       'b', 'g', 'r', 'c', 'm', 'y', 'k',
                       'b', 'g', 'r', 'c', 'm', 'y', 'k']
    c_ind = -1

    depot = [instance[DEPART][COORDINATES][X_COORD], instance[DEPART][COORDINATES][Y_COORD]]
    plt.plot(depot[0], depot[1], 'kP')
    for single_route in route:
        c_ind += 1
        coordinates = depot
        for customer_id in single_route:
            prev_coords = coordinates
            coordinates = [instance[F'C_{customer_id}'][COORDINATES][X_COORD],
                           instance[F'C_{customer_id}'][COORDINATES][Y_COORD]]
            plt.plot(coordinates[0], coordinates[1], color_pack[c_ind])
            plt.arrow(prev_coords[0], prev_coords[1], coordinates[0] - prev_coords[0],
                      coordinates[1] - prev_coords[1], color=line_color_pack[c_ind],
                      length_includes_head=True, head_width=1, head_length=2)
        plt.arrow(coordinates[0], coordinates[1], depot[0] - coordinates[0],
                  depot[1] - coordinates[1], color=line_color_pack[c_ind],
                  length_includes_head=True, head_width=1, head_length=2)
    plt.ylabel("y coordinate")
    plt.xlabel("x coordinate")
    plt.show()
    return


# printing the solution
def print_route(route):
    route_num = 0
    for sub_route in route:
        route_num += 1
        single_route = '0'
        for customer_id in sub_route:
            single_route = f'{single_route} -> {customer_id}'
        single_route = f'{single_route} -> 0'
        print(f' # Route {route_num} # {single_route}')


# runs the pso and prints the solution
# https://deap.readthedocs.io/en/master/examples/pso_basic.html
# Once the operators are registered in the toolbox, we can fire up the algorithm by firstly creating a new population,
# and then apply the original PSO algorithm.
# The variable best contains the best particle ever found (it is known as gbest in the original algorithm).
def run_pso(instance_name, particle_size, pop_size, max_iteration,
            cognitive_coef, social_coef, s_limit=3, plot=False, save=False, logs=False):

    instance = load_problem_instance(instance_name)

    if instance is None:
        return

    if plot:
        plot_instance(instance_name=instance_name, customer_number=particle_size)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
                   smin=None, smax=None, best=None)

    toolbox = base.Toolbox()
    toolbox.register("particle", generate_particle,
                     size=particle_size, val_min=1, val_max=particle_size, s_min=-s_limit, s_max=s_limit)
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
    iter_num = 0
    previous_best = 0

    print('### EVOLUTION START ###')
    start = time.time()

    for g in range(max_iteration):

        fit_count = 0
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if part.fitness.values[0] > previous_best:
                previous_best = part.fitness.values[0]
                iter_num = g + 1
            elif part.fitness.values[0] == previous_best:
                fit_count += 1

        if fit_count > int(numpy.ceil(pop_size * 0.15)):
            rand_pop = toolbox.population(n=pop_size)
            for part in rand_pop:
                part.fitness.values = toolbox.evaluate(part)
            some_inds = tools.selRandom(rand_pop, int(numpy.ceil(pop_size * 0.1)))  # random pop here
            mod_pop = tools.selWorst(pop, int(numpy.ceil(pop_size * 0.9)))
        else:
            some_inds = tools.selBest(pop, int(numpy.ceil(pop_size * 0.05)))  # elite pop here
            mod_pop = tools.selRandom(pop, int(numpy.ceil(pop_size * 0.95)))

        mod_pop = list(map(toolbox.clone, mod_pop))

        for part in mod_pop:
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values

        for part in mod_pop:
            toolbox.update(part, best)

        mod_pop.extend(some_inds)
        pop[:] = mod_pop

        # Gather all the stats in one list and print them
        logbook.record(gen=g+1, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

    end = time.time()
    print('### EVOLUTION END ###')
    best_ind = tools.selBest(pop, 1)[0]
    print(f'Best individual: {best_ind}')
    route = create_route_from_ind(best_ind, instance)
    print_route(route)
    print(f'Fitness: { round(best_ind.fitness.values[0],2) }')
    print(f'Total cost: { round(calculate_fitness(best_ind, instance)[1],2) }')
    print(f'Found in (iteration): { iter_num }')
    print(f'Execution time (s): { round(end - start,2) }')
    # print(f'{round(best_ind.fitness.values[0], 2)} & {round(calculate_fitness(best_ind, instance)[1],2)} & {iter_num} & {round(end - start, 2)}')

    if plot:
        plot_route(route=route, instance_name=instance_name)

    return route


# runs ga and prints the solution
# https://deap.readthedocs.io/en/master/examples/ga_onemax.html
def run_ga(instance_name, individual_size, pop_size, cx_pb, mut_pb, n_gen, plot=False, save=False, logs=False):

    instance = load_problem_instance(instance_name)

    if instance is None:
        return

    if plot:
        plot_instance(instance_name=instance_name, customer_number=individual_size)

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
    toolbox.register('mate', crossover_pmx)
    toolbox.register('mutate', mutate_swap)
    pop = toolbox.population(n=pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    iter_num = 0
    previous_best = 0

    print('### EVOLUTION START ###')
    start = time.time()

    # Evaluate the entire population
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    # Begin the evolution
    for gen in range(n_gen):
        # Keep the best individual
        elite_ind = tools.selBest(pop, 1)

        # Choose 10% of best offspring, roulette select the rest 90% of rest
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

        # Add elite member
        offspring.extend(elite_ind)
        # Replace population by offspring
        pop[:] = offspring

        # Evaluate new population
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
            if ind.fitness.values[0] > previous_best:
                previous_best = ind.fitness.values[0]
                iter_num = gen + 1

        logbook.record(gen=gen+1, evals=len(offspring), **stats.compile(offspring))
        print(logbook.stream)

    end = time.time()
    print('### EVOLUTION END ###')
    best_ind = tools.selBest(pop, 1)[0]
    print(f'Best individual: {best_ind}')
    route = create_route_from_ind(best_ind, instance)
    print_route(route)
    print(f'Fitness: { round(best_ind.fitness.values[0],2) }')
    print(f'Total cost: { round(calculate_fitness(best_ind, instance)[1],2) }')
    print(f'Found in (iteration): { iter_num }')
    print(f'Execution time (s): { round(end - start,2) }')
    # print(f'{ round(best_ind.fitness.values[0],2) } & { round(calculate_fitness(best_ind, instance)[1],2) } & { iter_num } & { round(end - start,2) }')

    if plot:
        plot_route(route=route, instance_name=instance_name)

    return route
