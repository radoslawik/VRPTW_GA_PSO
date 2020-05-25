# -*- coding: utf-8 -*-

import sys
from alg_creator import *

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("invalid arguments")
        sys.exit()

    random.seed(64)
    plot_result = True

    problem_name = str(sys.argv[1])
    alg_name = str(sys.argv[2])

    customers_count = 50
    max_generations = 500
    
    particles_pop_size = 80
    social_acceleration = 2
    cognitive_acceleration = 2
    speed_limit = 1.2

    population_size = 150
    crossover_prob = 0.85
    mutation_prob = 0.05

    print('### GENERAL INFO ###')
    print('Problem name: ' + problem_name)
    print(f'Customer count: {customers_count}')
    print(f'Max iterations: {max_generations}')
    print('Algorithm: ' + alg_name)
    print('### ALGORITHM PARAMETERS ###')

    if alg_name == "PSO":
        print(f'Particles population size: {particles_pop_size}')
        print(f'Social acceleration: {social_acceleration}')
        print(f'Cognitive acceleration: {cognitive_acceleration}')
        print(f'Speed limit: {speed_limit}')
        res = run_pso(instance_name=problem_name, particle_size=customers_count, pop_size=particles_pop_size,
                      max_iteration=max_generations, cognitive_coef=cognitive_acceleration,
                      social_coef=social_acceleration, s_limit=speed_limit, plot=plot_result)

    elif alg_name == "GA":
        print(f'Population size: {population_size}')
        print(f'Crossover probability: {crossover_prob}')
        print(f'Mutation probability: {mutation_prob}')
        res = run_ga(instance_name=problem_name, individual_size=customers_count, pop_size=population_size,
                     cx_pb=crossover_prob, mut_pb=mutation_prob, n_gen=max_generations, plot=plot_result)

    else:
        print("invalid algorithm")
        sys.exit()

