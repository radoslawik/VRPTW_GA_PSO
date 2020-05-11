# -*- coding: utf-8 -*-

import sys
from alg_creator import *

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("invalid arguments")
        sys.exit()

    random.seed(64)

    problem_name = str(sys.argv[1])
    alg_name = str(sys.argv[2])

    customers_count = 25
    max_generations = 100
    
    particles_pop_size = 80
    social_acceleration = 2
    cognitive_acceleration = 2

    population_size = 150
    crossover_prob = 0.85
    mutation_prob = 0.1

    if alg_name == "PSO":
        res = run_pso(instance_name=problem_name, particle_size=customers_count, pop_size=particles_pop_size,
                      max_iteration=max_generations, cognitive_coef=cognitive_acceleration,
                      social_coef=social_acceleration)

    elif alg_name == "GA":
        res = run_ga(instance_name=problem_name, individual_size=customers_count, pop_size=population_size,
                     cx_pb=crossover_prob, mut_pb=mutation_prob, n_gen=max_generations)

    else:
        print("invalid algorithm")
        sys.exit()

    instance = load_problem_instance(problem_name)

    '''
    for single_route in res:
        print("new route")
        for customer_id in single_route:
            print(f'new customer id: {customer_id}')
            coordinates = [instance[F'C_{customer_id}'][COORDINATES][X_COORD],
                           instance[F'C_{customer_id}'][COORDINATES][Y_COORD]]
            for cord in coordinates:
                print(cord)
    '''