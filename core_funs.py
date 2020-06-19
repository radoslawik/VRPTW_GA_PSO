
import random
import operator
import math
import collections

from deap import base, creator, tools
from process_data import *


def create_route_from_ind(individual, data):

    vehicle_capacity = data[VEHICLE_CAPACITY]
    depart_due_time = data[DEPART][DUE_TIME]

    route = []
    sub_route = []
    vehicle_load = 0
    time_elapsed = 0
    previous_cust_id = 0
    for customer_id in individual:
        demand = data[F'C_{customer_id}'][DEMAND]
        updated_vehicle_load = vehicle_load + demand
        service_time = data[F'C_{customer_id}'][SERVICE_TIME]
        return_time = data[DISTANCE_MATRIX][customer_id][0]
        travel_time = data[DISTANCE_MATRIX][previous_cust_id][customer_id]
        provisional_time = time_elapsed + travel_time + service_time + return_time
        # Validate vehicle load and elapsed time
        if (updated_vehicle_load <= vehicle_capacity) and (provisional_time <= depart_due_time):
            # Add to current sub-route
            sub_route.append(customer_id)
            vehicle_load = updated_vehicle_load
            time_elapsed = provisional_time - return_time
        else:
            # Save current sub-route
            route.append(sub_route)
            # Initialize a new sub-route and add to it
            sub_route = [customer_id]
            vehicle_load = demand
            travel_time = data[DISTANCE_MATRIX][0][customer_id]
            time_elapsed = travel_time + service_time
        # Update last customer ID
        previous_cust_id = customer_id
    if sub_route:
        # Save current sub-route before return if not empty
        route.append(sub_route)
    return route


def calculate_fitness(individual, data):

    transport_cost = 8.0  # cost of moving 1 vehicle for 1 unit
    vehicle_setup_cost = 50.0  # cost of adapting new vehicle
    wait_penalty = 0.5  # penalty for arriving too early
    delay_penalty = 3.0  # penalty for arriving too late

    route = create_route_from_ind(individual, data)
    total_cost = 999999
    fitness = 0
    max_vehicles_count = data[MAX_VEHICLE_NUMBER]

    # checking if we have enough vehicles
    if len(route) <= max_vehicles_count:
        total_cost = 0
        for sub_route in route:
            sub_route_time_cost = 0
            sub_route_distance = 0
            elapsed_time = 0
            previous_cust_id = 0
            for cust_id in sub_route:
                # Calculate section distance
                distance = data[DISTANCE_MATRIX][previous_cust_id][cust_id]
                # Update sub-route distance
                sub_route_distance = sub_route_distance + distance

                # Calculate time cost
                arrival_time = elapsed_time + distance

                waiting_time = max(data[F'C_{cust_id}'][READY_TIME] - arrival_time, 0)
                delay_time = max(arrival_time - data[F'C_{cust_id}'][DUE_TIME], 0)
                time_cost = wait_penalty * waiting_time + delay_penalty * delay_time

                # Update sub-route time cost
                sub_route_time_cost += time_cost

                # Update elapsed time
                service_time = data[F'C_{cust_id}'][SERVICE_TIME]
                elapsed_time = arrival_time + service_time

                # Update last customer ID
                previous_cust_id = cust_id

            # Calculate transport cost
            distance_depot = data[DISTANCE_MATRIX][previous_cust_id][0]
            sub_route_distance += distance_depot
            sub_route_transport_cost = vehicle_setup_cost + transport_cost * sub_route_distance
            # Obtain sub-route cost
            sub_route_cost = sub_route_time_cost + sub_route_transport_cost
            # Update total cost`
            total_cost += sub_route_cost

        # fitness = - math.log(1.0 / total_cost)
        fitness = 100000.0 / total_cost

    return fitness, total_cost


# Double point crossover
def crossover_pmx(ind1, ind2):

    ind_len = len(ind1)
    pos_ind1 = [0]*ind_len
    pos_ind2 = [0]*ind_len

    # position indices list
    for i in range(ind_len):
        pos_ind1[ind1[i]-1] = i
        pos_ind2[ind2[i]-1] = i

    # crossover points
    locus1 = random.randint(0, int(ind_len/2))
    locus2 = random.randint(int(ind_len/2)+1, ind_len-1)

    # crossover
    for i in range(locus1, locus2):
        temp1 = ind1[i]
        temp2 = ind2[i]
        # swap
        ind1[i], ind1[pos_ind1[temp2-1]] = temp2, temp1
        ind2[i], ind2[pos_ind2[temp1-1]] = temp1, temp2
        # save updated positions
        pos_ind1[temp1-1], pos_ind1[temp2-1] = pos_ind1[temp2-1], pos_ind1[temp1-1]
        pos_ind2[temp1-1], pos_ind2[temp2-1] = pos_ind2[temp2-1], pos_ind2[temp1-1]

    return ind1, ind2


# Chose 2 indexes and swap values between them
def mutate_swap(individual):

    ind_len = len(individual)
    locus1 = random.randint(0, int(ind_len / 2))
    locus2 = random.randint(int(ind_len / 2) + 1, ind_len - 1)

    temp = individual[locus1]
    individual[locus1] = individual[locus2]
    individual[locus2] = temp

    return individual,


# The initialization consist in generating a random position and a random speed for a particle.
# The next function creates a particle and initializes its attributes,
# except for the attribute best, which will be set only after evaluation
def generate_particle(size, val_min, val_max, s_min, s_max):
    vals = list(range(val_min, val_max + 1))
    random.shuffle(vals)
    part = creator.Particle(vals)
    part.speed = [random.uniform(s_min, s_max) for _ in range(size)]
    part.smin = s_min
    part.smax = s_max
    return part


def create_particle(vals, s_min, s_max):
    part = creator.Particle(vals)
    part.speed = [random.uniform(s_min, s_max) for _ in range(len(vals))]
    part.smin = s_min
    part.smax = s_max
    return part


def remove_duplicates(vals):
    duplic = [item for item, count in collections.Counter(vals).items() if count > 1]
    uniq_part = []
    offset = 0.001
    count = [1] * len(duplic)
    for val in vals:
        if val in duplic:
            ind = duplic.index(val)
            val += offset * count[ind]
            count[ind] += 1
        uniq_part.append(val)

    return uniq_part


# Change floats to integers and deal with duplicates
def validate_particle(particle):
    unique_part = remove_duplicates(particle)
    sorted_asc = sorted(unique_part, key=float)
    validated_part = []

    if len(sorted_asc) > len(set(sorted_asc)):
        print("problem")

    for val in unique_part:
        index = sorted_asc.index(val)
        validated_part.append((index + 1))

    return validated_part


def validate_particle2(particle):
    swap_list = remove_duplicates(list(map(operator.add, list(range(1, len(particle)+1)), particle.speed)))
    validated_part = []
    validated_speed = []
    sorted_asc = sorted(swap_list, key=float)

    if len(sorted_asc) > len(set(sorted_asc)):
        print("problem")
    for val in swap_list:
        index = sorted_asc.index(val)
        validated_part.append(particle[index])
        validated_speed.append(particle.speed[index])

    return validated_part, validated_speed

# The function updateParticle() first computes the speed,
# then limits the speed values between smin and smax,
# and finally computes the new particle position.


def update_particle(part, best, phi1, phi2):

    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    # the particle's best position
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    # the neighbourhood best
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    # update particle speed
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    # speed limits
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
            # adjust maximum speed if necessary
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)

    new_part = list(map(operator.add, part, part.speed))
    part[:] = validate_particle(new_part)

    # part[:], part.speed[:] = validate_particle2(part)




