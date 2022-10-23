import datetime
import math
import copy
import time

import numpy as np
import matplotlib.pyplot as plt
from gurobi import solve_asymmetric_tsp_gurobi


class Vertex:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def distance(self, vertex):
        return math.dist([self.x, self.y], [vertex.x, vertex.y])


class Tour:
    def __init__(self, tour):
        self.tour = tour
        self.length = 0
        self.tour_length()

    def __lt__(self, other):
        return self.length < other.length

    def tour_length(self):
        for i in range(len(self.tour)):
            self.length += self.tour[i].distance(self.tour[(i+1) % len(self.tour)])
        return self.length


def get_random_vertices(vertex_count):
    return [Vertex(np.random.rand(2)) for _ in range(vertex_count)]


def get_initial_population(vertices, population_size):
    return [Tour(np.random.permutation(vertices)) for _ in range(population_size)]


def get_selection(population, elitism_count, eps=0.1):
    population_count = len(population)

    population.sort()

    selection = population[:elitism_count]

    random_count = round(population_count * eps)
    while len(selection) < population_count:
        random_selection = np.random.choice(population, random_count, replace=False)
        random_selection.sort()
        selection.append(random_selection[0])

    return selection


def PMX_crossover(parents):
    p1 = list(copy.deepcopy(parents[0].tour))
    p2 = parents[1].tour

    n = len(p1)
    pos = np.random.randint(n)

    for i in range(pos):
        this = p2[i]
        p1_index = p1.index(this)
        p1[i], p1[p1_index] = p1[p1_index], p1[i]

    child = Tour(p1)

    return child


def get_children(selection, children_count):
    children = []
    while len(children) < children_count:
        parents = np.random.choice(selection, 2, replace=False)
        children.append(PMX_crossover(parents))

    return children


def mutate(population, vertex_count, probability):
    new_population = []
    for individual in population:
        if np.random.rand() > probability:
            new_population.append(individual)
        else:
            to_swap = np.random.choice(range(vertex_count), 2, replace=False)
            individual.tour[to_swap[0]], individual.tour[to_swap[1]] = individual.tour[to_swap[1]], individual.tour[to_swap[0]]
            new_population.append(Tour(individual.tour))

    return new_population


def draw_progress_plot(progress):
    plt.plot(progress)
    plt.xlabel('generation')
    plt.ylabel('tour length')
    plt.show()


def draw_tour(tour, opt_tour):
    for vertex in tour:
        plt.scatter(x=vertex.x, y=vertex.y, c='black')

    for i in range(len(tour) - 1):
        plt.plot([tour[opt_tour[i]].x, tour[opt_tour[i+1]].x], [tour[opt_tour[i]].y, tour[opt_tour[i+1]].y], c='grey', linestyle='dashed')
    plt.plot([tour[opt_tour[0]].x, tour[opt_tour[-1]].x], [tour[opt_tour[0]].y, tour[opt_tour[-1]].y], c='grey', linestyle='dashed')


    for i in range(len(tour) - 1):
        plt.plot([tour[i].x, tour[i+1].x], [tour[i].y, tour[i+1].y], c='blue')
    plt.plot([tour[0].x, tour[-1].x], [tour[0].y, tour[-1].y], c='blue')

    plt.show()


def genetic_algorithm(vertex_count, generation_count, population_size, elitism, mutate_probability):

    elitism_count = round(population_size * elitism)

    vertices = get_random_vertices(vertex_count)
    initial_population = get_initial_population(vertices, population_size)

    progress = []
    initial_population.sort()
    progress.append(initial_population[0].length)

    last_gen = initial_population
    for x in range(generation_count):
        selection = get_selection(last_gen, elitism_count)
        children = get_children(selection, population_size - elitism_count)
        children = mutate(children, vertex_count, mutate_probability)

        last_gen = selection[:elitism_count]
        last_gen.extend(children)

        last_gen.sort()
        progress.append(last_gen[0].length)

        # if x % 10 == 9:
            # print("Gen " + str(x+1) + '/' + str(generation_count) + ' done!')

    return last_gen[0], progress


def tour_to_adj_matrix(tour, n):
    tour = tour.tour
    matrix = np.zeros(shape=[n, n])
    for i in range(n):
        for j in range(n):
            if i!=j:
                matrix[i, j] = tour[i].distance(tour[j])

    return matrix


if __name__ == "__main__":
    vertex_count = 25
    generation_count = 100
    population_size = 100
    elitism = 0.1
    mutate_probability = 0.25

    rounds = 10
    avg_opt_gap = 0
    time_ga = 0
    for x in range(rounds):
        tmp_time = datetime.datetime.now()
        best_tour, progress = genetic_algorithm(vertex_count, generation_count, population_size, elitism, mutate_probability)
        time_ga += (datetime.datetime.now() - tmp_time).total_seconds()

        adj_matrix = tour_to_adj_matrix(best_tour, vertex_count)
        opt_solution = solve_asymmetric_tsp_gurobi(adj_matrix)

        opt_gap = (100 * best_tour.length) / opt_solution.length - 100
        avg_opt_gap += (opt_gap * (1/rounds))

        if x % 1 == 0:
            print("Round " + str(x+1) + '/' + str(rounds) + ' done!')

        if x == 99:
            draw_progress_plot(progress)
            draw_tour(best_tour.tour, opt_solution.tour)

    print(avg_opt_gap)
    print(time_ga / 10)
