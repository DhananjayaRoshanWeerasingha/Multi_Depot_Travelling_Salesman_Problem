import random
import numpy as np

class AntColonyOptimizer:
    def __init__(self, n_ants, n_iterations, decay_factor, alpha, beta, q):
        self.n_ants = n_ants  # number of ants
        self.n_iterations = n_iterations  # number of iterations
        self.decay_factor = decay_factor  # pheromone decay factor
        self.alpha = alpha  # pheromone exponent
        self.beta = beta  # heuristic exponent
        self.q = q  # pheromone intensity
        self.distances = None  # distance matrix
        self.pheromone = None  # pheromone matrix
        self.best_solution = None  # best solution found
        self.best_fitness = np.inf  # best fitness found

    def fit(self, distances):
        self.distances = distances
        n_cities = distances.shape[0]
        self.pheromone = np.ones((n_cities, n_cities))  # initialize pheromone matrix

        for i in range(self.n_iterations):
            solutions = self._generate_solutions()
            self._update_pheromone(solutions)
            best_solution, best_fitness = self._get_best_solution(solutions)
            if best_fitness < self.best_fitness:
                self.best_solution = best_solution
                self.best_fitness = best_fitness
            self.pheromone *= self.decay_factor  # decay pheromone

    def _generate_solutions(self):
        solutions = []
        for ant in range(self.n_ants):
            visited_cities = [0]  # start from city 0
            while len(visited_cities) < self.distances.shape[0]:
                unvisited_cities = set(range(self.distances.shape[0])) - set(visited_cities)
                current_city = visited_cities[-1]
                next_city = self._choose_next_city(current_city, unvisited_cities)
                visited_cities.append(next_city)
            solutions.append(visited_cities)
        return solutions

    def _choose_next_city(self, current_city, unvisited_cities):
        pheromone = self.pheromone[current_city, list(unvisited_cities)]
        distance = self.distances[current_city, list(unvisited_cities)]
        heuristic = 1 / distance
        probabilities = np.power(pheromone, self.alpha) * np.power(heuristic, self.beta)
        probabilities /= np.sum(probabilities)
        next_city = list(unvisited_cities)[np.random.choice(range(len(probabilities)), p=probabilities)]
        return next_city

    def _update_pheromone(self, solutions):
        for i in range(self.n_ants):
            path = solutions[i]
            path_length = sum([self.distances[path[j-1], path[j]] for j in range(1, len(path))])
            for j in range(1, len(path)):
                self.pheromone[path[j-1], path[j]] += self.q / path_length

    def _get_best_solution(self, solutions):
        best_solution = min(solutions, key=lambda x: sum([self.distances[x[j-1], x[j]] for j in range(1, len(x))]))
        best_fitness = sum([self.distances[best_solution[j-1], best_solution[j]] for j in range(1, len(best_solution))])
        return best_solution, best_fitness
