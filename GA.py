import numpy as np
import time
import pandas as pd
import os.path

class GenericAlgorithm(object):

    def __init__(self, pop_size, gen_size, num_selected_parents, crossover_rate, mutation_rate, epochs):

        self.pop_size = pop_size        # number of population
        self.gen_size = gen_size        # number of genes in each chromosome
        self.num_selected_parents = num_selected_parents    # the number of selected parents in each epoch
        self.mutation_rate = mutation_rate      # mutation rate
        self.crossover_rate = crossover_rate    # crossover_rate
        self.epochs = epochs            # number of epochs

    def initialize_population(self):

        init_population = np.random.uniform(-5, 5, (self.pop_size, self.gen_size))

        return init_population

    def choose_population(self, population, num_citizens):
        num_delete = self.pop_size - num_citizens
        random = np.random.permutation(self.pop_size)
        delete_index = random[0:num_delete]
        population = np.delete(population, delete_index, 0)

        return population

    def get_fitness(self, population):
        fitness_value = np.zeros((self.pop_size, 1))
        fitness = 0
        for i in range(self.pop_size):
            for j in range(self.gen_size):
                if j % 2 == 0:
                    fitness += population[i, j]**2
                else:
                    fitness += population[i, j]**3

            fitness_value[i] += fitness
        return fitness_value

    def get_best_fitness(self, population):

        fitness_value = self.get_fitness(population)
        indices = np.where(fitness_value == np.min(fitness_value))[0]
        best_solution = population[indices, :]
        best_fitness = fitness_value[indices]
        return best_fitness.reshape((1,))

    def get_index_chromosome_by_fitness(self, value, fitness_value, population):
        index = np.where(fitness_value == value)[0]
        chromosome = population[index]
        return chromosome[0]

    def select_mating_pool(self, num_parents, population):

        selected_parents = np.zeros((num_parents, self.gen_size))

        fitness_value = self.get_fitness(population)
        # print(fitness_value)
        sorted_fitness = np.sort(fitness_value, axis=0)
        # print(sorted_fitness)
        for i in range(num_parents):
            parent = self.get_index_chromosome_by_fitness(sorted_fitness[i], fitness_value, population)
            selected_parents[i] += parent

        return selected_parents

    def choose_parent_pair(self, parents):
        size = parents.shape[0]
        parent_pair_list = []
        parents_indices = np.random.permutation(size)
        for i in range(0, size, 2):
            parent1_index = parents_indices[i]
            parent2_index = parents_indices[i+1]
            parent1 = parents[parent1_index].reshape((1, self.gen_size))
            parent2 = parents[parent2_index].reshape((1, self.gen_size))
            pair = [parent1, parent2]
            parent_pair_list.append(pair)

        return parent_pair_list

    def crossover(self, parent_pair):

        parent1 = parent_pair[0]
        parent2 = parent_pair[1]

        child1 = np.zeros((1, self.gen_size))
        child2 = np.zeros((1, self.gen_size))

        num_gens_parent1 = int(self.gen_size*self.crossover_rate)
        num_gens_parent2 = self.gen_size - num_gens_parent1

        permutation = np.random.permutation(self.gen_size)

        for i in range(num_gens_parent1):
            index = permutation[i]
            child1[:, index] += parent1[:, index]
            child2[:, index] += parent2[:, index]

        permutation = permutation[num_gens_parent1:self.gen_size]

        for i in permutation:
            index = i
            child1[:, index] += parent2[:, index]
            child2[:, index] += parent1[:, index]

        return child1, child2

    def mutation(self, child):

        a = np.random.randint(0, self.gen_size-1, 2)
        a1 = a[0]
        # print(a1)
        range = int(self.mutation_rate*self.pop_size)
        if a1+range > self.pop_size:
            a2 = int(a1 + range)
        else:
            a2 = int(a1 - range)

        if a1 < a2:
            selected_part = child[:, a1:a2]
            reversed_part = np.flip(selected_part)
            child[:, a1:a2] = reversed_part

        if a1 > a2:
            selected_part = child[:, a2:a1]
            reversed_part = np.flip(selected_part)
            child[:, a2:a1] = reversed_part

        return child

    def save(self, epoch, best_fitness, time_i, result_file_path):
        result = {
            "epoch": epoch,
            "best_fitness": best_fitness,
            "time": time_i
        }

        df = pd.DataFrame(result)
        if not os.path.exists(result_file_path):
            columns = ["epoch", "best_fitness", "time"]
            df[columns]
            df.to_csv('result_GA.csv', index=False, columns=columns)
        else:
            with open('result_GA.csv', 'a') as csv_file:
                df.to_csv(csv_file, mode='a', header=False, index=False)



    def run(self):
        print("start to run....")
        population = self.initialize_population()
        result = []
        result_file_path = "result_GA.csv"
        for i in range(self.epochs):
            start_time = time.time()
            parents = self.select_mating_pool(int(self.pop_size/2), population)
            population = self.choose_population(population, int(self.pop_size / 2))
            pair_list = self.choose_parent_pair(parents)

            for pair in pair_list:
                child1, child2 = self.crossover(pair)
                population = np.concatenate((population, self.mutation(child1), self.mutation(child2)), axis=0)
            end_time = time.time()
            execute_time = end_time - start_time

            epoch = i+1
            best_fitness = np.round(self.get_best_fitness(population), 3)
            time_i = round(execute_time, 3)
            result.append(best_fitness)

            self.save(epoch, best_fitness, time_i, result_file_path)

        print("finish....")

        return result



