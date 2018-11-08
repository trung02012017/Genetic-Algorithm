from GA import GenericAlgorithm
import matplotlib.pyplot as plt


def draw_result(result):
    plt.figure(2)
    plt.plot(result, 'r-', label="Loss GA")
    plt.xlabel("epoch")
    plt.ylabel("Loss GA")
    plt.legend()
    name = 'result GA'
    name += '.png'
    plt.savefig(name)
    plt.clf()


def main():
    pop_size = 100
    gen_size = 50
    num_selected_parents = int(pop_size/2)
    crossover_rate = 0.4
    mutation_rate = 0.05
    epochs = 1000

    GA = GenericAlgorithm(pop_size, gen_size, num_selected_parents, crossover_rate, mutation_rate, epochs)
    result = GA.run()
    draw_result(result)


if __name__ == "__main__":
    main()