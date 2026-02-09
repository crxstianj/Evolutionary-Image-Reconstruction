import numpy as np
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
import random
import colour
from multiprocessing import Pool
import os

output_folder = "prueba"
os.makedirs(output_folder, exist_ok=True)

# Clase para generar polígonos
class Individual:
    def __init__(self, l, w):
        self.l = l  # ancho
        self.w = w  # alto
        self.fitness = float('inf')
        self.array = None
        self.image = None
        self.create_random_image_array()

    def rand_color(self):
        # Devuelve un color aleatorio en formato hexadecimal
        return "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    def create_random_image_array(self):
        # Se agrega un número aleatorio de polígonos (entre 2 y 5) sobre un fondo aleatorio
        iterations = random.randint(2, 5)
        region = (self.l + self.w) // 8

        # Fondo de la imagen con color aleatorio
        img = Image.new("RGBA", (self.l, self.w), self.rand_color())

        # Para cada polígono se elige un número aleatorio de puntos y se dibuja un polígono
        for i in range(iterations):
            num_points = random.randint(3, 6)
            region_x = random.randint(0, self.l)
            region_y = random.randint(0, self.w)
            xy = []
            for j in range(num_points):
                xy.append((random.randint(region_x - region, region_x + region),
                           random.randint(region_y - region, region_y + region)))
            img_draw = ImageDraw.Draw(img)
            img_draw.polygon(xy, fill=self.rand_color())
        self.image = img
        self.array = np.array(img)

    def add_shape(self):
        # Función de mutación: agrega una polígono a la imagen
        iterations = random.randint(1, 1)
        region = random.randint(1, (self.l + self.w) // 4)
        img = self.image.copy()
        for i in range(iterations):
            num_points = random.randint(3, 6)
            region_x = random.randint(0, self.l)
            region_y = random.randint(0, self.w)
            xy = []
            for j in range(num_points):
                xy.append((random.randint(region_x - region, region_x + region),
                           random.randint(region_y - region, region_y + region)))
            img_draw = ImageDraw.Draw(img)
            img_draw.polygon(xy, fill=self.rand_color())
        self.image = img
        self.array = np.array(img)

    # Fitness usando distancia euclidiana
    def get_fitness_euclidean(self, target):
        diff_array = np.subtract(np.array(target), self.array)
        self.fitness = np.mean(np.absolute(diff_array))

    # Fitness usando la métrica Delta E
    def get_fitness(self, target):
        self.fitness = np.mean(colour.difference.delta_e.delta_E_CIE1976(target, self.array))


# Evaluación paralela del fitness
def evaluate_fitness(args):
    ind, target_array = args
    ind.get_fitness(target_array)
    return ind


# Función de selección por torneo: se elige el individuo con mejor fitness de un grupo aleatorio
def tournament_selection(population, k=5):
    participantes = random.sample(population, k)
    participantes.sort(key=lambda ind: ind.fitness)
    return participantes[0]


# Función de cruce: realiza un cruce simple por mezcla (blending) entre dos imágenes
def crossover(parent1, parent2):
    # Se elige un alpha aleatorio entre 0 y 1 para mezclar las imágenes
    alpha = random.random()
    child_img = Image.blend(parent1.image, parent2.image, alpha)
    child = Individual(parent1.l, parent1.w)
    child.image = child_img
    child.array = np.array(child_img)
    return child


# Función de mutación: con cierta probabilidad se aplica add_shape al individuo
def mutate(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        individual.add_shape()
    return individual


# Algoritmo genético
def genetic_algorithm(target_image_path, pop_size=100, generations=5000, tournament_k=5, mutation_rate=0.1,
                      elite_ratio=0.3):
    # Obtener dimensiones
    target_img = Image.open(target_image_path).convert("RGBA")
    target_img = ImageOps.fit(target_img, (256, 256), method=Image.LANCZOS)
    l, w = target_img.size
    target_array = np.array(target_img)

    # Inicializar la población con individuos aleatorios
    population = [Individual(l, w) for _ in range(pop_size)]
    with Pool() as pool:
        population = pool.map(evaluate_fitness, [(ind, target_array) for ind in population])

    best_fitness_history = []

    for gen in range(generations):
        population.sort(key=lambda ind: ind.fitness)
        best = population[0]
        best_fitness_history.append(best.fitness)
        # print("Generación {} - Mejor fitness: {:.2f}".format(gen, best.fitness))

        # Condición de parada
        if best.fitness < 20.0:
            break

        elite_size = int(elite_ratio * pop_size)
        new_population = population[:elite_size]

        # Generar nueva población
        while len(new_population) < pop_size:
            # Seleccionar padres por torneo
            parent1 = tournament_selection(population, k=tournament_k)
            parent2 = tournament_selection(population, k=tournament_k)
            # Generar hijo mediante cruce
            child = crossover(parent1, parent2)
            # Aplicar mutación al hijo
            child = mutate(child, mutation_rate=mutation_rate)
            # Evaluar fitness del hijo
            child.get_fitness(target_array)
            new_population.append(child)

        population = new_population

        # Mostrar imagen cada 50 generaciones
        if gen % 50 == 0:
            plt.imshow(best.image)
            plt.title("Generación {}".format(gen))
            plt.axis("off")
            plt.show()
            best.image.save(os.path.join(output_folder, f"generacion_{gen}.png"))

    # Al final, se devuelve el mejor individuo y la evolución del fitness
    population.sort(key=lambda ind: ind.fitness)
    best = population[0]
    return best, best_fitness_history, target_img


if __name__ == "__main__":
    best_ind, fitness_history, target_img = genetic_algorithm("mona.jpg", pop_size=80, generations=3000, tournament_k=4,
                                                  mutation_rate=0.3, elite_ratio=0.3)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(target_img)
    plt.title("Imagen Objetivo")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(best_ind.image)
    plt.title("Mejor Imagen Generada")
    plt.axis("off")
    plt.show()