import numpy as np
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
import random
import os
import colour
from multiprocessing import Pool

# Crear carpeta para guardar resultados
output_folder = "pso_reconstruction"
os.makedirs(output_folder, exist_ok=True)


# Clase para representar una partícula (una posible solución)
class Particle:
    def __init__(self, width, height, num_shapes=None):
        self.width = width
        self.height = height
        self.fitness = float('inf')
        self.image = None
        self.array = None

        # Posición actual (lista de formas con sus propiedades)
        self.position = []
        # Mejor posición encontrada por esta partícula
        self.best_position = []
        self.best_fitness = float('inf')

        # Velocidad (para cada dimensión)
        self.velocity = []

        # Inicializar con formas aleatorias
        if num_shapes is None:
            num_shapes = random.randint(3, 10)

        self.initialize_shapes(num_shapes)
        self.render_image()

    def initialize_shapes(self, num_shapes):
        """Inicializa la partícula con formas aleatorias"""
        self.position = []
        region = (self.width + self.height) // 8

        for _ in range(num_shapes):
            shape_type = random.choice(['polygon', 'circle', 'rectangle'])
            shape = {
                'type': shape_type,
                'color': self.rand_color(),
                'center_x': random.randint(0, self.width),
                'center_y': random.randint(0, self.height),
                'size': random.randint(10, region),
                'points': random.randint(3, 8) if shape_type == 'polygon' else 0,
                'rotation': random.uniform(0, 360)
            }
            self.position.append(shape)

            # Inicializar velocidad con valores pequeños aleatorios
            vel = {
                'center_x': random.uniform(-5, 5),
                'center_y': random.uniform(-5, 5),
                'size': random.uniform(-3, 3),
                'color_delta': [random.uniform(-0.1, 0.1) for _ in range(3)],
                'rotation': random.uniform(-5, 5)
            }
            self.velocity.append(vel)

        self.best_position = self.deep_copy_position(self.position)

    def deep_copy_position(self, pos):
        """Hace una copia profunda de la posición"""
        return [{k: v for k, v in shape.items()} for shape in pos]

    def rand_color(self):
        """Devuelve un color aleatorio en formato RGBA"""
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(100, 255)  # Alpha para transparencia parcial
        )

    def render_image(self):
        """Renderiza la imagen basada en la posición actual"""
        img = Image.new("RGBA", (self.width, self.height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)

        for shape in self.position:
            try:
                if shape['type'] == 'polygon':
                    # Crear los puntos del polígono
                    points = []
                    num_points = max(3, shape['points'])  # Asegurar mínimo 3 puntos
                    for i in range(num_points):
                        angle = 2 * np.pi * i / num_points + np.radians(shape['rotation'])
                        x = shape['center_x'] + shape['size'] * np.cos(angle)
                        y = shape['center_y'] + shape['size'] * np.sin(angle)
                        points.append((x, y))

                    # Verificar que tenemos suficientes puntos
                    if len(points) >= 3:
                        draw.polygon(points, fill=shape['color'])

                elif shape['type'] == 'circle':
                    # Dibujar círculo
                    x1 = shape['center_x'] - shape['size']
                    y1 = shape['center_y'] - shape['size']
                    x2 = shape['center_x'] + shape['size']
                    y2 = shape['center_y'] + shape['size']
                    draw.ellipse([x1, y1, x2, y2], fill=shape['color'])

                elif shape['type'] == 'rectangle':
                    # Dibujar rectángulo rotado
                    rect_w = shape['size'] * 1.5
                    rect_h = shape['size']

                    # Crear un rectángulo centrado en el origen
                    points = [
                        (-rect_w / 2, -rect_h / 2),
                        (rect_w / 2, -rect_h / 2),
                        (rect_w / 2, rect_h / 2),
                        (-rect_w / 2, rect_h / 2)
                    ]

                    # Rotar y trasladar los puntos
                    rad = np.radians(shape['rotation'])
                    rot_points = []
                    for x, y in points:
                        rx = x * np.cos(rad) - y * np.sin(rad) + shape['center_x']
                        ry = x * np.sin(rad) + y * np.cos(rad) + shape['center_y']
                        rot_points.append((rx, ry))

                    draw.polygon(rot_points, fill=shape['color'])
            except Exception as e:
                # En caso de error, simplemente omitimos esta forma
                print(f"Error al dibujar forma: {e}")
                continue

        self.image = img
        self.array = np.array(img)

    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        """Actualiza la velocidad de la partícula según PSO"""
        for i, shape in enumerate(self.position):
            if i >= len(self.velocity):
                # Si no hay velocidad para esta forma, la inicializamos
                self.velocity.append({
                    'center_x': random.uniform(-5, 5),
                    'center_y': random.uniform(-5, 5),
                    'size': random.uniform(-3, 3),
                    'color_delta': [random.uniform(-0.1, 0.1) for _ in range(3)],
                    'rotation': random.uniform(-5, 5)
                })

            # Factor de inercia
            self.velocity[i]['center_x'] = w * self.velocity[i]['center_x']
            self.velocity[i]['center_y'] = w * self.velocity[i]['center_y']
            self.velocity[i]['size'] = w * self.velocity[i]['size']
            self.velocity[i]['rotation'] = w * self.velocity[i]['rotation']

            # Componente cognitiva (mejor posición personal)
            if i < len(self.best_position):
                r1 = random.random()
                self.velocity[i]['center_x'] += c1 * r1 * (self.best_position[i]['center_x'] - shape['center_x'])
                self.velocity[i]['center_y'] += c1 * r1 * (self.best_position[i]['center_y'] - shape['center_y'])
                self.velocity[i]['size'] += c1 * r1 * (self.best_position[i]['size'] - shape['size'])
                self.velocity[i]['rotation'] += c1 * r1 * (self.best_position[i]['rotation'] - shape['rotation'])

            # Componente social (mejor posición global)
            if i < len(global_best_position):
                r2 = random.random()
                self.velocity[i]['center_x'] += c2 * r2 * (global_best_position[i]['center_x'] - shape['center_x'])
                self.velocity[i]['center_y'] += c2 * r2 * (global_best_position[i]['center_y'] - shape['center_y'])
                self.velocity[i]['size'] += c2 * r2 * (global_best_position[i]['size'] - shape['size'])
                self.velocity[i]['rotation'] += c2 * r2 * (global_best_position[i]['rotation'] - shape['rotation'])

                # Color (tratamos cada componente por separado)
                for j in range(3):  # RGB
                    if 'color_delta' not in self.velocity[i]:
                        self.velocity[i]['color_delta'] = [0, 0, 0]
                    self.velocity[i]['color_delta'][j] = w * self.velocity[i]['color_delta'][j]

                    if i < len(self.best_position):
                        self.velocity[i]['color_delta'][j] += c1 * r1 * (
                                    self.best_position[i]['color'][j] - shape['color'][j])

                    if i < len(global_best_position):
                        self.velocity[i]['color_delta'][j] += c2 * r2 * (
                                    global_best_position[i]['color'][j] - shape['color'][j])

    def update_position(self):
        """Actualiza la posición basada en la velocidad"""
        for i, shape in enumerate(self.position):
            if i < len(self.velocity):
                # Actualizar posición
                shape['center_x'] += int(self.velocity[i]['center_x'])
                shape['center_y'] += int(self.velocity[i]['center_y'])
                shape['size'] += int(self.velocity[i]['size'])
                shape['rotation'] += self.velocity[i]['rotation']

                # Asegurar tamaño mínimo
                shape['size'] = max(5, shape['size'])

                # Mantener dentro de los límites
                shape['center_x'] = max(0, min(self.width, shape['center_x']))
                shape['center_y'] = max(0, min(self.height, shape['center_y']))

                # Actualizar color con límites
                if 'color_delta' in self.velocity[i]:
                    new_color = list(shape['color'])
                    for j in range(3):  # RGB
                        new_color[j] = max(0, min(255, int(new_color[j] + self.velocity[i]['color_delta'][j])))
                    # Mantener alpha original
                    shape['color'] = tuple(new_color[:3]) + (shape['color'][3],)

        # Re-renderizar la imagen con la nueva posición
        self.render_image()

    def evaluate_fitness(self, target_array):
        """Evalúa el fitness usando Delta E (diferencia de color perceptual)"""
        try:
            # Asegurar que ambas imágenes tengan el mismo tamaño
            if self.array.shape != target_array.shape:
                raise ValueError("Las dimensiones de las imágenes no coinciden")

            # Calcular diferencia media de color
            self.fitness = np.mean(colour.difference.delta_e.delta_E_CIE1976(
                target_array[:, :, :3],  # Excluir canal alpha para el cálculo
                self.array[:, :, :3]
            ))

            # Actualizar mejor posición si es necesario
            if self.fitness < self.best_fitness:
                self.best_fitness = self.fitness
                self.best_position = self.deep_copy_position(self.position)

            return self.fitness
        except Exception as e:
            print(f"Error en evaluate_fitness: {e}")
            self.fitness = float('inf')
            return float('inf')


# Evaluación paralela del fitness
def evaluate_particle(args):
    particle, target_array = args
    try:
        particle.evaluate_fitness(target_array)
    except Exception as e:
        print(f"Error en evaluate_particle: {e}")
        particle.fitness = float('inf')
    return particle


# Algoritmo PSO principal
def particle_swarm_optimization(target_image_path, num_particles=30, iterations=1000, w=0.7, c1=1.5, c2=1.5):
    # Cargar imagen objetivo
    target_img = Image.open(target_image_path).convert("RGBA")
    target_img = ImageOps.fit(target_img, (256, 256), method=Image.LANCZOS)
    width, height = target_img.size
    target_array = np.array(target_img)

    # Inicializar enjambre de partículas
    particles = [Particle(width, height) for _ in range(num_particles)]

    # Evaluar fitness inicial
    particles_with_fitness = []
    for p in particles:
        try:
            p.evaluate_fitness(target_array)
            particles_with_fitness.append(p)
        except Exception as e:
            print(f"Error al evaluar partícula inicial: {e}")

    # Si todas las partículas fallaron, crear nuevas
    if not particles_with_fitness:
        print("Todas las partículas iniciales fallaron. Creando nuevas...")
        particles = [Particle(width, height) for _ in range(num_particles)]
        for p in particles:
            try:
                p.evaluate_fitness(target_array)
                particles_with_fitness.append(p)
            except Exception as e:
                print(f"Error al evaluar partícula nueva: {e}")

    # Si aún no hay partículas válidas, salir
    if not particles_with_fitness:
        raise RuntimeError("No se pudo inicializar ninguna partícula válida")

    particles = particles_with_fitness

    # Encontrar mejor posición global
    global_best_particle = min(particles, key=lambda p: p.fitness)
    global_best_position = global_best_particle.deep_copy_position(global_best_particle.position)
    global_best_fitness = global_best_particle.fitness

    fitness_history = [global_best_fitness]
    print(f"Iteración 0: Mejor fitness = {global_best_fitness:.2f}")

    # Guardar imagen inicial
    global_best_particle.image.save(os.path.join(output_folder, "iteracion_0.png"))

    # Bucle principal de PSO
    for i in range(1, iterations + 1):
        # Ajustar el factor de inercia (reducir linealmente con el tiempo)
        w_current = w - (w - 0.4) * (i / iterations)

        # Actualizar cada partícula
        for p in particles:
            try:
                p.update_velocity(global_best_position, w=w_current, c1=c1, c2=c2)
                p.update_position()
            except Exception as e:
                print(f"Error al actualizar partícula: {e}")

        # Evaluar fitness actualizado
        particles_with_fitness = []
        for p in particles:
            try:
                p.evaluate_fitness(target_array)
                particles_with_fitness.append(p)
            except Exception as e:
                print(f"Error al evaluar fitness: {e}")

        # Si hay partículas válidas, actualizar
        if particles_with_fitness:
            particles = particles_with_fitness

            # Actualizar mejor posición global si es necesario
            current_best = min(particles, key=lambda p: p.fitness)
            if current_best.fitness < global_best_fitness:
                global_best_fitness = current_best.fitness
                global_best_position = current_best.deep_copy_position(current_best.position)
                global_best_particle = current_best
                print(f"Iteración {i}: Nuevo mejor fitness = {global_best_fitness:.2f}")

                # Guardar mejores imágenes
                if i % 20 == 0 or i == iterations:
                    global_best_particle.image.save(os.path.join(output_folder, f"iteracion_{i}.png"))

        fitness_history.append(global_best_fitness)

        # Condición de parada opcional
        if global_best_fitness < 20.0:
            print(f"Convergencia alcanzada en iteración {i}")
            break

        # Mostrar progreso
        if i % 50 == 0:
            print(f"Iteración {i}: Mejor fitness = {global_best_fitness:.2f}")
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 3, 1)
            plt.imshow(target_img)
            plt.title("Imagen Objetivo")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(global_best_particle.image)
            plt.title(f"Mejor Reconstrucción\nFitness: {global_best_fitness:.2f}")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.plot(fitness_history)
            plt.title("Evolución del Fitness")
            plt.xlabel("Iteraciones")
            plt.ylabel("Fitness (Delta E)")

            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"progreso_{i}.png"))
            plt.show()

    # Mostrar resultados finales
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(target_img)
    plt.title("Imagen Objetivo")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(global_best_particle.image)
    plt.title(f"Mejor Reconstrucción\nFitness: {global_best_fitness:.2f}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "resultado_final.png"))
    plt.show()

    # Guardar la evolución del fitness
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history)
    plt.title("Evolución del Fitness")
    plt.xlabel("Iteraciones")
    plt.ylabel("Fitness (Delta E)")
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "evolucion_fitness.png"))
    plt.show()

    return global_best_particle, fitness_history, target_img


if __name__ == "__main__":
    best_particle, fitness_history, target_img = particle_swarm_optimization(
        "mona.jpg",
        num_particles=30,
        iterations=500,
        w=0.7,  # Factor de inercia inicial
        c1=1.5,  # Factor cognitivo
        c2=1.5  # Factor social
    )

    # Guarda la mejor imagen resultante
    best_particle.image.save(os.path.join(output_folder, "mejor_reconstruccion.png"))

    print(f"Reconstrucción completada. Fitness final: {best_particle.fitness:.2f}")
    print(f"Resultados guardados en la carpeta '{output_folder}'")
