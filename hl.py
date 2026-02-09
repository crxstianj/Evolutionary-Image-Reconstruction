import numpy as np
from PIL import Image, ImageDraw
import random
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


def reconstruir_imagen_con_triangulos(img_path, num_puntos=100):
    # Cargar la imagen con Pillow
    img = Image.open(img_path)
    img_array = np.array(img)
    altura, ancho = img_array.shape[:2]

    # Crear puntos iniciales, incluyendo las esquinas
    puntos = [(0, 0), (0, altura - 1), (ancho - 1, 0), (ancho - 1, altura - 1)]

    # Añadir puntos aleatorios
    for _ in range(num_puntos - 4):
        x = random.randint(0, ancho - 1)
        y = random.randint(0, altura - 1)
        puntos.append((x, y))

    # Calcular triangulación de Delaunay
    puntos = np.array(puntos)
    triangulacion = Delaunay(puntos)
    triangulos = puntos[triangulacion.simplices]

    # Crear imagen final
    img_final = Image.new('RGB', (ancho, altura), (0, 0, 0))
    draw = ImageDraw.Draw(img_final)

    # Para cada triángulo
    for tri in triangulos:
        # Convertir a tuplas para PIL
        tri_puntos = [(int(x), int(y)) for x, y in tri]

        # Crear una máscara para el triángulo
        mascara = Image.new('L', (ancho, altura), 0)
        ImageDraw.Draw(mascara).polygon(tri_puntos, fill=255)
        mascara_array = np.array(mascara)

        # Calcular el color promedio del triángulo en la imagen original
        mascara_bool = mascara_array > 0
        pixeles_en_triangulo = img_array[mascara_bool]

        if len(pixeles_en_triangulo) > 0:
            color_promedio = tuple(map(int, np.mean(pixeles_en_triangulo, axis=0)))
            draw.polygon(tri_puntos, fill=color_promedio)

    return img_final


def optimizar_puntos(img_path, num_puntos=100, iteraciones=1000):
    # Cargar la imagen con Pillow
    img = Image.open(img_path)
    img_array = np.array(img)
    altura, ancho = img_array.shape[:2]

    # Puntos iniciales (incluyendo las esquinas)
    puntos = [(0, 0), (0, altura - 1), (ancho - 1, 0), (ancho - 1, altura - 1)]

    # Añadir puntos aleatorios
    for _ in range(num_puntos - 4):
        x = random.randint(0, ancho - 1)
        y = random.randint(0, altura - 1)
        puntos.append((x, y))

    mejor_error = float('inf')
    mejor_img = None

    # Optimización iterativa
    for i in range(iteraciones):
        # Generar triangulación y reconstruir imagen
        puntos_np = np.array(puntos)
        triangulacion = Delaunay(puntos_np)
        triangulos = puntos_np[triangulacion.simplices]

        img_reconstruida = Image.new('RGB', (ancho, altura), (0, 0, 0))
        draw = ImageDraw.Draw(img_reconstruida)

        for tri in triangulos:
            # Convertir a tuplas para PIL
            tri_puntos = [(int(x), int(y)) for x, y in tri]

            # Crear una máscara para el triángulo
            mascara = Image.new('L', (ancho, altura), 0)
            ImageDraw.Draw(mascara).polygon(tri_puntos, fill=255)
            mascara_array = np.array(mascara)

            # Calcular el color promedio
            mascara_bool = mascara_array > 0
            pixeles_en_triangulo = img_array[mascara_bool]

            if len(pixeles_en_triangulo) > 0:
                color_promedio = tuple(map(int, np.mean(pixeles_en_triangulo, axis=0)))
                draw.polygon(tri_puntos, fill=color_promedio)

        # Calcular error
        img_reconstruida_array = np.array(img_reconstruida)
        error = np.sum((img_array.astype(np.float32) - img_reconstruida_array.astype(np.float32)) ** 2)

        if error < mejor_error:
            mejor_error = error
            mejor_img = img_reconstruida.copy()
            print(f"Iteración {i}: Nuevo mejor error = {mejor_error}")

        # Mover un punto aleatorio (que no sea una esquina)
        if i < iteraciones - 1:  # No mover en la última iteración
            idx = random.randint(4, len(puntos) - 1)
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)

            nuevo_x = max(0, min(ancho - 1, puntos[idx][0] + dx))
            nuevo_y = max(0, min(altura - 1, puntos[idx][1] + dy))

            puntos[idx] = (nuevo_x, nuevo_y)

    return mejor_img


def reconstruir_con_circulos(img_path, num_circulos=200, radio_maximo=30):
    # Cargar la imagen con Pillow
    img = Image.open(img_path)
    img_array = np.array(img)
    altura, ancho = img_array.shape[:2]

    # Crear imagen final
    img_final = Image.new('RGB', (ancho, altura), (0, 0, 0))
    draw = ImageDraw.Draw(img_final)

    # Colocar círculos aleatorios
    for _ in range(num_circulos):
        # Posición aleatoria para el círculo
        x = random.randint(0, ancho - 1)
        y = random.randint(0, altura - 1)

        # Radio aleatorio
        radio = random.randint(5, radio_maximo)

        # Crear una máscara para el círculo
        mascara = Image.new('L', (ancho, altura), 0)
        ImageDraw.Draw(mascara).ellipse((x - radio, y - radio, x + radio, y + radio), fill=255)
        mascara_array = np.array(mascara)

        # Calcular el color promedio del círculo en la imagen original
        mascara_bool = mascara_array > 0
        pixeles_en_circulo = img_array[mascara_bool]

        if len(pixeles_en_circulo) > 0:
            color_promedio = tuple(map(int, np.mean(pixeles_en_circulo, axis=0)))
            draw.ellipse((x - radio, y - radio, x + radio, y + radio), fill=color_promedio)

    return img_final


def reconstruir_con_rectangulos(img_path, num_rectangulos=200, tamaño_maximo=50):
    # Cargar la imagen con Pillow
    img = Image.open(img_path)
    img_array = np.array(img)
    altura, ancho = img_array.shape[:2]

    # Crear imagen final
    img_final = Image.new('RGB', (ancho, altura), (0, 0, 0))
    draw = ImageDraw.Draw(img_final)

    # Colocar rectángulos aleatorios
    for _ in range(num_rectangulos):
        # Posición aleatoria para el rectángulo
        x = random.randint(0, ancho - 1)
        y = random.randint(0, altura - 1)

        # Dimensiones aleatorias
        ancho_rect = random.randint(5, tamaño_maximo)
        alto_rect = random.randint(5, tamaño_maximo)

        # Asegurar que el rectángulo no se salga de la imagen
        x1 = max(0, x - ancho_rect // 2)
        y1 = max(0, y - alto_rect // 2)
        x2 = min(ancho - 1, x + ancho_rect // 2)
        y2 = min(altura - 1, y + alto_rect // 2)

        # Crear una máscara para el rectángulo
        mascara = Image.new('L', (ancho, altura), 0)
        ImageDraw.Draw(mascara).rectangle((x1, y1, x2, y2), fill=255)
        mascara_array = np.array(mascara)

        # Calcular el color promedio del rectángulo en la imagen original
        mascara_bool = mascara_array > 0
        pixeles_en_rectangulo = img_array[mascara_bool]

        if len(pixeles_en_rectangulo) > 0:
            color_promedio = tuple(map(int, np.mean(pixeles_en_rectangulo, axis=0)))
            draw.rectangle((x1, y1, x2, y2), fill=color_promedio)

    return img_final

imagen_reconstruida = reconstruir_imagen_con_triangulos("mona.jpg", num_puntos=2000)
imagen_reconstruida.save("triangulos.jpg")
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.array(Image.open("mona.jpg")))
plt.title("Original")
plt.subplot(122)
plt.imshow(np.array(imagen_reconstruida))
plt.title("Reconstruida con triángulos")
plt.show()

# Círculos:
#imagen_circulos = reconstruir_con_circulos("mona.jpg", num_circulos=800, radio_maximo=30)
#imagen_circulos.save("circulos.jpg")

# Rectángulos:
# imagen_rectangulos = reconstruir_con_rectangulos("mona.jpg", num_rectangulos=300, tamaño_maximo=30)
# imagen_rectangulos.save("rectangulos.jpg")