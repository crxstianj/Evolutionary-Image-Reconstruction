# Evolutionary Image Reconstruction

## ¿Cómo funciona?

El sistema parte de una imagen objetivo y busca aproximarla colocando formas geométricas (polígonos, círculos, rectángulos, triángulos) sobre un lienzo en blanco. Cada algoritmo evalúa qué tan cerca está la imagen generada de la original usando métricas de error como **Delta E (CIE1976)** o distancia euclidiana, y va mejorando iterativamente.

## Algoritmos implementados

| Archivo | Algoritmo | Descripción |
|---------|-----------|-------------|
| `AG.py` | Algoritmo Genético | Evoluciona una población de individuos (imágenes con polígonos) mediante selección, cruce y mutación |
| `PSO.py` | Particle Swarm Optimization | Cada partícula representa una imagen; las partículas se mueven en el espacio de soluciones guiadas por su mejor posición y la del enjambre |
| `hl.py` | Heurística Local (Delaunay) | Triangula la imagen usando puntos aleatorios y optimiza su posición iterativamente para minimizar el error |

## Tecnologías

- `Pillow` — generación y manipulación de imágenes
- `NumPy` — operaciones matriciales sobre píxeles
- `scikit-learn / scipy` — triangulación de Delaunay
- `colour` — métrica perceptual Delta E
- `multiprocessing` — evaluación paralela del fitness
- `matplotlib` — visualización del proceso

## Uso

Cada script es independiente. Ejemplo con el algoritmo genético:
```bash
python AG.py
```

Los resultados se guardan en carpetas de salida:

- `AG.py` → `prueba/`
- `PSO.py` → `pso_reconstruction/`

## Dependencias
```bash
pip install numpy pillow matplotlib colour-science scipy
```
