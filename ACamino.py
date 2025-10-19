import pygame
import heapq

pygame.init()
pygame.font.init()

ANCHO_VENTANA = 600
FILAS = 10
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA + 100))
pygame.display.set_caption("Visualización de A*")

BLANCO = (23, 32, 42)
NEGRO = (255,255,255)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (118, 215, 196)
PURPURA = (128, 0, 128)
AZUL = (0, 0, 255)

COSTO_HORIZONTAL_VERTICAL = 10
COSTO_DIAGONAL = 14

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.g = float("inf")
        self.h = 0
        self.f = float("inf")
        self.padre = None
        self.numero = col * total_filas + fila + 1

    def __lt__(self, otro):
        return self.f < otro.f

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_camino(self):
        if not self.es_inicio() and not self.es_fin():
            self.color = AZUL

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))
        font = pygame.font.Font(None, 20)
        text = font.render(str(self.numero), True, NEGRO)
        ventana.blit(text, (self.x + 5, self.y + 20))

def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def heuristica(nodo1, nodo2):
    x1, y1 = nodo1.get_pos()
    x2, y2 = nodo2.get_pos()
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruir_camino(nodo_actual):
    camino = []
    while nodo_actual.padre:
        camino.append(nodo_actual.numero)
        nodo_actual.hacer_camino()
        nodo_actual = nodo_actual.padre
    return camino[::-1]

def obtener_vecinos(nodo, grid):
    vecinos = []
    filas = len(grid)
    direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for d in direcciones:
        fila, col = nodo.fila + d[0], nodo.col + d[1]
        if 0 <= fila < filas and 0 <= col < filas:
            vecino = grid[fila][col]
            if not vecino.es_pared():
                vecinos.append((vecino, COSTO_DIAGONAL if d[0] != 0 and d[1] != 0 else COSTO_HORIZONTAL_VERTICAL))
    return vecinos

def algoritmo_a_estrella(grid, inicio, fin):
    inicio.g = 0
    inicio.h = heuristica(inicio, fin)
    inicio.f = inicio.h
    open_set = [(inicio.f, inicio)]
    heapq.heapify(open_set)

    while open_set:
        _, nodo_actual = heapq.heappop(open_set)

        if nodo_actual == fin:
            camino = reconstruir_camino(fin)
            print("Ruta más corta:", camino)
            return True

        for vecino, costo in obtener_vecinos(nodo_actual, grid):
            nuevo_g = nodo_actual.g + costo
            if nuevo_g < vecino.g:
                vecino.g = nuevo_g
                vecino.h = heuristica(vecino, fin)
                vecino.f = vecino.g + vecino.h
                vecino.padre = nodo_actual
                heapq.heappush(open_set, (vecino.f, vecino))
    return False

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    pygame.display.update()

def main():
    grid = crear_grid(FILAS, ANCHO_VENTANA)
    inicio = None
    fin = None
    boton_buscar_rect = pygame.Rect(ANCHO_VENTANA // 2 - 50, ANCHO_VENTANA + 10, 100, 30)
    boton_salir_rect = pygame.Rect(ANCHO_VENTANA // 2 - 50, ANCHO_VENTANA + 50, 100, 30)
    corriendo = True

    while corriendo:
        dibujar(VENTANA, grid, FILAS, ANCHO_VENTANA)
        pygame.draw.rect(VENTANA, VERDE, boton_buscar_rect)
        font = pygame.font.Font(None, 24)
        text = font.render("Buscar", True, NEGRO)
        VENTANA.blit(text, (ANCHO_VENTANA // 2 - 25, ANCHO_VENTANA + 15))
        pygame.draw.rect(VENTANA, ROJO, boton_salir_rect)
        text = font.render("Salir", True, NEGRO)
        VENTANA.blit(text, (ANCHO_VENTANA // 2 - 25, ANCHO_VENTANA + 55))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                if boton_buscar_rect.collidepoint(pos) and inicio and fin:
                    encontrado = algoritmo_a_estrella(grid, inicio, fin)
                    dibujar(VENTANA, grid, FILAS, ANCHO_VENTANA)

                    if encontrado:
                        esperando = True
                        while esperando:
                            for evento in pygame.event.get():
                                if evento.type == pygame.QUIT:
                                    corriendo = False
                                    esperando = False
                                elif evento.type == pygame.KEYDOWN or evento.type == pygame.MOUSEBUTTONDOWN:
                                    esperando = False

                    if not encontrado:
                        font = pygame.font.Font(None, 36)
                        mensaje = font.render("No hay camino posible", True, ROJO)
                        VENTANA.blit(mensaje, (ANCHO_VENTANA // 2 - 100, ANCHO_VENTANA // 2))
                        pygame.display.update()
                        pygame.time.delay(2000)
                        corriendo = False

                elif boton_salir_rect.collidepoint(pos):
                    corriendo = False

                else:
                    fila, col = obtener_click_pos(pos, FILAS, ANCHO_VENTANA)
                    nodo = grid[fila][col]
                    if not inicio and nodo != fin:
                        inicio = nodo
                        inicio.hacer_inicio()
                    elif not fin and nodo != inicio:
                        fin = nodo
                        fin.hacer_fin()
                    elif nodo != fin and nodo != inicio:
                        nodo.hacer_pared()

    pygame.quit()

main()