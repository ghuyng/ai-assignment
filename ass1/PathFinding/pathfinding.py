# Start state: 
# - Start point with position
# - End point with position
# - Barriers with position

# Rules to move: suppose that we move value 0
# - Start point can move up (x, y+1)
# - Start point can move down (x, y-1)
# - Start point can move left (x-1, y)
# - Start point can move right (x+1,y)

# Heuristic function: 
# - cost function: f(x) = h(x) + g(x)
#   + h(x) is the heuristic function equal to abs(x_cur-x_end) + abs(y_cur-y_end)
#   + g(x) is the shortest path from start node to x

# End state: 
# - Start point reach End point : x_start = x_end, y_start = y_end

import pygame
from draw_UI import RED, GREEN, BLUE, YELLOW, WHITE, BLACK , PURPLE , ORANGE, GREY, TURQUOISE
from draw_UI import make_grid, draw, draw_grid
from AStarAlgorithm import AStar as algorithm

ROWS = 50
WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

def get_clicked_pos(pos, rows, width):
	gap = width // rows
	y, x = pos

	row = y // gap
	col = x // gap

	return row, col

def main():

    grid = make_grid(ROWS, WIDTH)

    start = None
    end = None

    run = True
    while run:
        draw(WIN, grid, ROWS, WIDTH)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]: # LEFT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, WIDTH)
                spot = grid[row][col]
                if not start and spot != end:
                    start = spot
                    start.make_start()

                elif not end and spot != start:
                    end = spot
                    end.make_end()

                elif spot != end and spot != start:
                    spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]: # RIGHT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, WIDTH)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)

                    algorithm(lambda: draw(WIN, grid, ROWS, WIDTH), grid, start, end)

                # press C in keyboard to stop game
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, WIDTH)

    pygame.quit()

main()