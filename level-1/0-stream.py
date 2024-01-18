import pygame
from djitellopy import Tello
import numpy as np 
import cv2 

WIDTH = 960
HEIGHT = 720

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
FPS = 30 
clock = pygame.time.Clock() 

drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read()

is_running = True

while is_running: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            drone.streamoff()
            is_running = False

    frame = frame_read.frame 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = np.flipud(frame) 
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))
        
    pygame.display.flip() 
    clock.tick(FPS)