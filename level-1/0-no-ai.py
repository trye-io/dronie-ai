import pygame
from djitellopy import Tello
import numpy as np 
import cv2 

pygame.init()
screen = pygame.display.set_mode((960, 720))
FPS = 30 
clock = pygame.time.Clock() 

is_running = True
is_flying = False

drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read() 

while is_running: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:

            if is_flying: 
                drone.land()
                is_flying = False

            drone.streamoff()
            
            is_running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t and not(is_flying):
                drone.takeoff()
                is_flying = True
            if event.key == pygame.K_l and is_flying:
                drone.land()
                is_flying = False

    frame = frame_read.frame 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    frame = np.rot90(frame) 
    frame = np.flipud(frame) 
    frame = pygame.surfarray.make_surface(frame) 
    screen.blit(frame, (0, 0)) 

    pygame.display.flip() 

    clock.tick(FPS)