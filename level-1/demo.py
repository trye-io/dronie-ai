from djitellopy import Tello
import pygame
import cv2
import numpy as np

pygame.init()
screen = pygame.display.set_mode((1000, 1000))

drone = Tello()
drone.connect()
drone.set_video_direction(Tello.CAMERA_DOWNWARD) # рядок, який змінює камеру
drone.streamon()
frame_read = drone.get_frame_read()

frame = frame_read.frame

running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            drone.streamoff()
            running = False
    
    frame = frame_read.frame
    print(frame.shape)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = np.rot90(frame)
    frame = np.flipud(frame)

    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))
    pygame.display.flip()