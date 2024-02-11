import pygame
from djitellopy import Tello
import numpy as np 
import cv2 
from ultralytics import YOLO # Вантажемо ultralytcs
import numpy

WIDTH = 640
HEIGHT = 480

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
FPS = 5 # Нижча кількість FPS 
clock = pygame.time.Clock() 

model = YOLO("level-3/box-detector.pt")

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
    frame = cv2.resize(frame, (WIDTH, HEIGHT)) # зменшуємо розмір зображення

    results = model.predict(frame)

    for bbox in results[0].boxes:
        xyxy = bbox.numpy().xyxy.astype(numpy.int64).flatten()
        cv2.rectangle(
            frame,
            (xyxy[0], xyxy[1]),
            (xyxy[2], xyxy[3]),
            color=(255, 0, 0),
            thickness=2
        )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = np.flipud(frame) 
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))
        
    pygame.display.flip() 
    clock.tick(FPS)