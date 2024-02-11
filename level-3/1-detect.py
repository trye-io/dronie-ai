import pygame
from djitellopy import Tello
import numpy as np 
import cv2 
from ultralytics import YOLO # завантажуємо ultralytics

WIDTH = 640 # менша ширина
HEIGHT = 480 # менша висота

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
FPS = 5 # менша частота кадрів
clock = pygame.time.Clock() 

model = YOLO("level-3/box-detector.pt") # завантажуємо модель

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

    # виявляємо обмежувальні коробки на зображенні
    results = model.predict(frame)

    # проходимо по усім виявленим коробкам та візуалізуємо їх
    for bbox in results[0].boxes:
        xyxy = bbox.numpy().xyxy.astype(np.int8).flatten()
        cv2.rectangle(
            frame,
            (xyxy[0], xyxy[1]),
            (xyxy[2], xyxy[3]),
            color=(0, 0, 255),
            thickness=2
        )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = np.flipud(frame) 
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))
        
    pygame.display.flip() 
    clock.tick(FPS)