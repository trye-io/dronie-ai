import pygame
from djitellopy import Tello
import numpy as np 
import cv2 
from ultralytics import YOLO
import threading # завантажуємо threding для потокового виконання

WIDTH = 640
HEIGHT = 480

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
FPS = 5
clock = pygame.time.Clock() 

model = YOLO("level-3/box-detector.pt")

drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read()

# швидкості дрона
left_right_velocity = 0
forward_backward_velocity = 0
up_down_velocity = 0
yaw_velocity = 0

is_running = True

GROUNDED = 0 # режим на землі
MANUAL = 1 # режим ручного керування
AUTOPILOT = 2 # режим автопілота
mode = GROUNDED

while is_running: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            if mode != GROUNDED: # злітаємо
                threading.Thread(target=drone.land).start()
            drone.streamoff()
            is_running = False
        # перемикаємо режим з клавіатури
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_0 and mode != GROUNDED:
                threading.Thread(target=drone.land).start()
                mode = GROUNDED
            if event.key == pygame.K_1:
                if mode == GROUNDED:
                    threading.Thread(target=drone.takeoff).start()
                if mode == AUTOPILOT:
                    left_right_velocity = 0
                    forward_backward_velocity = 0
                    up_down_velocity = 0
                    yaw_velocity = 0
                mode = MANUAL
            if event.key == pygame.K_2 and mode == MANUAL:
                mode = AUTOPILOT
        # зчитуємо швидкості з клавіатури, якщо дрон у ручному режимі
        if mode == MANUAL:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    left_right_velocity = -50
                if event.key == pygame.K_RIGHT:
                    left_right_velocity = 50
                if event.key == pygame.K_UP:
                    forward_backward_velocity = 50
                if event.key == pygame.K_DOWN:
                    forward_backward_velocity = -50
                if event.key == pygame.K_w:
                    up_down_velocity = 50
                if event.key == pygame.K_s:
                    up_down_velocity = -50
                if event.key == pygame.K_a:
                    yaw_velocity = -50
                if event.key == pygame.K_d:
                    yaw_velocity = 50
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    left_right_velocity = 0
                if event.key == pygame.K_RIGHT:
                    left_right_velocity = 0
                if event.key == pygame.K_UP:
                    forward_backward_velocity = 0
                if event.key == pygame.K_DOWN:
                    forward_backward_velocity = 0
                if event.key == pygame.K_w:
                    up_down_velocity = 0
                if event.key == pygame.K_s:
                    up_down_velocity = 0
                if event.key == pygame.K_a:
                    yaw_velocity = 0
                if event.key == pygame.K_d:
                    yaw_velocity = 0

    frame = frame_read.frame 
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    results = model.predict(frame)

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
    
    # якщо дрон не на землі, відправляємо швидкості на нього
    if mode == MANUAL or mode == AUTOPILOT:
        drone.send_rc_control(
            left_right_velocity,
            forward_backward_velocity,
            up_down_velocity,
            yaw_velocity
        )
        
    pygame.display.flip() 
    clock.tick(FPS)