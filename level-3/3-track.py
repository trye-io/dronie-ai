import pygame
from djitellopy import Tello
import numpy as np 
import cv2 
from ultralytics import YOLO
import threading 

WIDTH = 640
HEIGHT = 480
TARGET_X = WIDTH / 2 # середина екрану
TARGET_Y = 120 # координату висоти взяли з власного пролітання через рамку

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
FPS = 5
clock = pygame.time.Clock() 

model = YOLO("level-3/box-detector.pt")

drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read()

left_right_velocity = 0
forward_backward_velocity = 0
up_down_velocity = 0
yaw_velocity = 0

is_running = True

GROUNDED = 0
MANUAL = 1
AUTOPILOT = 2
mode = GROUNDED

box_x = 0 # координата x центра обмежувальної коробки 
box_y = 0 # координата y центра обмежувальної коробки 

Kp_x = -0.2 # коефіцієнт пропроційного регулятора для осі x 
Kp_y = 0.3 # коефіцієнт пропроційного регулятора для осі y 

while is_running: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            if mode != GROUNDED: 
                threading.Thread(target=drone.land).start()
            drone.streamoff()
            is_running = False
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

    results = model.predict(frame, verbose=False) # вимикаємо детальний друк

    for bbox in results[0].boxes:
        xyxy = bbox.numpy().xyxy.astype(np.int_).flatten()
        # розраховуємо цетр обмежувальної коробки 
        box_x = xyxy[0] + (xyxy[2] - xyxy[0]) / 2 
        box_y = xyxy[1] + (xyxy[3] - xyxy[1]) / 2
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

    # розраховуємо швидкості для автопілоту 
    if mode == AUTOPILOT:
        if results[0].boxes:
            error_x = TARGET_X - box_x
            error_y = TARGET_Y - box_y
            
            left_right_velocity = int(Kp_x * error_x)
            forward_backward_velocity = 40
            up_down_velocity = int(Kp_y * error_y)
            yaw_velocity = 0
        else:
            left_right_velocity = 0
            forward_backward_velocity = 40
            up_down_velocity = 0
            yaw_velocity = 0

    if mode == MANUAL or mode == AUTOPILOT:
        drone.send_rc_control(
            left_right_velocity,
            forward_backward_velocity,
            up_down_velocity,
            yaw_velocity
        )
        
    pygame.display.flip() 
    clock.tick(FPS)