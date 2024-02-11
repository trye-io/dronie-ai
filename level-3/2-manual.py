import pygame
from djitellopy import Tello
import numpy as np 
import cv2 
from ultralytics import YOLO
import numpy # завантажуємо numpy для типу змінної
import threading # завантажуємо threding для потокового виконання

WIDTH = 640
HEIGHT = 480
# TARGET = (WIDTH / 2, 135) # координату висоти взяли з власного пролітання через рамку

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
FPS = 5 # Нижча кількість FPS 
clock = pygame.time.Clock() 

MODEL_PATH = "level-3/box-detector.pt"
model = YOLO(MODEL_PATH)

drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read()

left_right_velocity = 0
forward_backward_velocity = 0
up_down_velocity = 0
yaw_velocity = 0

is_running = True

status = 0
GROUNDED = 0
MANUAL = 1 # manual mode
TRACKING = 2 # tracking mode

# Kp = (0.125, 0.125)

while is_running: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            if status != GROUNDED: 
                threading.Thread(target=drone.land).start()
                status = GROUNDED
            drone.streamoff()
            is_running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_0 and status != GROUNDED:
                # drone.send_rc_control(0, 0, 0, 0)
                threading.Thread(target=drone.land).start()
                status = GROUNDED
            if event.key == pygame.K_1:
                if status == GROUNDED:
                    threading.Thread(target=drone.takeoff).start()
                if status == TRACKING:
                    drone.send_rc_control(0, 0, 0, 0)
                status = MANUAL
            if event.key == pygame.K_2 and status == MANUAL:
                status = TRACKING

            if event.key == pygame.K_LEFT:
                left_right_velocity = 50
            if event.key == pygame.K_RIGHT:
                left_right_velocity = -50
            if event.key == pygame.K_UP:
                forward_backward_velocity = 50
            if event.key == pygame.K_DOWN:
                forward_backward_velocity = -50
            if event.key == pygame.K_w:
                up_down_velocity = 50
            if event.key == pygame.K_s:
                up_down_velocity = -50
            if event.key == pygame.K_a:
                yaw_velocity = 50
            if event.key == pygame.K_d:
                yaw_velocity = -50
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

    
    if status == MANUAL:
        drone.send_rc_control(
            left_right_velocity,
            forward_backward_velocity,
            up_down_velocity,
            yaw_velocity
        )
    if status == TRACKING:
        print("tracking")
        
    pygame.display.flip() 
    clock.tick(FPS)