import pygame
from djitellopy import Tello
import numpy as np 
import cv2 

# ---
# Loading and setting up mediapipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'gesture_recognizer.task'

base_options = python.BaseOptions(model_asset_path='level-1/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)
# ---

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

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    recognition_result = recognizer.recognize(mp_image)

    if recognition_result.gestures:
        for gesture in recognition_result.gestures:
            print(gesture[0].category_name)
            if gesture[0].category_name == "Thumb_Up" and not(is_flying):
                drone.takeoff()
            if gesture[0].category_name == "Thumb_Down" and is_flying:
                drone.land()
    
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    frame = np.rot90(frame) 
    frame = np.flipud(frame) 
    frame = pygame.surfarray.make_surface(frame) 
    screen.blit(frame, (0, 0)) 

    pygame.display.flip() 

    clock.tick(FPS)