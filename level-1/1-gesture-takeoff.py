import pygame
from djitellopy import Tello
import numpy as np 
import cv2 
import mediapipe as mp
# from visualize_hand_landmark_detection import draw_landmarks_on_image

# ---
# Loading and setting up mediapipe
MODEL_PATH = 'level-1/gesture_recognizer.task'
WIDTH = 384# 1280
HEIGHT = 288# 720

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def render_frame(result, output_image, timestamp_ms):
    print("1")

    global is_flying

    frame = output_image.numpy_view()

    # frame = draw_landmarks_on_image(output_image.numpy_view(), result)

    # if result.gestures:
    #     for gesture in result.gestures:
    #         print(gesture[0].category_name)
    #         if gesture[0].category_name == "Thumb_Up" and not(is_flying):
    #             drone.takeoff()
    #         if gesture[0].category_name == "Thumb_Down" and is_flying:
    #             drone.land()

    frame = cv2.resize(frame, (WIDTH, HEIGHT)) 
    frame = np.rot90(frame)
    frame = np.flipud(frame) 
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0,0))

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=render_frame
)

timestamp = 0
# ---

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
FPS = 30 
clock = pygame.time.Clock() 

is_running = True
is_flying = False

drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read() 

with GestureRecognizer.create_from_options(options) as recognizer:
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

    timestamp += 1

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    recognizer.recognize_async(
        mp_image,
        timestamp # see https://github.com/google/mediapipe/issues/4448
    )
        
    pygame.display.flip() 
    clock.tick(FPS)