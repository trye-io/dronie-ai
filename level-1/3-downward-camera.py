import pygame
from djitellopy import Tello
import numpy as np 
import cv2 
import mediapipe as mp 
from helpers import draw_landmarks 
import threading 

BaseOptions = mp.tasks.BaseOptions 
GestureRecognizer = mp.tasks.vision.GestureRecognizer 
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions 
VisionRunningMode = mp.tasks.vision.RunningMode 

MODEL_PATH = 'level-1/gesture_recognizer.task'

def render_frame(result, output_image, timestamp_ms):

    global is_flying 

    frame = draw_landmarks(output_image.numpy_view(), result)

    frame = np.rot90(frame)
    frame = np.flipud(frame) 
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0,0))

    if result.gestures:
        for gesture in result.gestures:
            print(gesture[0].category_name)
            # залишаємо тільки одну умову для посадки, і змінюємо жест на Open_Palm
            # треба підходити з правої сторони до дрону
            if gesture[0].category_name == "Open_Palm" and is_flying:
                threading.Thread(target=drone.land).start()
                is_flying = False


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM, 
    result_callback=render_frame 
)

# нова ширина та висота зображення
WIDTH = 320
HEIGHT = 240

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
# Змінемо FPS на меншу кількість 
FPS = 5 
clock = pygame.time.Clock() 

drone = Tello()
drone.connect()
# змінюємо стрім відео на камеру, що внизу
drone.set_video_direction(Tello.CAMERA_DOWNWARD)
drone.streamon()
frame_read = drone.get_frame_read()
is_flying = False 

timestamp = 0 
is_running = True

with GestureRecognizer.create_from_options(options) as recognizer:
    while is_running: 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if is_flying: 
                    threading.Thread(target=drone.land).start()
                    is_flying = False
                drone.streamoff()
                is_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t and not(is_flying):
                    is_flying = True
                    threading.Thread(target=drone.takeoff).start()
                if event.key == pygame.K_l and is_flying:
                    is_flying = False
                    threading.Thread(target=drone.land).start()

        frame = frame_read.frame 

        # отримуємо тільки перші 240 пікселів висоти
        frame = frame[:240, :, :]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        recognizer.recognize_async(
            mp_image,
            timestamp 
        )

        timestamp += 1 

        pygame.display.flip() 
        clock.tick(FPS)