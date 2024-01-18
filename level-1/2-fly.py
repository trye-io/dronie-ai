import pygame
from djitellopy import Tello
import numpy as np 
import cv2 
import mediapipe as mp 
from helpers import draw_landmarks 
import threading # завантажуємо threding для асинхронного виконання


BaseOptions = mp.tasks.BaseOptions 
GestureRecognizer = mp.tasks.vision.GestureRecognizer 
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions 
VisionRunningMode = mp.tasks.vision.RunningMode 

MODEL_PATH = 'level-1/gesture_recognizer.task'

def render_frame(result, output_image, timestamp_ms):

    global is_flying # позначаємо що змінна є глобальною

    frame = draw_landmarks(output_image.numpy_view(), result)

    frame = np.rot90(frame)
    frame = np.flipud(frame) 
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))

    # проходимо по усім жестам
    if result.gestures:
        for gesture in result.gestures:
            if gesture[0].category_name == "Thumb_Up" and not(is_flying):
                # створюємо новий потік для зльоту
                threading.Thread(target=drone.takeoff).start()
                is_flying = True
            if gesture[0].category_name == "Thumb_Down" and is_flying:
                # створюємо новий потік для посадки
                threading.Thread(target=drone.land).start()
                is_flying = False


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM, 
    result_callback=render_frame 
)

WIDTH = 960
HEIGHT = 720

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
FPS = 30 
clock = pygame.time.Clock() 

drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read()
is_flying = False # статус дрона, True -- в польоті
                  # False -- на землі

timestamp = 0 
is_running = True

with GestureRecognizer.create_from_options(options) as recognizer:
    while is_running: 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # якщо виходимо з програми, і ми ще в польоті, сідаємо 
                if is_flying: 
                    threading.Thread(target=drone.land).start()
                    is_flying = False
                drone.streamoff()
                is_running = False
            # заради безпеки, забезпечуємо зліт та посадку за допомогою 
            # клавіш T (злетіти) та L (сісти)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t and not(is_flying):
                    is_flying = True
                    threading.Thread(target=drone.takeoff).start()
                if event.key == pygame.K_l and is_flying:
                    is_flying = False
                    threading.Thread(target=drone.land).start()

        frame = frame_read.frame 

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        recognizer.recognize_async(
            mp_image,
            timestamp 
        )

        timestamp += 1 

        pygame.display.flip() 
        clock.tick(FPS)