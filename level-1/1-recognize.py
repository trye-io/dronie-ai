import pygame
from djitellopy import Tello
import numpy as np 
import cv2 
import mediapipe as mp # завантажуємо mediapipe
from helpers import draw_landmarks # завантажуємо домоміжну функцію для візуалізації

# створюємо псевдоніми 
BaseOptions = mp.tasks.BaseOptions # базова конфігурація
GestureRecognizer = mp.tasks.vision.GestureRecognizer # 
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions # конфігурація розпізнавання жестів
VisionRunningMode = mp.tasks.vision.RunningMode # змінна, яка містить три режими використання моделі: фото, відео та пряма трансляція 

MODEL_PATH = 'level-1/gesture_recognizer.task' # шлях до моделі

def render_frame(result, output_image, timestamp_ms):

    # малюємо ориєнтирні точки на зображенні
    frame = draw_landmarks(output_image.numpy_view(), result)

    # безпосереднє зображення на екрані (скопійовано з циклу)
    frame = np.rot90(frame)
    frame = np.flipud(frame) 
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0,0))

    # проходимо по усім жестам та друкуємо їхні імена
    if result.gestures:
        for gesture in result.gestures:
            print(gesture[0].category_name)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM, # режим використання 
    result_callback=render_frame # функція, яка буде викликатись, коли модель розпізнала жест 
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

timestamp = 0 # лічильник, необхідний для методу .recognize_async()
is_running = True

# ініціалізуємо розпізнавач і використовуємо його в циклі
with GestureRecognizer.create_from_options(options) as recognizer:
    while is_running: 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                drone.streamoff()
                is_running = False

        frame = frame_read.frame 

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # код, закоментований нижче буде виконуватись у функції render_frame()
        # frame = np.rot90(frame)
        # frame = np.flipud(frame) 
        # frame = pygame.surfarray.make_surface(frame)
        # screen.blit(frame, (0,0))

        # переводимо зображення у формат, який зрозумілий фунції .recognize_async()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # викликаємо фунцію, яка розпізнає жест / використовуємо розпізнавач
        recognizer.recognize_async(
            mp_image,
            timestamp 
        )

        timestamp += 1 # підвищуємо лічильник на один

        pygame.display.flip() 
        clock.tick(FPS)