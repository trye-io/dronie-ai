import pygame
from djitellopy import Tello
import numpy as np 
import cv2 
import mediapipe as mp # завантажуємо mediapipe

# створюємо псевдоніми 
BaseOptions = mp.tasks.BaseOptions # базова конфігурація
FaceDetector = mp.tasks.vision.FaceDetector # створення детектора
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions # конфігурація виявлення обличчя
VisionRunningMode = mp.tasks.vision.RunningMode # змінна, яка містить три режими використання моделі: фото, відео та пряма трансляція 


MODEL_PATH = 'level-2/blaze_face_short_range.tflite' # шлях до моделі

def render_frame(result, output_image, timestamp_ms):
    
    # конвертуємо назад у Numpy масив
    frame = output_image.numpy_view()

    # малюємо обмежувальну коробку для усіх виявлених облич
    # і друкуємо їхні координати
    if result.detections:
        for detection in result.detections:
            bbox = detection.bounding_box
            cv2.rectangle(
                frame,
                (bbox.origin_x, bbox.origin_y),
                (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                color=(255,0,0),
                thickness=2
            )
            print(bbox)

    # безпосереднє зображення на екрані
    frame = np.rot90(frame)
    frame = np.flipud(frame) 
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0,0))

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM, # режим використання 
    result_callback=render_frame # функція, яка буде викликатись, коли модель виявлення обличчя 
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

timestamp = 0 # лічильник, необхідний для методу .detect_async()
is_running = True

# ініціалізуємо детектор і використовуємо його в циклі
with FaceDetector.create_from_options(options) as detector:
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

        # переводимо зображення у формат, який зрозумілий фунції .detect_async()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # викликаємо фунцію, яка виявляє обличчя / використовуємо детектор
        detector.detect_async(
            mp_image,
            timestamp 
        )

        timestamp += 1 # підвищуємо лічильник на один

        pygame.display.flip() 
        clock.tick(FPS)