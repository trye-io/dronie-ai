import pygame
from djitellopy import Tello
import numpy as np 
import cv2 
import mediapipe as mp
from helpers import draw_bbox

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


MODEL_PATH = 'level-2/blaze_face_short_range.tflite'

def render_frame(result, output_image, timestamp_ms):
    # позначаємо що змінна є глобальною
    global error

    frame = draw_bbox(output_image.numpy_view(), result)

    frame = np.rot90(frame)
    frame = np.flipud(frame) 
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0,0))

    if result.detections:
        for detection in result.detections:
            # знаходимо центр зображення по осі x 
            center_x = detection.bounding_box.origin_x + detection.bounding_box.width // 2
        if is_tracking:
            error = track_face(center_x, error)

def track_face(center_x, error):
    current_error = center_x - WIDTH // 2
    delta = current_error - error
    yaw_velocity = int(PID[0] * current_error + PID[2] * delta)
    print(yaw_velocity)
    # потім тут ми передамо команди на дрон за допомогою .send_rc_control()

    return current_error

options = FaceDetectorOptions(
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

is_tracking = False # статус відстежувача, True -- стежити
                    # False -- не стежити

PID = (0.2, 0, 0.2) # коефіцієнти ПІД контролера
error = 0 # помилка (різниця між центром зображення та центром обличчя)

timestamp = 0
is_running = True

with FaceDetector.create_from_options(options) as detector:
    while is_running: 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                drone.streamoff()
                is_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    is_tracking = True
                if event.type == pygame.K_0:
                    is_tracking = False

        frame = frame_read.frame 

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        detector.detect_async(
            mp_image,
            timestamp 
        )

        timestamp += 1

        pygame.display.flip() 
        clock.tick(FPS)