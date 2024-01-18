import pygame
from djitellopy import Tello
import numpy as np 
import cv2 
import mediapipe as mp
import threading # завантажуємо threding для асинхронного виконання

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


MODEL_PATH = 'level-2/blaze_face_short_range.tflite'

def render_frame(result, output_image, timestamp_ms):
    
    frame = output_image.numpy_view()

    if result.detections:
        for detection in result.detections:
            bbox = detection.bounding_box
            cv2.rectangle(
                frame,
                (bbox.origin_x, bbox.origin_y),
                (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                color=(255, 0, 0),
                thickness=2
            )
            face_center = bbox.origin_x + bbox.width // 2
        if is_tracking:
            track_face(face_center)

    frame = np.rot90(frame)
    frame = np.flipud(frame) 
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))

def track_face(face_center):
    error = FRAME_CENTER - face_center
    yaw_velocity = int(Kp * error)
    # замість друку, передаємо на дрон команди, але тільки якщо дрон в польоті
    if is_flying: 
        drone.send_rc_control(0, 0, 0, yaw_velocity)

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=render_frame
)

WIDTH = 960
HEIGHT = 720
FRAME_CENTER = WIDTH / 2

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

is_tracking = False

Kp = -0.125

timestamp = 0
is_running = True

with FaceDetector.create_from_options(options) as detector:
    while is_running: 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # якщо виходимо з програми, і ми ще в польоті, сідаємо 
                # і перемикаємо is_flying та is_tracking на False
                if is_flying: 
                    threading.Thread(target=drone.land).start()
                    is_tracking = False
                    is_flying = False
                drone.streamoff()
                is_running = False
            if event.type == pygame.KEYDOWN:
                # ми можемо переключити режим, тільки якщо летимо
                if event.key == pygame.K_1 and is_flying:
                    is_tracking = True
                if event.key == pygame.K_0:
                    is_tracking = False
                    # передаємо нульові швидкості, які можуть бути ненульові
                    # з попереднього використання коли is_tracking True
                    drone.send_rc_control(0, 0, 0, 0)
                # заради безпеки, забезпечуємо зліт та посадку за допомогою 
                # клавіш T (злетіти) та L (сісти)
                if event.key == pygame.K_t and not(is_flying):
                    is_flying = True
                    threading.Thread(target=drone.takeoff).start()
                if event.key == pygame.K_l and is_flying:
                    is_flying = False
                    is_tracking = False # Вимикаємо режим слідкування! 
                    threading.Thread(target=drone.land).start()

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