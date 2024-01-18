import pygame
from djitellopy import Tello
import numpy as np 
import cv2 
import mediapipe as mp
from helpers import draw_bbox
import threading # завантажуємо threding для асинхронного виконання

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


MODEL_PATH = 'level-2/blaze_face_short_range.tflite'

def render_frame(result, output_image, timestamp_ms):
    global error

    frame = draw_bbox(output_image.numpy_view(), result)

    frame = np.rot90(frame)
    frame = np.flipud(frame) 
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0,0))

    if result.detections:
        for detection in result.detections:
            face_center = detection.bounding_box.origin_x + detection.bounding_box.width // 2
        if is_tracking:
            error = track_face(face_center, error)

def track_face(face_center, error):
    current_error = FRAME_CENTER - face_center
    delta = current_error - error
    yaw_velocity = int(Kp * current_error + Kd * delta)
    # замість друку, передаємо на дрон команди, але тільки якщо дрон в польоті
    if is_flying: 
        drone.send_rc_control(0, 0, 0, yaw_velocity)

    return current_error

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
Kd = 0
error = 0

timestamp = 0
is_running = True

with FaceDetector.create_from_options(options) as detector:
    while is_running: 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # якщо виходимо з програми, і ми ще в польоті, сідаємо 
                if is_flying: 
                    threading.Thread(target=drone.land).start()
                    is_flying = False
                drone.streamoff()
                is_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    is_tracking = True
                if event.key == pygame.K_0:
                    is_tracking = False
                    # передаємо нульові швидкості, які можуть бути ненульові
                    # з попереднього використання Режиму 1
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

        print(drone.get_battery())

        pygame.display.flip() 
        clock.tick(FPS)