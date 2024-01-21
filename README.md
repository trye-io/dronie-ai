# Скрипти з використання AI (штучного інтелекту) з дроном DJI Tello

Цей репозиторій містить скрипти з навчальних відео про використання AI для дрону DJI Tello. Ці скрипти використовують мову програмування Python, неофіційний пакет DJITelloPy, а також бібліотеку машинного навчання mediapipe для комп'ютерного зору.

## Рівень 1

**Відео у YouTube:** [https://youtu.be/qV54PhrxGVw?si=v-iaNbFMAvFyuEZb](https://youtu.be/qV54PhrxGVw?si=v-iaNbFMAvFyuEZb)

- 👀 Отримуємо відео стрімінг з фронтальної камери дрону: [0-stream.py](https://github.com/trye-io/dronie-ai/blob/main/level-1/0-stream.py)
- 🖐️ Розпізнаємо жести за допомогою Mediapipe: [1-recognize.py](https://github.com/trye-io/dronie-ai/blob/main/level-1/1-recognize.py)
- 🛫 Злітаємо та приземлюємось за допомогою жестів 👍 та 👎: [2-fly.py](https://github.com/trye-io/dronie-ai/blob/main/level-1/2-fly.py)
- 🛬 Приземляємо дрон за допомогою нижньої камери та жесту 🖐️ (як HOVERAir X1): [3-downward-camera.py](https://github.com/trye-io/dronie-ai/blob/main/level-1/3-downward-camera.py)

## Рівень 2

**Відео у YouTube:** [https://youtu.be/wOaDrppG5Ko?si=ZKRjcEX1dGkK2Agd](https://youtu.be/wOaDrppG5Ko?si=ZKRjcEX1dGkK2Agd)

- 😜 Виявляємо обличчя за допомогою Mediapipe: [1-detect.py](https://github.com/trye-io/dronie-ai/blob/main/level-2/1-detect.py)
- 🎮 Додаємо ПІД регулятор (тільки пропорційний компонент): [2-tracker.py](https://github.com/trye-io/dronie-ai/blob/main/level-2/2-tracker.py)
- 🛫 Злітаємо: [3-fly.py](https://github.com/trye-io/dronie-ai/blob/main/level-2/3-fly.py)
- 🎮 Бонус: Додаємо диференціальну ПІД регулятора: [3.5-fly-pd-bonus.py](https://github.com/trye-io/dronie-ai/blob/main/level-2/3.5-fly-pd-bonus.py)