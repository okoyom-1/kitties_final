# импортируем torch
# torch (pytorch) - один из самых известных фреймворков python, ->
# -> используется для машинного обучения

import torch
from torch import nn # nn (neural networks) - модуль для работы с нейронными сетями в pytorch

# выбор вычислительного устройства (если поддерживается технология CUDA, то будет использована она)

device = "cuda" if torch.cuda.is_available() else "cpu"

# проверка наличия файлов и папок в целевой папке (где лежат картинки для обучения)

import os
def walk_through_dir(dir_path): # функция для прохода по папке
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"В папке '{dirpath}' {len(dirnames)} папок и {len(filenames)} файлов.")
    
image_path = "." # название целевой папки (должна находиться в одной папке с кодом)
walk_through_dir(image_path)

# определяем папки для тренировочных и тестовых фотографий
train_dir = "training_set"
test_dir = "test_set"
walk_through_dir(train_dir)

# достаем случайную картинку, выводим информацию о ней

import random
from PIL import Image
import glob # используется для перебора адресов в папках и файлах
from pathlib import Path # используется для управления путями к файлам и папкам

random.seed(random.randint(1,100)) # случайный паттерн

image_path_list= glob.glob(f"{image_path}/*/*/*.jpg") # полный список адресов картинок

random_image_path = random.choice(image_path_list) # получаем случайую картинку

image_class = Path(random_image_path).parent.stem # получаем название папки, в которой находится картинка (cats или dogs)

img = Image.open(random_image_path) # открываем картинку
img.show()

print(f"Адрес картинки: {random_image_path}") # выводим в консоль данные о картинке
print(f"Класс картинки (cats или dogs): {image_class}")
print(f"Высота картинки: {img.height}") 
print(f"Ширина картинки: {img.width}")
