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

# подготовка к машинному обучению 

import torch 
from torch.utils.data import DataLoader # загрузчик инфоромации в модель машинного обучения
from torchvision import datasets, transforms # стандарты для данных и трансформаций данных в torchvision

IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

# алгоритм трансформации для картинки
data_transform = transforms.Compose([
    # картинка адаптируется под нужный размер 
    transforms.Resize(size=IMAGE_SIZE),
    # картинка поворачивается 
    transforms.RandomHorizontalFlip(p=0.5), 
    # картинка превращается в таблицу чисел
    transforms.ToTensor() 
])


# создается тренировочное хранилище картинок для проекта
train_data = datasets.ImageFolder(root=train_dir, # адрес хранилища
                                  transform=data_transform, # правила трансформации для картинок
                                  target_transform=None) # угабуга

# создается тестировочное хранилище картинок для проекта
test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

print(f"Тренировочная папка:\n{train_data}\nТестировочная папка:\n{test_data}")

# список всех классов (кошек и собак)
class_names = train_data.classes
print("Список классов: ",class_names)

# словарь всех классов 
class_dict = train_data.class_to_idx
print("Словарь классов: ",class_dict)

# проверка длины
print("Длина тренировочного и тестировочного пакетов: ", len(train_data), len(test_data))


# выделение процессов процессора для машинного обучения
NUM_WORKERS = os.cpu_count()

# создание загрузчиков данных
train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=1, # размер пачки данных
                              num_workers=NUM_WORKERS,
                              shuffle=True) # перемешивание данных

test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             num_workers=NUM_WORKERS, 
                             shuffle=False) 

print(train_dataloader, test_dataloader)


# другой размер картинки
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

# дополнительные возможности для трансформации данных при помощи аугментаций
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()])

test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()])


train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform)
test_data_augmented = datasets.ImageFolder(test_dir, transform=test_transform)

print(train_data_augmented, test_data_augmented) 

# большё бессмысленных параметров!!!
BATCH_SIZE = 32 # размер порции карттттттттттттттттТ_Тинок
torch.manual_seed(42)

train_dataloader_augmented = DataLoader(train_data_augmented, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True,
                                        num_workers=NUM_WORKERS)

test_dataloader_augmented = DataLoader(test_data_augmented, 
                                       batch_size=BATCH_SIZE, 
                                       shuffle=False, 
                                       num_workers=NUM_WORKERS)

print(train_dataloader_augmented, test_dataloader_augmented)








