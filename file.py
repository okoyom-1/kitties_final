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

# создание классификатора pytorch
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
          nn.Conv2d(3, 64, 3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.MaxPool2d(2))
        self.conv_layer_2 = nn.Sequential(
          nn.Conv2d(64, 512, 3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(2))
        self.conv_layer_3 = nn.Sequential(
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(2)) 
        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=512*3*3, out_features=2))
    def forward(self, x: torch.Tensor):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x
# Создаем новую модель машинного обучения.
model = ImageClassifier().to(device)

# 1. получить пачку картинок из объекта DataLoader
img_batch, label_batch = next(iter(train_dataloader_augmented))

# 2. взять из этой пачки картинку и адаптирвоать ее под модель
img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
print(f"Адаптированная картинка: {img_single.shape}\n")

# 3. применить модель к картинке
model.eval()
with torch.inference_mode():
    pred = model(img_single.to(device))

# 4. вывести результаты
print(f"Логиты:\n{pred}\n")
print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
print(f"Actual label:\n{label_single}")


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    # Включение режима обучения модели
    model.train()
    
    # Задать начальное состояние метрики
    train_loss, train_acc = 0, 0
    
    # Пройти по каждой пачке данных
    for batch, (X, y) in enumerate(dataloader):
        # Отправить картинки на CPU или GPU
        X, y = X.to(device), y.to(device)
        
        # 1. Шаг вперед
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc



