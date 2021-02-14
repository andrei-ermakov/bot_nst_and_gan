"""
Источник - статья по переносу стиля: https://nextjournal.com/gkoehler/pytorch-neural-style-transfer

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from .vgg_model import get_vgg
from PIL import Image
import io


class NeuralTransferStyle(nn.Module):
    def __init__(self, content_picture, style_picture, max_size, number_epochs=1000):
        super(NeuralTransferStyle, self).__init__()
        # количество эпох обучения
        self.epochs = number_epochs

        # выбедем архитектуру, на которой будем работать и переведем модель на нее
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("We use: ", self.device)

        # передадим входные картинки переменным класса
        self.content_before = Image.open(content_picture).convert("RGB")
        self.style_before = Image.open(style_picture).convert("RGB")

        # определим модель
        self.vgg = get_vgg().to(self.device).eval()

        # коэффициент для того, чтобы выходная картинки сохранила пропорцию
        self.koef_for_size_img = self.content_before.size[1] / self.content_before.size[0]

        # задаем количество пикселей для выходной картинки
        if max(self.content_before.size) > max_size:
            self.size = max_size
        else:
            self.size = max(self.content_before.size)

        # применяем трансформер для входных данных и переводим на cuda
        self.content = self.transform_image(self.content_before).to(self.device)
        self.style = self.transform_image(self.style_before).to(self.device)

        # получаем карты активаций для обеих картинок
        self.content_features = self.get_features(self.content, self.vgg)
        self.style_features = self.get_features(self.style, self.vgg)

        # считаем матрицы Грама для сверток картинки-стиля
        self.style_grams = {
            layer_r: self.gram_matrix(self.style_features[layer_r]) for layer_r in self.style_features}

        # это наша основа, пиксели которой мы будем подбирать
        self.target = torch.randn_like(self.content).requires_grad_(True).to(self.device)

        # определяем весовые коэффициенты каждого слоя
        self.style_weights = {'conv1_1': 0.75,
                              'conv2_1': 0.5,
                              'conv3_1': 0.2,
                              'conv4_1': 0.2,
                              'conv5_1': 0.2}

        # определяем веса для картинки-основы и картинки-стиля
        self.content_weight = 1e4
        self.style_weight = 1e2

    # трансформер
    def transform_image(self, img):
        in_transform = transforms.Compose([
            transforms.Resize((int(self.koef_for_size_img * self.size), self.size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # применим трансформер к картинке, добавив еще одно измерение
        img = in_transform(img)[:3, :, :].unsqueeze(0)

        return img

    def train_nst(self):
        optimizer = optim.Adam([self.target], lr=0.01)

        for i_i in range(1, self.epochs+1):

            total_loss = 0

            # обнулим градиенты
            optimizer.zero_grad()

            # получим карты активаций с слоев модели для нашей рандомной картинки, которую будем преобразовывать
            target_features = self.get_features(self.target, self.vgg)

            # определим лосс, как среднее значение квадратов разности карт
            # активаций на conv4_2 слое между шумовой картинкой (целевой картинкой) и картинкой-основой
            content_loss = torch.mean((target_features['conv4_2'] - self.content_features['conv4_2']) ** 2)

            style_loss = 0

            # итерируемся по весам, заданным для каждого слоя в словаре style_weights
            # layer == conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
            for layer_i in self.style_weights:
                # получаем карту активации целевой картинки для слоя layer
                target_feature = target_features[layer_i]

                # вычисляем Грам матрицу для текущей карты активации
                target_gram = self.gram_matrix(target_feature)

                # получаем размеры карты активации для текущего слоя: d - количество каналов (RGB)
                _, d, h, w = target_feature.shape

                # получаем матрицу Грама карты активации для слоя layer для картины-стиля
                style_gram = self.style_grams[layer_i]

                # вычисляем лосс: вес для данного слоя умножить на среднее
                # квадратов разности матриц Грама целевой картинки и картины-стиля
                layer_style_loss = self.style_weights[layer_i] * torch.mean((target_gram - style_gram) ** 2)

                # суммируем в пределах всех слоев усредненный по пикселям лосс (стиль лосс)
                style_loss += layer_style_loss / (d * h * w)

                # суммарный лосс: сумма произведений основы лосс на его вес и стиль лосса на его вес
                total_loss = self.content_weight * content_loss + self.style_weight * style_loss

                # считаем антиградиенты на уменьшение суммарного лосса
                total_loss.backward(retain_graph=True)

            # и делаем шаг спуска в сторону антиградиентов
            optimizer.step()

            if i_i % 50 == 0:
                # округленный до 2-х знаков лосс
                total_loss_rounded = round(total_loss.item(), 2)

                # доля вносимая в лосс картинкой основой
                content_fraction = round(self.content_weight * content_loss.item() / total_loss.item(), 2)

                # доля вносимая в лосс картинкой-стилем
                style_fraction = round(self.style_weight * style_loss.item() / total_loss.item(), 2)

                # выводим все эти показатели
                print('Iteration {}, Total loss: {} - (content: {}, style {})'.format(
                    i_i, total_loss_rounded, content_fraction, style_fraction))

        self.target = self.convert_to_bytes(self.target)

        return self.target

    # вход - tensor, выход - объект класса io.BytesIO()
    @staticmethod
    def convert_to_bytes(tensor):
        # экземпляр трансформера
        transform_to_pil = transforms.ToPILImage()
        # копируем тензор и отвязываем его от родительского (тот, что подавался на вход не будет изменяться, создаем копию)
        image = tensor.to("cpu").clone().detach()
        # удаляем лишнее измерение
        image = torch.squeeze(image)
        # переместим rgb вправо
        image = image.permute(1, 2, 0)
        # умножить на ско и прибавить матожидание
        image = image * torch.tensor((0.229, 0.224, 0.225)) + torch.tensor((0.485, 0.456, 0.406))
        # переместим rgb влево
        image = image.permute(2, 0, 1)
        # заключить в диапазон от 0 до 1 (все, что больше 1 и меньше 0 становится им равным)
        image = torch.clamp(image, min=0, max=1)
        # преобразуем в PIL
        image = transform_to_pil(image)
        # переводим в байтовый поток
        bio = io.BytesIO()
        bio.name = 'image_result.jpeg'
        image.save(bio, 'JPEG')
        # изменить позицию потока на заданное смещение байта
        bio.seek(0)

        return bio

    @staticmethod
    def get_features(image, model, layers=None):
        if layers is None:
            # зададим, с каких слояев мы хотим получать карты активаций
            layers = {'0': 'conv1_1',
                      '5': 'conv2_1',
                      '10': 'conv3_1',
                      '19': 'conv4_1',
                      '21': 'conv4_2',
                      '28': 'conv5_1'}
        features = {}
        x = image

        # итерируемся по сверточным слоям
        for name, layer_i in enumerate(model.features):
            x = layer_i(x)
            # если имя слоя совпадает с ключом в layers, то добавляем в features карту активации соответствующего слоя
            if str(name) in layers:
                features[layers[str(name)]] = x

        return features

    # функция вычисления матрицы Грама
    @staticmethod
    def gram_matrix(tensor):
        _, n_filters, h, w = tensor.size()
        tensor = tensor.view(n_filters, h * w)
        gram = torch.mm(tensor, tensor.t())  # матричное умножение матриц

        return gram