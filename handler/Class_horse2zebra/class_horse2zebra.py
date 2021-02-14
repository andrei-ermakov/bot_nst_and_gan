import io
import torch
from torch import nn
import torchvision.transforms as transforms
from .function_for_h2z import load_checkpoint, cuda
from .generator import define_Gen
from PIL import Image
from data import checkpoint_dir


class GAN_for_Transform(nn.Module):
    def __init__(self, picture, h2z=True):
        super(GAN_for_Transform, self).__init__()

        # для того, чтобы знать, какое преобразование выполняем (из зебры в лошадь или из лошади в зебру)
        self.h2z = h2z

        # выбедем архитектуру, на которой будем работать и переведем модель на нее
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("We use in GAN_for_Transform: ", self.device)

        # передадим входную картинку переменной класса
        self.picture = Image.open(picture).convert("RGB")

    def transformation(self):
        # задаем трансформер
        transform = transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # применяем трансформер к входной картинке и добавим еще одно измерение
        x = transform(self.picture)
        x = x.unsqueeze(0)

        # задаем генератор
        generator = define_Gen(input_nc=3, output_nc=3, ngf=64, norm='instance', use_dropout=True)

        # загружаем сохраненные веса обученных генераторов и дискриминаторов
        ckpt = load_checkpoint('%s/Weight_for_bot.ckpt' % checkpoint_dir)

        if self.h2z:
            generator.load_state_dict(ckpt['Gba'])
        else:
            generator.load_state_dict(ckpt['Gab'])

        # переводим вычисления на cuda и в режим предсказаний
        x = cuda(x)
        generator.eval()

        # превращаем животных в антипода
        with torch.no_grad():
            x = generator(x)

        # удалим одно измерение и проведем восстановление пикселей
        x = x.squeeze()
        x = (x.data + 1) / 2.0

        x = self.convert_to_bytes(x)

        return x

    # вход - tensor, выход - объект класса io.BytesIO()
    @staticmethod
    def convert_to_bytes(tensor):
        # экземпляр трансформера
        transform_to_pil = transforms.ToPILImage()
        # копируем тензор и отвязываем его от родительского (тот, что подавался на вход не будет изменяться, создаем копию)
        image = tensor.to("cpu").clone().detach()
        # преобразуем в PIL
        image = transform_to_pil(image)
        # переводим в байтовый поток
        bio = io.BytesIO()
        bio.name = 'image_result.jpeg'
        image.save(bio, 'JPEG')
        # изменить позицию потока на заданное смещение байта
        bio.seek(0)

        return bio