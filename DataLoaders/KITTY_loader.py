import numpy as np
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


from utils.conts import V2X_RADARI_PATH


class KITTILoader:
    """Простой загрузчик для датасета V2X-Radar в формате KITTI."""

    def __init__(self, base_path, split='training'):
        """
        Args:
            base_path (str): Путь к папке V2X-Radar-I или V2X-Radar-V
            split (str): 'training' или 'testing'
        """
        self.base_path = Path(base_path)
        self.split = split

        # Формируем пути к подпапкам
        self.velodyne_dir = self.base_path / split / 'velodyne'
        self.radar_dir = self.base_path / split / 'radar'
        self.image_dir = self.base_path / split / 'image_2'  # для V2X-Radar-V
        # Для V2X-Radar-I также есть image_1, image_3
        self.calib_dir = self.base_path / split / 'calib'
        self.label_dir = self.base_path / split / 'label_2'

        # Получаем список всех доступных ID (имен файлов без расширения)
        self.ids = []
        if self.velodyne_dir.exists():
            self.ids = sorted([f.stem for f in self.velodyne_dir.glob('*.bin')])

    def __len__(self):
        return len(self.ids)

    def read_lidar(self, idx):
        """Чтение LiDAR облака точек из .bin файла (формат KITTI)."""
        lidar_path = self.velodyne_dir / f'{idx:06d}.bin'
        if not lidar_path.exists():
            return None
        # В KITTI формате точки хранятся как float32: x, y, z, reflectance
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        return points

    def read_radar(self, idx):
        """Чтение 4D Radar облака точек из .bin файла."""
        radar_path = self.radar_dir / f'{idx:06d}.bin'
        if not radar_path.exists():
            return None
        # Для радара ожидаем 5 каналов: x, y, z, doppler, intensity
        points = np.fromfile(radar_path, dtype=np.float32).reshape(-1, 5)
        return points

    def read_image(self, idx, camera_id=2):
        """Чтение изображения с камеры."""
        """if camera_id == 2:
            img_path = self.image_dir / f'{idx:06d}.png'  # или .jpg
        else:"""
            # Для V2X-Radar-I есть несколько камер
        img_path = self.base_path / self.split / f'image_{camera_id}' / f'{idx:06d}.jpg'

        if not img_path.exists():
            return None
        return cv2.imread(str(img_path))

    def read_calib(self, idx):
        """Чтение файла калибровки."""
        calib_path = self.calib_dir / f'{idx:06d}.txt'
        if not calib_path.exists():
            return None

        calib_data = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    # Парсим матрицы 3x4 или 3x3
                    values = [float(v) for v in value.strip().split()]
                    calib_data[key] = np.array(values).reshape(3, -1)
        return calib_data

    def read_labels(self, idx):
        """Чтение аннотаций 3D bounding boxes."""
        label_path = self.label_dir / f'{idx:06d}.txt'
        if not label_path.exists():
            return []

        objects = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 15:
                    continue

                obj = {
                    'type': parts[0],
                    'truncated': float(parts[1]),
                    'occluded': int(parts[2]),
                    'alpha': float(parts[3]),
                    'bbox': [float(p) for p in parts[4:8]],
                    'dimensions': [float(p) for p in parts[8:11]],  # height, width, length
                    'location': [float(p) for p in parts[11:14]],  # x, y, z в системе камеры
                    'rotation_y': float(parts[14])
                }
                objects.append(obj)
        return objects

    def visualize_sample(self, idx):
        """Простая визуализация для проверки загрузки."""

        # Загружаем данные
        lidar = self.read_lidar(idx)
        radar = self.read_radar(idx)
        image = self.read_image(idx)
        objects = self.read_labels(idx)

        print(f"Sample {idx}:")
        print(f"  LiDAR points: {lidar.shape if lidar is not None else 'None'}")
        print(f"  Radar points: {radar.shape if radar is not None else 'None'}")
        print(f"  Image: {image.shape if image is not None else 'None'}")
        print(f"  Objects detected: {len(objects)}")

        # Если есть изображение, рисуем 2D bounding boxes
        if image is not None:
            plt.figure(figsize=(12, 8))
            for obj in objects:
                # Рисуем 2D bbox из аннотации
                bbox = obj['bbox']
                cv2.rectangle(image,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 255, 0), 2)
                cv2.putText(image, obj['type'],
                            (int(bbox[0]), int(bbox[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f'Frame {idx} with 2D boxes')
            plt.axis('off')
            plt.show()


# Пример использования
if __name__ == "__main__":
    # Укажите путь к скачанному датасету
    loader = KITTILoader(
        base_path=V2X_RADARI_PATH,
        split='training'
    )

    print(f"Найдено {len(loader)} файлов")

    # Загружаем и визуализируем первый пример
    if len(loader) > 0:
        loader.visualize_sample(0)

        # Пример чтения отдельных компонентов
        points = loader.read_lidar(0)
        calib = loader.read_calib(0)
        labels = loader.read_labels(0)

        print("\nПервый объект в сцене:")
        if labels:
            print(labels[0])