import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris # pip install scikit-learn
from sklearn.decomposition import TruncatedSVD, PCA

from skimage import io
from PIL import Image

# Описание
st.title('Сингулярное разложение изображений')
st.write('Загрузи картинку, выбери количество сингулярных чисел, смотри магию.')

## Шаг 1. Загрузка картинки
url = st.text_input('Вставьте ссылку на изображение:')

image = io.imread(url)

st.image(image, caption='Ваша исходная картинка:', use_column_width=True)

# Поле для выбора количества сингулярных чисел
num_singular_values = st.slider('Выберите количество сингулярных чисел:', 1, min(image.shape[0], image.shape[1]), 100)


def svd_reconstruct(image, num_singular_values):
    u, s, vt = np.linalg.svd(image, full_matrices=False)
    s[num_singular_values:] = 0  # Обнуление всех сингулярных чисел кроме первых num_singular_values
    return np.dot(u, np.dot(np.diag(s), vt))

# Применение SVD к каждому каналу RGB
reconstructed_image = np.zeros_like(image)
for i in range(3):  # Для каждого канала R, G, B
    reconstructed_image[:, :, i] = svd_reconstruct(image[:, :, i], num_singular_values)

# Преобразование в формат изображения для отображения
reconstructed_image_pil = Image.fromarray(np.uint8(reconstructed_image))

# Отображение восстановленного изображения
st.image(reconstructed_image_pil, caption=f'Изображение с {num_singular_values} сингулярными числами', use_column_width=True)
