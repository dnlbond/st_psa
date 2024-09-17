import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title('Сингулярное разложение изображений')
st.write('Загрузите картинку, выберите количество сингулярных чисел, смотрите магию.')

# Поле для загрузки файла изображения
uploaded_file = st.file_uploader("Выберите файл изображения", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Загрузка изображения с помощью PIL
        image = Image.open(uploaded_file)
        # Конвертация в RGB (если изображение в формате RGBA или другим)
        image = image.convert('RGB')
        image_np = np.array(image)

        # Отображение исходного изображения
        st.image(image_np, caption='Исходное изображение', use_column_width=True)

        # Поле для выбора количества сингулярных чисел
        num_singular_values = st.slider('Выберите количество сингулярных чисел', 1, min(image_np.shape[0], image_np.shape[1]), 50)

        # Функция для выполнения SVD и восстановления изображения
        def svd_reconstruct(image, num_singular_values):
            u, s, vt = np.linalg.svd(image, full_matrices=False)
            s[num_singular_values:] = 0  # Обнуление всех сингулярных чисел кроме первых num_singular_values
            return np.dot(u, np.dot(np.diag(s), vt))

        # Применение SVD к каждому каналу RGB
        reconstructed_image = np.zeros_like(image_np)
        for i in range(3):  # Для каждого канала R, G, B
            reconstructed_image[:, :, i] = svd_reconstruct(image_np[:, :, i], num_singular_values)

        # Преобразование в формат изображения для отображения
        reconstructed_image_pil = Image.fromarray(np.uint8(reconstructed_image))

        # Отображение восстановленного изображения
        st.image(reconstructed_image_pil, caption=f'Изображение с {num_singular_values} сингулярными числами', use_column_width=True)

    except Exception as e:
        st.error(f"Не удалось обработать изображение. Ошибка: {e}")