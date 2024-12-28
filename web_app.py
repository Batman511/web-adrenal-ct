import streamlit as st
import os
import torch
import numpy as np
import cv2

MODEL_NAME = 'full_model.pth'
model = torch.load(MODEL_NAME, map_location=torch.device('cpu'))

# Если модель обернута в DataParallel, извлекаем основную модель
if isinstance(model, torch.nn.DataParallel):
    model = model.module


device = torch.device('cpu')
model = model.to(device)

# Функция предобработки изображений
def preprocess_video(cap, frame_skip=3, target_size=(240, 240)):
    left_frames = []
    right_frames = []

    frame_count = 0 # для пропуска кадров
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            # Делим файл на 2
            left_frame = frame[:, :frame.shape[1] // 2]
            right_frame = frame[:, frame.shape[1] // 2:]

            left_frame = cv2.cvtColor(cv2.resize(left_frame, target_size), cv2.COLOR_BGR2GRAY)
            left_frames.append(left_frame)

            right_frame = cv2.cvtColor(cv2.resize(right_frame, target_size), cv2.COLOR_BGR2GRAY)
            right_frames.append(right_frame)
        frame_count += 1
    cap.release()
    left_video = np.array(left_frames, dtype=np.uint8)
    right_video = np.array(left_frames, dtype=np.uint8)
    return left_video, right_video


# Функция для вывода мультилейблов
def get_class_label(labels):
    classes = ['Злокачественный фенотип', 'Дорброкачественный фенотип', 'Неопределенный фенотип']
    selected_classes = []
    threshold = 0.5

    # Преобразуем тензор в NumPy массив
    labels = labels.detach().cpu().numpy().flatten()
    print("\n\n-------------------------------")
    print(labels)

    # Проходим по меткам и добавляем номер класса, если метка = 1
    for idx, label in enumerate(labels):
        if label > threshold:
            selected_classes.append(classes[idx])

    return ", ".join(selected_classes) if selected_classes else "Отсутствие образований!"



# вывод заголовка
st.title('Предсказание фенотипа надпочечников по КТ-снимку брюшной полости')
st.subheader('Введите данныe и загрузите КТ-снимок:')

# Ввод данных пользователем
age = st.slider('Возраст:', 1, 100, 30)
gender = st.radio('Ваш пол:', ('Муж.', 'Жен.'), index=0, key='gender')
contrast = st.selectbox('Фаза КТ:', ["Нативная", "Артериальная", "Венозная", "Отсроченная"], key='cholesteerol')


# Загрузка изображения
uploaded_file = st.file_uploader("Загрузите видео:", type=["mp4"])

if uploaded_file is not None:
    # Сохраняем файл на диск
    temp_file_path = os.path.join("temp_video.mp4")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    cap = cv2.VideoCapture(temp_file_path)
    left_video, right_video = preprocess_video(cap)

    left_video_tensor = torch.Tensor(left_video).unsqueeze(0).unsqueeze(1).to(device)  # Преобразуем данные в тензор и добавляем размерность для батча
    right_video_tensor = torch.Tensor(right_video).unsqueeze(0).unsqueeze(1).to(device)  # Преобразуем данные в тензор и добавляем размерность для батча

    model.eval()
    with torch.no_grad():  # Отключаем вычисление градиентов для ускорения
        left_predictions = model(left_video_tensor)
        right_predictions = model(right_video_tensor)

    # Получаем классы
    left_predictions = get_class_label(left_predictions)
    right_predictions = get_class_label(right_predictions)


    # Выводим результат
    lcol, rcol = st.columns(2)
    with lcol:
        st.text('Левый надпочечник имеет:')
        st.write(left_predictions)
    with rcol:
        st.text('Правый надпочечник имеет:')
        st.write(right_predictions)

    # Вывод рекомендаций
    if left_predictions != "Отсутствие образований!" or right_predictions != "Отсутствие образований!":
        st.subheader('Пожалуйста пройдите медицинское обследование!')
    else:
        st.subheader('Всё хорошо!')
