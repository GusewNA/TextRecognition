import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import easyocr
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import uuid
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация Flask приложения
app = Flask(__name__)
app.secret_key = 'your_secure_secret_key_here'  # Замените на ваш надежный секретный ключ

# Конфигурация путей
UPLOAD_FOLDER = 'static/uploads/'
PREPROCESSED_FOLDER = 'static/preprocessed/'
RESULTS_FOLDER = 'static/results/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Допустимые расширения файлов
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Инициализация EasyOCR reader для русского и английского языков
ocr_languages = ['ru', 'en']
reader = easyocr.Reader(ocr_languages, gpu=False)

# Обеспечение существования необходимых папок
for folder in [UPLOAD_FOLDER, PREPROCESSED_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)


# Функция для проверки допустимого расширения файла
def allowed_file(filename):
    if '.' not in filename:
        return False
    name, ext = filename.rsplit('.', 1)
    return name != '' and ext.lower() in ALLOWED_EXTENSIONS


# Функция для предобработки изображения с коррекцией наклона
def preprocess_image(image_path, preprocessed_path):
    try:
        # Загрузка изображения с помощью OpenCV в оттенках серого
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Не удалось загрузить изображение.")

        # Бинаризация изображения
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Определение угла наклона
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        logger.info(f"Определённый угол наклона: {angle} градусов")

        # Коррекция наклона изображения
        (h, w) = thresh.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Конвертация обратно в PIL Image для дальнейшей обработки
        pil_img = Image.fromarray(rotated)

        # Дополнительная предобработка с помощью PIL
        pil_img = pil_img.filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.5)

        # Опционально: увеличение размера изображения для улучшения OCR
        pil_img = pil_img.resize((pil_img.width * 2, pil_img.height * 2), Image.Resampling.LANCZOS)

        # Сохранение предобработанного изображения
        pil_img.save(preprocessed_path)

        logger.info(f"Предобработанное изображение сохранено по пути: {preprocessed_path}")
    except Exception as e:
        logger.error(f"Ошибка при предобработке изображения: {e}")
        raise e


# Роут для главной страницы
@app.route('/')
def index():
    return render_template('index.html')


# Роут для обработки загрузки файла и распознавания текста
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            logger.warning("Нет файла для загрузки")
            return jsonify({'error': 'Нет файла для загрузки'}), 400
        file = request.files['file']

        if file.filename == '':
            logger.warning("Нет выбранного файла")
            return jsonify({'error': 'Нет выбранного файла'}), 400

        if file and allowed_file(file.filename):
            # Извлечение расширения
            _, ext = os.path.splitext(file.filename)
            ext = ext.lower().strip('.')
            if ext not in ALLOWED_EXTENSIONS:
                logger.warning(f"Недопустимое расширение файла: {ext}")
                return jsonify({'error': 'Недопустимый тип файла'}), 400
            # Генерация уникального имени файла
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            try:
                file.save(file_path)
                logger.info(f"Файл сохранён по пути: {file_path}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении файла: {e}")
                return jsonify({'error': 'Ошибка при сохранении файла'}), 500

            # Предобработка изображения
            preprocessed_path = os.path.join(PREPROCESSED_FOLDER, f'preprocessed_{unique_filename}')
            try:
                preprocess_image(file_path, preprocessed_path)
            except Exception as e:
                logger.error(f"Ошибка при предобработке изображения: {e}")
                return jsonify({'error': f'Ошибка при предобработке изображения: {str(e)}'}), 500

            # Распознавание текста с помощью EasyOCR
            try:
                result = reader.readtext(preprocessed_path, detail=0, paragraph=True)
                recognized_text = '\n'.join(result)
                logger.info(f"Распознанный текст: {recognized_text}")
            except Exception as e:
                logger.error(f"Ошибка при распознавании текста: {e}")
                return jsonify({'error': f'Ошибка при распознавании текста: {str(e)}'}), 500

            if not recognized_text.strip():
                logger.warning("Не удалось распознать текст.")
                return jsonify({'error': 'Не удалось распознать текст. Попробуйте другое изображение.'}), 400

            # Сохранение распознанного текста в файл
            result_text_path = os.path.join(RESULTS_FOLDER, f'{unique_filename}.txt')
            try:
                with open(result_text_path, 'w', encoding='utf-8') as f:
                    f.write(recognized_text)
                logger.info(f"Распознанный текст сохранён по пути: {result_text_path}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении результатов: {e}")
                return jsonify({'error': f'Ошибка при сохранении результатов: {str(e)}'}), 500

            # Возвращение успешного ответа с URL загруженного файла и распознанного текста
            return jsonify({
                'file_url': url_for('static', filename=f'uploads/{unique_filename}'),
                'text': recognized_text,
                'download_url': url_for('static', filename=f'results/{unique_filename}.txt')
            }), 200

        logger.warning("Недопустимый тип файла")
        return jsonify({'error': 'Недопустимый тип файла'}), 400
    except Exception as e:
        logger.error(f"Неизвестная ошибка: {e}")
        return jsonify({'error': 'Неизвестная ошибка произошла.'}), 500


# Запуск приложения
if __name__ == '__main__':
    app.run(debug=True)
