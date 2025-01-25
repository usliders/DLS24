# nst.py
import tensorflow as tf
from tensorflow.keras import Model, applications
import numpy as np
from PIL import Image
import logging
import gc
import multiprocessing

logger = logging.getLogger(__name__)

def _wct_transform(content, style):
    """Реализация Whitening and Coloring Transform"""
    if content.shape[-1] != style.shape[-1]:
        raise ValueError("Несовпадение каналов контента и стиля")

    c_shape = tf.shape(content)
    content_flat = tf.reshape(content, [-1, c_shape[-1]])
    style_flat = tf.reshape(style, [-1, c_shape[-1]])

    mean_c = tf.reduce_mean(content_flat, axis=0)
    mean_s = tf.reduce_mean(style_flat, axis=0)

    content_centered = content_flat - mean_c
    style_centered = style_flat - mean_s

    cov_c = tf.matmul(content_centered, content_centered, transpose_a=True)
    cov_s = tf.matmul(style_centered, style_centered, transpose_a=True)

    s_c, u_c, _ = tf.linalg.svd(cov_c)
    s_c = tf.linalg.diag(1.0/tf.sqrt(s_c + 1e-5))
    whitening = tf.matmul(u_c, tf.matmul(s_c, u_c, transpose_b=True))

    s_s, u_s, v_s = tf.linalg.svd(cov_s)
    s_s = tf.linalg.diag(tf.sqrt(s_s + 1e-5))
    coloring = tf.matmul(u_s, tf.matmul(s_s, v_s, transpose_b=True))

    transformed = tf.matmul(content_centered, whitening)
    transformed = tf.matmul(transformed, coloring) + mean_s

    return tf.reshape(transformed, c_shape)

def style_transfer_worker(args):
    """Изолированный процесс для обработки стиля"""
    content_path, style_path, img_size, iterations, alpha = args
    try:
        # Инициализация внутри процесса
        vgg = applications.VGG19(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
        feature_extractor = Model(vgg.input, vgg.get_layer('block4_conv1').output)
        feature_extractor.trainable = False
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)

        # Загрузка изображений
        def load_image(path):
            img = Image.open(path).convert('RGB')
            w, h = img.size
            size = min(w, h)
            img = img.crop(((w-size)//2, (h-size)//2, (w+size)//2, (h+size)//2))
            img = img.resize((img_size, img_size))
            return np.array(img, dtype=np.float32)/127.5 - 1.0

        content = tf.expand_dims(load_image(content_path), 0)
        style = tf.expand_dims(load_image(style_path), 0)

        # Извлечение признаков
        content_feat = feature_extractor(content)
        style_feat = feature_extractor(style)

        # Оптимизация
        generated = tf.Variable(content)
        for i in range(iterations):
            with tf.GradientTape() as tape:
                gen_feat = feature_extractor(generated)
                transformed = _wct_transform(gen_feat, style_feat)
                content_loss = tf.reduce_mean(tf.abs(gen_feat - content_feat))
                style_loss = tf.reduce_mean(tf.abs(transformed - gen_feat))
                total_loss = (1-alpha)*content_loss + alpha*style_loss*10
            gradients = tape.gradient(total_loss, generated)
            optimizer.apply_gradients([(gradients, generated)])

            if i % 50 == 0:
                logger.info(f"Iter {i}: Loss={total_loss.numpy():.2f}")
        # Постобработка
        result = (generated.numpy().squeeze() + 1.0) * 127.5
        logger.info("Обучение успешно завершено")
        return result.astype(np.uint8)

    except Exception as e:
        logger.error(f"Ошибка обработки: {str(e)}")
        raise
    finally:
        # Полная очистка ресурсов
        del vgg, feature_extractor, optimizer
        tf.keras.backend.clear_session()
        gc.collect()

class NeuralStyleTransfer:
    def __init__(self, img_size=512):
        self.img_size = img_size
        self.pool = multiprocessing.Pool(processes=1)
        logger.info("NST Processor initialized")

    def transfer_style(self, content_path, style_path, iterations=500, alpha=0.7):
        args = (content_path, style_path, self.img_size, iterations, alpha)
        result = self.pool.apply(style_transfer_worker, (args,))
        return Image.fromarray(result)

    def __del__(self):
        logger.info("Closing NST Processor pool...")
        self.pool.close()
        self.pool.join()
        logger.info("NST Processor pool closed")