import numpy as np
import tensorflow as tf

# Підготовка даних
# Визначаємо кількість зразків, розмір партії та кількість ітерацій для навчання
n_samples, batch_size, num_steps = 1000, 100, 20000

# Генеруємо дані: X_data - випадкові числа в діапазоні [1, 10],
# y_data - залежні змінні з додаванням шуму
X_data = np.random.uniform(1, 10, (n_samples, 1))
y_data = 2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))

# Перетворюємо дані у тензори для використання в TensorFlow
X_data = tf.convert_to_tensor(X_data, dtype=tf.float32)
y_data = tf.convert_to_tensor(y_data, dtype=tf.float32)

# Оголошуємо змінні для моделі.
# k - коефіцієнт нахилу (випадкова ініціалізація), b - зсув (ініціалізується нулем)
k = tf.Variable(tf.random.normal((1, 1), mean=0.0, stddev=0.1), name='slope')
b = tf.Variable(tf.zeros((1,)), name='bias')

# Лінійна регресійна модель
# Функція для розрахунку прогнозованих значень: y = X * k + b
def model(X):
    return tf.matmul(X, k) + b

# Функція втрат (Mean Squared Error - середньоквадратична похибка)
# Розраховуємо різницю між прогнозованими значеннями (y_pred) та істинними (y_true)
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Оптимізатор із малою швидкістю навчання (learning_rate=0.001)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

# Цикл навчання
display_step = 100
for i in range(num_steps):
    # Вибираємо випадкову партію даних для поточної ітерації
    indices = np.random.choice(n_samples, batch_size)
    X_batch, y_batch = tf.gather(X_data, indices), tf.gather(y_data, indices)

    # Відстежуємо обчислення градієнтів
    with tf.GradientTape() as tape:
        # Розраховуємо прогнозовані значення для поточної партії
        y_pred = model(X_batch)
        # Обчислюємо втрати (похибку)
        loss_val = loss_fn(y_batch, y_pred)

    # Обчислюємо градієнти для параметрів k та b
    grads = tape.gradient(loss_val, [k, b])
    # Оновлюємо параметри моделі за допомогою оптимізатора
    optimizer.apply_gradients(zip(grads, [k, b]))

    # Вивід прогресу кожні 100 ітерацій
    if (i + 1) % display_step == 0:
        print('Епоха %d: %.8f, k=%.4f, b=%.4f' % (i + 1, loss_val.numpy(), k.numpy()[0][0], b.numpy()[0]))
