Отлично! Вот подробная таблица методов и функций для работы с **тензорами в TensorFlow** и **PyTorch** — двух самых популярных библиотек для глубокого обучения.  

---

## **🔹 Основные методы и функции для тензоров**  

### **1. Создание тензоров**  

#### **TensorFlow (`tf.Tensor`)**  
| Метод/Функция | Описание | Пример |
|--------------|----------|--------|
| `tf.constant(value)` | Создание тензора из данных | `tf.constant([[1, 2], [3, 4]])` |
| `tf.zeros(shape)` | Тензор из нулей | `tf.zeros((2, 3))` → `[[0., 0., 0.], [0., 0., 0.]]` |
| `tf.ones(shape)` | Тензор из единиц | `tf.ones((2, 2))` → `[[1., 1.], [1., 1.]]` |
| `tf.random.normal(shape)` | Тензор из нормального распределения | `tf.random.normal((2, 2))` |
| `tf.range(start, stop, step)` | Аналог `range()` для тензоров | `tf.range(0, 5, 1)` → `[0, 1, 2, 3, 4]` |

#### **PyTorch (`torch.Tensor`)**  
| Метод/Функция | Описание | Пример |
|--------------|----------|--------|
| `torch.tensor(data)` | Создание тензора из данных | `torch.tensor([[1, 2], [3, 4]])` |
| `torch.zeros(shape)` | Тензор из нулей | `torch.zeros((2, 3))` → `[[0., 0., 0.], [0., 0., 0.]]` |
| `torch.ones(shape)` | Тензор из единиц | `torch.ones((2, 2))` → `[[1., 1.], [1., 1.]]` |
| `torch.rand(shape)` | Тензор из равномерного распределения | `torch.rand((2, 2))` |
| `torch.arange(start, stop, step)` | Аналог `range()` | `torch.arange(0, 5, 1)` → `[0, 1, 2, 3, 4]` |

---

### **2. Атрибуты тензоров**  

#### **TensorFlow & PyTorch**  
| Атрибут | Описание | Пример |
|---------|----------|--------|
| `tensor.shape` | Размерность тензора | `tensor.shape` → `torch.Size([2, 3])` |
| `tensor.dtype` | Тип данных (`float32`, `int64` и др.) | `tensor.dtype` → `torch.float32` |
| `tensor.device` | Устройство (`cpu` / `gpu`) | `tensor.device` → `device('cuda:0')` |

---

### **3. Основные операции**  

#### **Математические операции**  
| Операция | TensorFlow | PyTorch |
|----------|-----------|---------|
| Сложение | `tf.add(a, b)` или `a + b` | `torch.add(a, b)` или `a + b` |
| Умножение | `tf.multiply(a, b)` или `a * b` | `torch.mul(a, b)` или `a * b` |
| Матричное умножение | `tf.matmul(a, b)` или `a @ b` | `torch.matmul(a, b)` или `a @ b` |
| Сумма по оси | `tf.reduce_sum(tensor, axis)` | `torch.sum(tensor, dim)` |
| Среднее по оси | `tf.reduce_mean(tensor, axis)` | `torch.mean(tensor, dim)` |
| Экспонента | `tf.exp(tensor)` | `torch.exp(tensor)` |
| Логарифм | `tf.math.log(tensor)` | `torch.log(tensor)` |

#### **Изменение формы тензора**  
| Операция | TensorFlow | PyTorch |
|----------|-----------|---------|
| Изменение формы | `tf.reshape(tensor, shape)` | `torch.reshape(tensor, shape)` |
| Транспонирование | `tf.transpose(tensor)` | `torch.t(tensor)` (2D) или `tensor.T` |
| Объединение | `tf.concat([a, b], axis)` | `torch.cat([a, b], dim)` |
| Разделение | `tf.split(tensor, parts, axis)` | `torch.split(tensor, parts, dim)` |

---

### **4. Индексация и фильтрация**  

#### **TensorFlow & PyTorch**  
| Операция | Пример |
|----------|--------|
| Индексация | `tensor[0, 1]` → элемент в 0-й строке, 1-м столбце |
| Срезы | `tensor[:, 1:3]` → все строки, столбцы 1 и 2 |
| Булева маска | `tensor[tensor > 0.5]` → значения > 0.5 |

---

### **5. Автоматическое дифференцирование (Autograd)**  

#### **TensorFlow (GradientTape)**  
```python
with tf.GradientTape() as tape:
    y = x ** 2
dy_dx = tape.gradient(y, x)  # dy/dx = 2x
```

#### **PyTorch (Autograd)**  
```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
dy_dx = x.grad  # dy/dx = 2x → 4.0
```

---

### **6. Перемещение тензоров (CPU ↔ GPU)**  

#### **TensorFlow**  
```python
# Если GPU доступен, тензор создаётся на GPU по умолчанию
with tf.device('/GPU:0'):
    tensor = tf.constant([1, 2, 3])
```

#### **PyTorch**  
```python
tensor = torch.tensor([1, 2, 3])
tensor = tensor.to('cuda')  # Перемещение на GPU
```

---

## **🔹 Итог**  
### **TensorFlow**  
✅ Оптимизирован для продакшена (TF Serving, TFLite).  
✅ Использует статический граф (но в TF 2.x есть eager execution).  
✅ Интеграция с Keras для удобного создания моделей.  

### **PyTorch**  
✅ Более гибкий и "питонический" (динамические графы).  
✅ Широко используется в исследованиях.  
✅ Лучшая поддержка GPU через CUDA.  

Обе библиотеки предоставляют:  
- **Создание тензоров** (аналогично NumPy).  
- **Автоматическое дифференцирование** (autograd).  
- **Операции линейной алгебры** (матричные умножения, свёртки).  
- **Работу с GPU** (CUDA).  

Если нужно углубиться в какой-то аспект — дайте знать! 😊