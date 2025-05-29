Отлично! Модуль **`re`** в Python предоставляет функции для работы с **регулярными выражениями** (regular expressions, regex) — мощным инструментом для поиска и обработки текста.  

---

## 🔹 **1. Основные функции модуля `re`**  

### **`re.search(pattern, string)`** – поиск первого совпадения  
Возвращает `Match` объект или `None`, если совпадений нет.  
```python
import re

text = "Python 3.10 released in 2021"
match = re.search(r'\d+\.\d+', text)  # Ищем версию (число.число)
if match:
    print(match.group())  # "3.10"
```

### **`re.match(pattern, string)`** – проверка совпадения **с начала строки**  
```python
text = "Python is awesome"
match = re.match(r'Python', text)  # Совпадение есть
print(match.group())  # "Python"
```

### **`re.findall(pattern, string)`** – поиск **всех** совпадений  
Возвращает список всех найденных подстрок.  
```python
text = "10 apples, 5 bananas, 3 oranges"
numbers = re.findall(r'\d+', text)
print(numbers)  # ['10', '5', '3']
```

### **`re.finditer(pattern, string)`** – итератор с `Match` объектами  
Полезно для обработки больших текстов.  
```python
text = "Email me at user@example.com or support@site.org"
emails = re.finditer(r'[\w.-]+@[\w.-]+', text)
for email in emails:
    print(email.group())
# user@example.com
# support@site.org
```

### **`re.sub(pattern, repl, string)`** – замена совпадений  
```python
text = "Python 2.7 is old, use Python 3.10"
updated = re.sub(r'2\.7', '3.10', text)
print(updated)  # "Python 3.10 is old, use Python 3.10"
```

### **`re.split(pattern, string)`** – разделение строки по шаблону  
```python
text = "Apple,Banana;Orange|Grape"
fruits = re.split(r'[,;|]', text)  # Разделяем по , ; или |
print(fruits)  # ['Apple', 'Banana', 'Orange', 'Grape']
```

---

## 🔹 **2. Синтаксис регулярных выражений**  

### **Базовые метасимволы**  
| Паттерн  | Описание                     | Пример                     |
|----------|------------------------------|----------------------------|
| `.`      | Любой символ (кроме `\n`)    | `a.c` → "abc", "a c"       |
| `\d`     | Цифра (`[0-9]`)              | `\d+` → "123"              |
| `\D`     | Не цифра (`[^0-9]`)          | `\D+` → "abc"              |
| `\w`     | Буква, цифра или `_`         | `\w+` → "user123"          |
| `\W`     | Не `\w`                      | `\W+` → "@#$"              |
| `\s`     | Пробельный символ            | `\s+` → " \t\n"            |
| `\S`     | Не пробельный символ         | `\S+` → "abc"              |

### **Квантификаторы (повторы)**  
| Паттерн  | Описание                     | Пример                     |
|----------|------------------------------|----------------------------|
| `*`      | 0 или больше                 | `a*` → "", "a", "aa"       |
| `+`      | 1 или больше                 | `\d+` → "1", "123"         |
| `?`      | 0 или 1                      | `a?b` → "b", "ab"          |
| `{n}`    | Ровно `n` раз                | `a{3}` → "aaa"             |
| `{n,}`   | `n` или больше               | `\d{2,}` → "12", "123"     |
| `{n,m}`  | От `n` до `m` раз            | `a{2,4}` → "aa", "aaa"     |

### **Группы и условия**  
| Паттерн         | Описание                     | Пример                     |
|-----------------|------------------------------|----------------------------|
| `(...)`         | Группа                       | `(ab)+` → "abab"           |
| `(?:...)`       | Незахватывающая группа       | `(?:ab)+` → "abab"         |
| `\|`            | ИЛИ                          | `cat\|dog` → "cat", "dog"  |
| `^`             | Начало строки                | `^Hello` → "Hello world"   |
| `$`             | Конец строки                 | `end$` → "The end"         |

---

## 🔹 **3. Флаги (модификаторы)**  
Флаги изменяют поведение регулярного выражения:  

| Флаг            | Описание                     | Пример                     |
|-----------------|------------------------------|----------------------------|
| `re.IGNORECASE` | Игнорировать регистр         | `re.search(r'python', 'PYTHON', re.I)` |
| `re.MULTILINE`  | `^` и `$` для каждой строки  | `re.findall(r'^\d+', '1\n2\n3', re.M)` → ["1", "2", "3"] |
| `re.DOTALL`     | `.` включает `\n`            | `re.search(r'a.b', 'a\nb', re.S)` |

Пример:  
```python
text = "Python\npython\nPYTHON"
matches = re.findall(r'python', text, re.IGNORECASE)
print(matches)  # ['Python', 'python', 'PYTHON']
```

---

## 🔹 **4. Примеры использования**  

### **Проверка email**  
```python
email = "user@example.com"
pattern = r'^[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}$'
if re.match(pattern, email):
    print("Valid email")
```

### **Извлечение дат**  
```python
text = "Dates: 2023-01-15, 2024/12/31, 01.05.2025"
dates = re.findall(r'\d{4}[-/.]\d{2}[-/.]\d{2}', text)
print(dates)  # ['2023-01-15', '2024/12/31', '01.05.2025']
```

### **Удаление HTML-тегов**  
```python
html = "<p>Hello <b>world</b></p>"
clean = re.sub(r'<[^>]+>', '', html)
print(clean)  # "Hello world"
```

---

## 💡 **Когда использовать `re`?**  
- Для валидации данных (email, номера телефонов).  
- Для парсинга текста (логи, HTML, CSV).  
- Для сложного поиска и замены в строках.  

Хочешь разобрать конкретный кейс? 😊