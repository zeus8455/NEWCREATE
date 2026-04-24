# PokerVision

Рабочий каркас проекта по ТЗ PokerVision.

## Зависимости

Минимум для mock/headless:

```powershell
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pip install numpy opencv-python
```

Для реального захвата и YOLO:

```powershell
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pip install numpy opencv-python mss ultralytics
```

Для UI:

```powershell
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pip install pyside6
```

## Важно по путям моделей

В `pokervision/config.py` лучше указывать путь сразу к `best.pt`.
Но если указан путь к папке `weights`, проект сам попробует найти:
- `best.pt`
- `last.pt`
- первый найденный `*.pt`

## Правильные запуски

Перейти в корень проекта:

```powershell
cd C:\PokerAI\PokerVision\Python_PY
```

### 1. Запуск mock без UI

```powershell
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pokervision.main --mock --headless --iterations 8
```

### 2. Запуск mock с UI

```powershell
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pokervision.main --mock
```

### 3. Реальный режим без UI

```powershell
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pokervision.main --real --headless --iterations 10
```

### 4. Реальный режим с UI

```powershell
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pokervision.main --real
```

### 5. Прямой запуск файла тоже поддержан

```powershell
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe C:\PokerAI\PokerVision\Python_PY\pokervision\main.py --mock --headless --iterations 8
```

## Запуск тестов

Из корня проекта:

```powershell
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m unittest discover -s pokervision/tests -p "test_*.py"
```

## Куда сохраняются файлы

По умолчанию:

```text
C:\PokerAI\PokerVision\PokerVision_DataSafeFiles
```

### Успешные раздачи
- `hands/hand_000001/...`

### Ошибочные кадры до создания hand
- `temp/failed_frames/<stage>/<frame_id>/...`

Это сделано специально, чтобы можно было смотреть, где ломается pipeline.
