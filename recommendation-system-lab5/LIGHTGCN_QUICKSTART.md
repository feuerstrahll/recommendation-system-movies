"""
🚀 Быстрый старт LightGCN

Шаг 1: Установка зависимостей
=============================
Запусти один из вариантов:

ВАРИАНТ A (Windows .bat):
  install_lightgcn_deps.bat

ВАРИАНТ B (PowerShell):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install torch_geometric tqdm

ВАРИАНТ C (Linux/Mac):
  pip install torch torchvision torchaudio
  pip install torch_geometric tqdm

⏱️  Установка займет 5-15 минут (зависит от интернета)


Шаг 2: Запуск LightGCN
======================
В PowerShell или терминале:

  cd c:\Users\nasty\recsys\lab-5\recommendation-system-lab5
  python recommender/lightgcn_model.py


Шаг 3: Интерпретация результатов
==================================

📊 Метрики (на примере):
  Precision@10   : 0.0324  (3.24% рекомендаций правильных)
  Recall@10      : 0.0652  (6.52% всех фильмов пользователя рекомендованы)
  NDCG@10        : 0.0421  (релевантность с учетом позиции)

⚖️  Сравнение с LightFM:
  - LightGCN медленнее, но учитывает структуру графа
  - LightFM быстрее, хорошо для быстрого прототипирования
  - Оба имеют приблизительно схожие метрики на этом датасете


Шаг 4: Настройка параметров (опционально)
===========================================
В recommender/lightgcn_model.py, функция main():

  emb_dim = 64          # размер эмбеддинга (больше = медленнее)
  n_layers = 3          # слоёв GCN (больше = медленнее, но лучше)
  epochs = 20           # эпох (меньше = быстрее, но менее обучено)
  batch_size = 1024     # размер батча (меньше = медленнее, меньше памяти)
  lr = 0.001            # learning rate


📁 Входные файлы
================
Модель ожидает:
  data/processed/ratings.csv     # колонки: userId, movieId, rating (опционально)
  data/processed/movies.csv      # для справки


🔧 Troubleshooting
==================

❌ ModuleNotFoundError: No module named 'torch'
   → Запусти install_lightgcn_deps.bat и перезагрузи IDE

❌ CUDA out of memory
   → Установи CPU версию PyTorch (см. выше)

❌ Слишком медленно
   → Уменьши emb_dim, n_layers, epochs или batch_size
   → На CPU: ~1-3 минуты на ~1M interactions — нормально

❌ Ошибка в ratings.csv
   → Проверь колонки: userId, movieId (не user_id, movie_id)
   → Код должен автоматически переименовать если нужно


📖 Дополнительная информация
=============================
Запуск из Jupyter/Notebook:

  import sys
  sys.path.insert(0, '/path/to/recommender')
  from lightgcn_model import main
  
  model, user_embs, item_embs, metrics_k10, metrics_k20 = main()

"""
