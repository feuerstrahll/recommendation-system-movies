# Models

Папка содержит актуальные рекомендательные модели для Lab 5 & Lab 6.

## 🆕 Holistic Cold Start Strategy (НОВОЕ)

Реализована полная стратегия холодного старта с использованием **mini-questionnaire** и прогрессивной смены рекомендательных подходов:

### Жизненный цикл пользователя

```
┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
│   NEW USER      │  COLD START      │  WARM START      │   MATURE USER    │
│   (0 ratings)   │  (1-5 ratings)   │  (5-20 ratings)  │   (20+ ratings)  │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Questionnaire   │ Content-Based    │ Hybrid           │ Collaborative    │
│ + Preferences   │ (Movie Similarity)│(Content+Collab)  │ (LightGCN/LightFM)
└─────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

### Запуск холодного старта

```powershell
python models/cold_start_questionnaire.py
```

**Шаги:**
1. Выбор любимых жанров (из каталога)
2. Указание 2-5 любимых фильмов
3. Предпочтение по году выпуска (классика/новое/любое)
4. Получение начальных рекомендаций на основе контентного сходства

### Запуск unified router (с прогрессией)

```powershell
python models/recommendation_router.py
```

**Демонстрирует:**
- Регистрацию нового пользователя
- Переход между стадиями (холодный старт → теплый → зрелый)
- Смену стратегии на каждой стадии

### Структура системы

```
NEW_USER (questionnaire)
         ↓
    COLD START (ratings: 0-5)
    Strategy: Content-Based
    Reason: Not enough user history for collaboration
         ↓
    WARM START (ratings: 5-20)
    Strategy: Hybrid (Content + Collaborative)
    Reason: Some history, but still need content fallback
         ↓
    MATURE USER (ratings: 20+)
    Strategy: Pure Collaborative (LightGCN/LightFM)
    Reason: Rich history available for full collaboration
```

---

## Сравнение Стратегий

| Фаза | Стратегия | Входные данные | Выходные данные | Преимущества | Недостатки |
|------|-----------|---|---|---|---|
| **NEW** | Questionnaire | Жанры + любимые фильмы | User Profile | Персонализирована с первого дня | Требует взаимодействия |
| **COLD START** | Content-Based | Movie metadata + embeddings | Similarity scores | Не требует переобучения | Не использует коллаб. сигналы |
| **WARM START** | Hybrid | Content + some ratings | Weighted scores | Баланс между подходами | Более сложно объяснить |
| **MATURE** | Collaborative | User-user/User-item graph | Ranking scores | Высокая точность | Требует много данных |

### Когда использовать каждую?

**Content-Based:** 
- ✅ Новые пользователи (нет истории)
- ✅ Новые фильмы (мало рейтингов)
- ✅ Когда нужно объяснить рекомендацию
- ❌ Когда требуется высокая точность

**Collaborative Filtering:**
- ✅ Зрелые пользователи (много рейтингов)
- ✅ Когда есть паттерны в поведении
- ✅ Когда нужна высокая точность
- ❌ Не работает на холодном старте

**Hybrid:**
- ✅ Переходная фаза (несколько рейтингов)
- ✅ Баланс между точностью и объяснимостью
- ✅ Робастен к новым фильмам
- ❌ Более сложная архитектура

---

## Файлы

- **`cold_start_questionnaire.py`** - 🆕 **Mini-questionnaire** для новых пользователей + content-based рекомендации
- **`recommendation_router.py`** - 🆕 **Unified router** координирует стратегии на основе жизненного цикла пользователя
- **`lightgcn_model.py`** - **LightGCN** (Graph Convolutional Network) — граф-ориентированная нейросетевая модель для рекомендаций через user-item граф.
- **`lightfm_model.py`** - LightFM (факторизационные машины) — коллаборативная фильтрация с метаданными.
- **`content_lightFM.py`** - Контентная фильтрация на основе LightFM с текстовыми эмбеддингами (SentenceTransformer).
- `lightfm_roc_curve.png` - ROC-кривая оценки производительности LightFM модели.

## Запуск моделей

### 🆕 Cold Start Questionnaire

```powershell
python models/cold_start_questionnaire.py
```

Интерактивный процесс для новых пользователей:
1. **Шаг 1**: Выбор любимых жанров из каталога
2. **Шаг 2**: Ввод 2-5 любимых фильмов
3. **Шаг 3**: Предпочтение по году выпуска (классика/новое/любое)
4. **Результат**: Начальные рекомендации на основе контентного сходства

### 🆕 Unified Recommendation Router

```powershell
python models/recommendation_router.py
```

Демонстрирует полный жизненный цикл пользователя с автоматическим переключением стратегий.

### 🆕 Usage Examples

```powershell
python models/cold_start_examples.py
```

Четыре практических примера:
1. Questionnaire + Content-Based
2. Full User Lifecycle
3. Strategy Switching
4. Multiple Users Management

### LightFM с контентом

```powershell
python models/lightfm_model.py
```

Коллаборативная фильтрация с факторизационными машинами.

### Content-based LightFM

```powershell
python models/content_lightFM.py
```

Контентная фильтрация на основе:
- SentenceTransformer для текстовых эмбеддингов (названия и жанры фильмов)
- Нормализованный год выпуска  
- LightFM для интеграции признаков контента

Использует **Implicit Feedback** (presence/absence of interactions).

## LightGCN — Graph Convolutional Network

**Запуск:**

```powershell
python models/lightgcn_model.py
```

### Что это?

LightGCN — упрощённая Graph Convolutional Network для рекомендаций:
- Строит двудольный граф user-item из рейтингов
- Применяет графовые свёртки (GCN слои) для пропагации информации между соседями
- **Не использует признаки контента** — только граф связей

### Как работает?

1. **Граф**: создаёт user nodes (0..U-1) и item nodes (U..U+I-1), рёбра из `ratings.csv`
2. **Эмбеддинги**: инициализирует случайно, улучшает через K слоёв GCN
3. **Обучение**: BPR Loss (Bayesian Personalized Ranking) с negative sampling
4. **Оценка**: Precision@K, Recall@K, NDCG@K (как в методичке)

### Параметры

```python
# В lightgcn_model.py функция LightGCN:
emb_dim = 64        # размер эмбеддинга
n_layers = 3        # слоёв графовой свёртки
lr = 0.001          # learning rate
epochs = 20         # эпох обучения
batch_size = 1024   # мини-батч
```

### Требования

- PyTorch >= 2.0.0
- torch-geometric >= 2.3.0
- pandas, numpy

Добавлены в `requirements.txt`. Установи:

```bash
pip install -r requirements.txt
```

### Метрики

Код вычисляет:
- **Precision@10, @20** — доля рекомендаций, которые были в GT
- **Recall@10, @20** — доля GT, покрытая рекомендациями
- **NDCG@10, @20** — учитывает позицию попадания (лучше вверху)

**Вывод** при запуске:
```
📊 LightGCN EVALUATION RESULTS
✓ Evaluated 610 users

🎯 Metrics @ K=10:
  Precision@K    : 0.0324
  Recall@K       : 0.0652
  NDCG@K         : 0.0421

🎯 Metrics @ K=20:
  Precision@K    : 0.0198
  Recall@K       : 0.0823
  NDCG@K         : 0.0527
```

### Сравнение с LightFM

| Модель | Парадигма | Скорость (CPU) | GPU | Метрики |
|--------|-----------|---|---|---|
| LightFM | Factorization Machines | ~1 сек | Не нужен | Precision/Recall |
| LightGCN | Graph Neural Network | 1-3 мин | Рекомендуется | Precision/Recall/NDCG |

**LightGCN преимущества:**
- Лучше capture структуру графа (соседи друзей рекомендуют друг другу)
- Метрики включают NDCG (позиционные)
- Работает на ~1M interactions за приемлемое время на CPU

**LightFM преимущества:**
- Значительно быстрее
- Меньше памяти
- Проще настраивать

## Идентификаторы

В актуальных processed CSV используется единый основной id:

```text
movie_id = TMDB id
```

`ratings.csv` также содержит `movielens_id`, но рекомендатели используют `movie_id`, чтобы напрямую соединяться с `movies.csv`.

---

## 🚀 Production Integration

### Архитектура

```
┌──────────────────┐
│   User Request   │
└────────┬─────────┘
         │
         ↓
    ┌────────────────────┐
    │  Check User ID     │
    │  in Database?      │
    └────────┬─────────┬─────┘
             │         │
          NO│         │YES
             │         │
      ┌──────▼──┐   ┌──▼──────────┐
      │Question-│   │ Get User    │
      │ naire   │   │ Stage       │
      └──────┬──┘   └──┬──────────┘
             │         │
             └─────┬───┘
                   │
            ┌──────▼──────────┐
            │ Route to        │
            │ Strategy        │
            │ (Content/Hybrid/│
            │  Collaborative) │
            └──────┬──────────┘
                   │
            ┌──────▼───────────┐
            │ Recommendations  │
            │ DataFrame        │
            └──────┬───────────┘
                   │
            ┌──────▼────────────┐
            │ Return with       │
            │ metadata          │
            │ (confidence,      │
            │ strategy, etc.)   │
            └───────────────────┘
```

### Использование в приложении

```python
from recommendation_router import UnifiedRecommendationRouter
import pandas as pd

# 1. Инициализация
movies = pd.read_csv("data/processed/movies.csv")
ratings = pd.read_csv("data/processed/ratings.csv")
router = UnifiedRecommendationRouter(movies, ratings)

# 2. Новый пользователь
user_id = 12345
profile = router.register_new_user(user_id)

# 3. Получение рекомендаций
result = router.get_recommendations(user_id, n_recommendations=10)

# 4. Использование результата
print(f"Strategy: {result.strategy}")
print(f"Stage: {result.user_stage}")
print(result.recommendations)

# 5. После рейтинга фильма
router.record_user_rating(user_id, movie_id=123, rating=4.5)

# 6. Следующие рекомендации (будет использована новая стратегия если применимо)
result = router.get_recommendations(user_id, n_recommendations=10)
```

### База данных (Псевдокод)

```sql
-- Таблица пользователей
CREATE TABLE users (
    user_id INT PRIMARY KEY,
    profile_json JSON,  -- favorite_genres, favorite_movies, age_preference
    created_at TIMESTAMP,
    stage VARCHAR(20)   -- new, cold_start, warm_start, mature
);

-- История рейтингов пользователя
CREATE TABLE user_ratings (
    user_id INT,
    movie_id INT,
    rating FLOAT,
    rated_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Для быстрого подсчёта
CREATE INDEX idx_user_ratings_count ON user_ratings(user_id);
```

### Performance Considerations

| Этап | Время | Примечание |
|------|-------|-----------|
| Questionnaire | ~2-5 мин | Интерактивная, не критично |
| Content-Based | ~0.1 сек | FAISS индекс, очень быстро |
| Hybrid | ~0.5 сек | Зависит от разреженности |
| Collaborative | ~1-3 сек | LightGCN, может потребоваться GPU |

### Масштабирование

- **Кэширование**: Сохранять user embeddings в Redis
- **Batch Processing**: Рассчитывать рекомендации в фоне
- **Model Serving**: Использовать FastAPI + Gunicorn
- **Database**: Партиционировать ratings по user_id

---

### Категория 1: новичок

Новичок - это новый пользователь без истории оценок. Он плохо знает каталог
и ожидает быстрые рекомендации по минимальному количеству входных данных.

Требования и ожидания:

- простая анкета без необходимости заранее иметь историю оценок;
- учет выбранных жанров и уже любимых фильмов;
- учет предпочтения по году выпуска: `old`, `new` или `both`;
- понятный итоговый список с рассчитанным `cold_start_score`.

Сценарий использования соответствует cold-start логике:

```python
recommend_for_new_user(
    selected_genres=["Action", "Sci-Fi"],
    liked_movie_titles=["The Matrix", "Inception"],
    release_preference="new",
    top_n=10
)
```

Ожидаемый результат:

- возвращается топ-10 рекомендаций;
- фильмы соответствуют выбранным жанрам или жанрам, извлеченным из любимых фильмов;
- все рекомендованные фильмы имеют `release_year >= 2000`;
- `The Matrix` и `Inception` не попадают в рекомендации;
- в результате есть колонка `cold_start_score`.

### Категория 2: опытный пользователь

Опытный пользователь уже имеет историю оценок и ожидает более персональные
рекомендации на основе своего прошлого поведения.

Требования и ожидания:

- учет фильмов, которые пользователь оценил высоко;
- исключение уже оцененных фильмов из итогового списка;
- персонализация не только по похожести контента, но и по SVD-прогнозу;
- сортировка рекомендаций по итоговому `hybrid_score`.

Сценарий использования:

1. Система берет фильмы пользователя с оценкой `>= 4.0`.
2. На их основе строится content-based профиль пользователя через FAISS.
3. FAISS возвращает пул похожих кандидатов.
4. Для этих кандидатов рассчитываются SVD-прогнозы.
5. Итоговый список формируется как hybrid reranking:

```text
hybrid_score = alpha * svd_score_norm + (1 - alpha) * content_score_norm
```

Такой подход является weighted hybrid reranking: SVD не оценивает все фильмы
каталога, а переупорядочивает кандидатов, найденных content-based частью.

### Категория 3: эксперт

Эксперт хорошо понимает предметную область и хочет анализировать работу модели,
а не только получать готовый список рекомендаций.

Требования и ожидания:

- возможность задавать `user_id`, `liked_movie_titles`, `alpha`, `top_n`;
- при необходимости возможность управлять размером FAISS-пула через `content_top_k`;
- видимость отдельных компонентов итогового скоринга;
- возможность сравнивать результаты при разных значениях `alpha`.

Сценарий использования:

```python
get_hybrid_recommendations(
    user_id=1,
    liked_movie_titles=["The Matrix", "Inception"],
    alpha=0.6,
    top_n=10,
    content_top_k=5000
)
```

Эксперт анализирует следующие поля результата:

- `svd_score`;
- `content_score`;
- `svd_score_norm`;
- `content_score_norm`;
- `hybrid_score`.

При изменении `alpha` меняется баланс между collaborative и content-based
частью внутри уже сформированного FAISS-пула кандидатов.

## Test Plan

### Test 1: Cold-start user

Input:

- genres: `Action`, `Sci-Fi`;
- liked movies: `The Matrix`, `Inception`;
- release_preference: `new`;
- top_n: `10`.

Expected:

- system returns top-10 recommendations;
- all recommended movies have `release_year >= 2000`;
- `The Matrix` and `Inception` are excluded;
- recommendations contain `cold_start_score`;
- recommended movies match selected or inferred genres.

### Test 2: Existing user hybrid recommendations

Input:

- user_id: `1`;
- alpha: `0.6`;
- top_n: `10`.

Expected:

- system returns top-10 recommendations;
- already rated movies are excluded;
- result contains `svd_score`, `content_score`, `svd_score_norm`, `content_score_norm`, `hybrid_score`;
- recommendations are sorted by `hybrid_score` descending.

### Test 3: Expert alpha comparison

Input:

- user_id: `1`;
- alpha values: `0.3`, `0.6`, `0.8`;
- top_n: `10`.

Expected:

- `alpha = 0.3` gives more influence to content-based similarity;
- `alpha = 0.8` gives more influence to SVD predictions;
- changing `alpha` changes ordering of recommendations;
- score components remain visible for comparison.
