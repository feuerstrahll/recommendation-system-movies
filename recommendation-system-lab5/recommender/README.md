# Recommender

Папка содержит актуальные рекомендательные сценарии.

## Файлы

- `questionnaire.py` - cold-start рекомендации для нового пользователя.
- `hybrid.py` - гибридная модель: content-based FAISS + SVD.
- `content_based&colaborative.ipynb` - исследовательский ноутбук для content-based и collaborative подходов.

Файлы `content_based.py` и `collaborative.py` в текущей версии проекта отсутствуют. Старые ссылки на них в IDE или `demo.py` относятся к предыдущей структуре.

## Идентификаторы

В актуальных processed CSV используется единый основной id:

```text
movie_id = TMDB id
```

`ratings.csv` также содержит `movielens_id`, но рекомендатели используют `movie_id`, чтобы напрямую соединяться с `movies.csv`.

## Cold Start

Запуск анкеты:

```powershell
python recommender/questionnaire.py
```

Метод `recommend_for_new_user()` учитывает:

- выбранные жанры;
- фильмы, которые пользователь уже любит;
- предпочтение по году выхода: `old`, `new`, `both`;
- `genre_match_score`;
- `vote_average`, `popularity`, `vote_count`.

Итоговая формула:

```text
cold_start_score =
    genre_match_score * 0.35
  + vote_average_norm * 0.30
  + popularity_norm * 0.25
  + vote_count_norm * 0.10
```

## Hybrid

Запуск:

```powershell
python recommender/hybrid.py
```

`hybrid.py` ожидает:

- `data/processed/movies.csv`;
- `data/processed/ratings.csv`;
- SVD-модель в `models/collab_model_svd.pkl` или `models/collab_model_svd`.

SVD должна быть обучена на текущем наборе колонок:

```python
["user_id", "movie_id", "rating"]
```

## FAISS и эмбеддинги

В `content_based&colaborative.ipynb` кэширование уже встроено. Ноутбук сначала проверяет файлы:

```text
models/cache/movie_features.npy
models/cache/movies_faiss.index
```

Если они есть и совпадают по размеру с текущим `movies.csv`, notebook загружает их:

```python
normalized_features = np.load(FEATURES_PATH)
faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
```

Если кэша нет или он несовместим, notebook пересчитывает эмбеддинги, строит FAISS index и сохраняет их в `models/cache/`.
Это убирает повторный пересчет эмбеддингов для всех фильмов при следующих запусках.

## Задание 4. Тестирование пользовательских сценариев

Для проверки рекомендательной системы выделены три категории пользователей:
новичок, опытный пользователь и эксперт. Сценарии ниже привязаны к текущей
реализации в `questionnaire.py` и `hybrid.py`.

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
