# В файле model_creation cначала идет код создания SVD
## Потом показан как работает faiss

### Первая итерация пайплайна
**Лог выполнения:** `Running LightFM recommendation pipeline...`


| Название фильма (Title) | Оценка LightFM (Score) |
| :--- | :--- |
| The Dark Backward | 2.144949e+11 |
| The Canyon | 8.656903e+10 |
| Why We Fight: The Nazis Strike | 5.612668e+10 |
| Glory Road | 3.515324e+10 |
| The Winning Season | 3.362758e+10 |
| Jaco | 3.037592e+10 |
| After Sex | 2.830315e+10 |
| The Wolfpack | 2.334469e+10 |
| Bluebeard | 2.259910e+10 |
| Bag It | 1.964603e+10 |

**Статус:** `done`

---

### Вторая итерация пайплайна
**Команда запуска:** `(venv) den@Win11-PC:/mnt/d/recommendation-system-movies-main/models$ python lightfm_model.py`  
**Лог выполнения:** `Running LightFM recommendation pipeline...`

#### Топ-10 Рекомендаций (Набор А)


| Название фильма (Title) | Оценка LightFM (Score) |
| :--- | :--- |
| Sublime | 6.467534e+11 |
| The Presence | 3.834368e+10 |
| The Far Country | 2.845057e+09 |
| Northwest Passage | 7.014341e+08 |
| Under the Same Moon | 3.701562e+08 |
| 96 Minutes | 1.896165e+08 |
| H.M.S. Defiant | 1.740739e+08 |
| Guns of the Magnificent Seven | 1.732493e+08 |
| 100 Rifles | 1.686234e+08 |
| Cry, the Beloved Country | 1.453672e+08 |

#### Топ-10 Рекомендаций (Набор Б)


| Название фильма (Title) | Оценка LightFM (Score) |
| :--- | :--- |
| The Decline of Western Civilization Part II: The Metal Years | 1.859880e+08 |
| See Spot Run | 4.798546e+06 |
| Bodies, Rest & Motion | 2.289886e+06 |
| The Invisible War | 2.125554e+06 |
| Querelle | 1.705374e+06 |
| The Anniversary Party | 1.045288e+06 |
| Sybil | 9.986668e+05 |
| Nothing Personal | 9.867862e+05 |
| W.R. - Mysteries of the Organism | 9.846798e+05 |
| Tuvalu | 5.415688e+05 |
#### Metrtics

--- LightFM Native Metrics ---
ROC AUC Score (Sampled 500 Users): 0.7277
done
