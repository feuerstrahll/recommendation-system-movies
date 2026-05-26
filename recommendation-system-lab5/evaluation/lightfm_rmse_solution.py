"""
РЕШЕНИЕ ПРОБЛЕМЫ RMSE/MSE ДЛЯ LIGHTFM

Проблема: LightFM предсказывает скоры (неограниченный диапазон),
         а рейтинги пользователя - дискретные [1, 2, 3, 4, 5]

Три подхода к решению:
1. Нормализация + RMSE на нормализованных данных ✅ РЕКОМЕНДУЕМЫЙ
2. Квантизация (дискретизация) предсказаний
3. Явный регрессионный режим LightFM
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightfm import LightFM
from lightfm.data import Dataset as LightFMDataset


class LightFMMetricsCalculator:
    """Расчет всех метрик для LightFM, включая RMSE/MSE."""
    
    def __init__(self):
        self.results = {}
    
    # ============================================================================
    # ПОДХОД 1: НОРМАЛИЗАЦИЯ СКОРОВ В [1, 5] + RMSE
    # ============================================================================
    
    def approach_1_normalization_rmse(self, 
                                      model, 
                                      user_ids_test,
                                      item_ids_test,
                                      true_ratings_test,
                                      user_id_map,
                                      item_id_map):
        """
        Подход 1: Нормализировать предсказания в [1, 5], затем считать RMSE.
        
        Плюсы:
        - Интерпретируемо (предсказания в шкале 1-5)
        - Просто реализовать
        - Работает для любых моделей
        
        Минусы:
        - Нормализация зависит от данных
        - RMSE может быть завышен если распределение асимметрично
        
        Args:
            model: Обученная LightFM модель
            user_ids_test: Тестовые user_id
            item_ids_test: Тестовые item_id
            true_ratings_test: Истинные рейтинги [1, 2, 3, 4, 5]
            user_id_map: Маппинг user_id -> внутренний индекс
            item_id_map: Маппинг item_id -> внутренний индекс
        
        Returns:
            dict с метриками RMSE, MAE, нормализованными предсказаниями
        """
        
        print("\n" + "="*70)
        print("ПОДХОД 1: Нормализация скоров в [1, 5] + RMSE")
        print("="*70)
        
        # Маппируем на внутренние индексы
        user_indices = np.array([user_id_map[uid] for uid in user_ids_test])
        item_indices = np.array([item_id_map[iid] for iid in item_ids_test])
        true_ratings = np.array(true_ratings_test)
        
        # Получаем RAW скоры из модели
        raw_predictions = model.predict(user_indices, item_indices, num_threads=1)
        
        print(f"\nРастределение RAW предсказаний:")
        print(f"  Min: {raw_predictions.min():.4f}")
        print(f"  Max: {raw_predictions.max():.4f}")
        print(f"  Mean: {raw_predictions.mean():.4f}")
        print(f"  Std: {raw_predictions.std():.4f}")
        
        # === НОРМАЛИЗАЦИЯ В [1, 5] ===
        min_score = raw_predictions.min()
        max_score = raw_predictions.max()
        
        # Если все предсказания одинаковые
        if max_score == min_score:
            normalized_predictions = np.full_like(raw_predictions, 3.0)
        else:
            # Линейная нормализация: score_norm = min_val + (score - min) / (max - min) * (max_val - min_val)
            normalized_predictions = 1.0 + 4.0 * (raw_predictions - min_score) / (max_score - min_score)
        
        # Clip на [1, 5] для надежности
        normalized_predictions = np.clip(normalized_predictions, 1, 5)
        
        print(f"\nРаспределение НОРМАЛИЗОВАННЫХ предсказаний [1, 5]:")
        print(f"  Min: {normalized_predictions.min():.4f}")
        print(f"  Max: {normalized_predictions.max():.4f}")
        print(f"  Mean: {normalized_predictions.mean():.4f}")
        print(f"  Std: {normalized_predictions.std():.4f}")
        
        print(f"\nИстинные рейтинги (тестовое множество):")
        print(f"  Min: {true_ratings.min():.4f}")
        print(f"  Max: {true_ratings.max():.4f}")
        print(f"  Mean: {true_ratings.mean():.4f}")
        print(f"  Std: {true_ratings.std():.4f}")
        
        # === СЧИТАЕМ МЕТРИКИ ===
        rmse = np.sqrt(mean_squared_error(true_ratings, normalized_predictions))
        mae = mean_absolute_error(true_ratings, normalized_predictions)
        
        # Корреляция
        correlation = np.corrcoef(true_ratings, normalized_predictions)[0, 1]
        
        print(f"\n🎯 МЕТРИКИ (Нормализованные предсказания):")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  Корреляция Пирсона: {correlation:.4f}")
        
        results = {
            'method': 'normalization',
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation,
            'raw_predictions': raw_predictions,
            'normalized_predictions': normalized_predictions,
            'true_ratings': true_ratings
        }
        
        return results
    
    # ============================================================================
    # ПОДХОД 2: КВАНТИЗАЦИЯ (дискретизация) на [1, 2, 3, 4, 5]
    # ============================================================================
    
    def approach_2_quantization(self, 
                                model,
                                user_ids_test,
                                item_ids_test,
                                true_ratings_test,
                                user_id_map,
                                item_id_map):
        """
        Подход 2: Преобразовать скоры в дискретные оценки [1, 2, 3, 4, 5].
        
        Плюсы:
        - Результаты дискретные (как в реальности)
        - Не зависит от нормализации
        
        Минусы:
        - Потеря информации о степени уверенности
        - Пороги нужно выбирать вручную
        
        Args:
            (же как в подходе 1)
        
        Returns:
            dict с метриками accuracy, confusion matrix и т.д.
        """
        
        print("\n" + "="*70)
        print("ПОДХОД 2: Квантизация (дискретизация) на [1,2,3,4,5]")
        print("="*70)
        
        # Маппируем на внутренние индексы
        user_indices = np.array([user_id_map[uid] for uid in user_ids_test])
        item_indices = np.array([item_id_map[iid] for iid in item_ids_test])
        true_ratings = np.array(true_ratings_test).astype(int)
        
        # Получаем RAW скоры
        raw_predictions = model.predict(user_indices, item_indices, num_threads=1)
        
        # === КВАНТИЗАЦИЯ (метод 1: равные интервалы) ===
        quantiles = np.percentile(raw_predictions, [0, 20, 40, 60, 80, 100])
        quantized_predictions = np.digitize(raw_predictions, bins=quantiles[1:-1])
        quantized_predictions = np.clip(quantized_predictions, 1, 5)  # В [1, 5]
        
        print(f"\nПороги квантизации (по квантилям):")
        print(f"  {quantiles}")
        
        print(f"\nПредсказанные оценки:")
        print(f"  Уникальные: {np.unique(quantized_predictions)}")
        print(f"  Распределение: {np.bincount(quantized_predictions)}")
        
        # === МЕТРИКИ ===
        rmse_quantized = np.sqrt(mean_squared_error(true_ratings, quantized_predictions))
        mae_quantized = mean_absolute_error(true_ratings, quantized_predictions)
        
        # Accuracy (точное совпадение)
        accuracy = np.mean(true_ratings == quantized_predictions)
        
        # Accuracy в пределах ±1
        accuracy_within_1 = np.mean(np.abs(true_ratings - quantized_predictions) <= 1)
        
        print(f"\n🎯 МЕТРИКИ (Квантизированные предсказания):")
        print(f"  RMSE: {rmse_quantized:.4f}")
        print(f"  MAE:  {mae_quantized:.4f}")
        print(f"  Exact Accuracy: {accuracy:.1%}")
        print(f"  Accuracy (±1): {accuracy_within_1:.1%}")
        
        results = {
            'method': 'quantization',
            'rmse': rmse_quantized,
            'mae': mae_quantized,
            'accuracy': accuracy,
            'accuracy_within_1': accuracy_within_1,
            'quantized_predictions': quantized_predictions,
            'true_ratings': true_ratings
        }
        
        return results
    
    # ============================================================================
    # ПОДХОД 3: ЯВНЫЙ РЕГРЕССИОННЫЙ РЕЖИМ LIGHTFM
    # ============================================================================
    
    @staticmethod
    def approach_3_regression_mode_info():
        """
        Подход 3: Использовать LightFM в режиме явного предсказания рейтингов.
        
        Информация о том, как это сделать.
        """
        
        print("\n" + "="*70)
        print("ПОДХОД 3: Явный регрессионный режим LightFM")
        print("="*70)
        
        info = """
Для явного предсказания рейтингов нужно:

1. При создании модели использовать подходящую loss function:
   
   a) 'warp_kl' - для явных рейтингов
      model = LightFM(loss='warp_kl', k=5, learning_rate=0.01)
   
   b) Или преобразовать ratings в implicit feedback (с весами):
      weighted_interactions = interactions * weight_matrix
      model.fit(weighted_interactions, epochs=30)

2. После обучения предсказания будут в диапазоне, 
   более близком к истинным рейтингам.

3. Преимущества:
   ✅ Прямое предсказание рейтингов
   ✅ RMSE считается напрямую
   ✅ Интерпретируемо
   
4. Недостатки:
   ❌ Медленнее обучается (20-50% больше времени)
   ❌ Может быть менее точным для ранжирования

РЕКОМЕНДАЦИЯ: Использовать ПОДХОД 1 (нормализация)
         для текущей модели LightFM, так как она уже обучена.
         
         Для новых экспериментов можно попробовать ПОДХОД 3.
        """
        
        print(info)


def demo_rmse_calculation():
    """Демонстрация расчета RMSE для LightFM."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ДЕМОНСТРАЦИЯ: Как считать RMSE/MSE для LightFM                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Симуляция данных
    print("СИМУЛЯЦИЯ ДАННЫХ:")
    print("-" * 70)
    
    # LightFM обычно предсказывает в диапазоне примерно [-3, 3]
    # (зависит от инициализации и параметров)
    raw_scores = np.random.uniform(-2.5, 2.5, size=1000)
    
    # Истинные рейтинги: дискретные значения 1-5
    true_ratings = np.random.choice([1, 2, 3, 4, 5], size=1000)
    
    print(f"RAW предсказания (диапазон): [{raw_scores.min():.2f}, {raw_scores.max():.2f}]")
    print(f"Истинные рейтинги: {np.unique(true_ratings)}")
    
    # === ПОДХОД 1: НОРМАЛИЗАЦИЯ ===
    print("\n" + "="*70)
    print("ПОДХОД 1 - НОРМАЛИЗАЦИЯ:")
    print("="*70)
    
    min_score = raw_scores.min()
    max_score = raw_scores.max()
    
    normalized = 1.0 + 4.0 * (raw_scores - min_score) / (max_score - min_score)
    normalized = np.clip(normalized, 1, 5)
    
    rmse_norm = np.sqrt(mean_squared_error(true_ratings, normalized))
    mae_norm = mean_absolute_error(true_ratings, normalized)
    
    print(f"RMSE (Нормализованные): {rmse_norm:.4f}")
    print(f"MAE  (Нормализованные): {mae_norm:.4f}")
    print(f"Пример предсказаний: {normalized[:5]}")
    
    # === ПОДХОД 2: КВАНТИЗАЦИЯ ===
    print("\n" + "="*70)
    print("ПОДХОД 2 - КВАНТИЗАЦИЯ:")
    print("="*70)
    
    quantiles = np.percentile(raw_scores, [0, 20, 40, 60, 80, 100])
    quantized = np.digitize(raw_scores, bins=quantiles[1:-1])
    quantized = np.clip(quantized, 1, 5)
    
    rmse_quant = np.sqrt(mean_squared_error(true_ratings, quantized))
    mae_quant = mean_absolute_error(true_ratings, quantized)
    accuracy = np.mean(true_ratings == quantized)
    
    print(f"RMSE (Квантизированные): {rmse_quant:.4f}")
    print(f"MAE  (Квантизированные): {mae_quant:.4f}")
    print(f"Exact Accuracy: {accuracy:.1%}")
    print(f"Пример предсказаний: {quantized[:5]}")
    
    # === СРАВНЕНИЕ ===
    print("\n" + "="*70)
    print("СРАВНЕНИЕ ПОДХОДОВ:")
    print("="*70)
    
    comparison_df = pd.DataFrame({
        'Метрика': ['RMSE', 'MAE'],
        'Нормализация': [f'{rmse_norm:.4f}', f'{mae_norm:.4f}'],
        'Квантизация': [f'{rmse_quant:.4f}', f'{mae_quant:.4f}'],
    })
    
    print(comparison_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("РЕКОМЕНДАЦИЯ:")
    print("="*70)
    print("""
✅ Для LightFM рекомендуется ПОДХОД 1 (нормализация):
   - RMSE: {:.4f}
   - Интерпретируемо в шкале [1, 5]
   - Просто реализовать
   - Работает для существующих моделей
   
   Использовать в evaluation/metrics.py:
   
   def rmse_normalized(raw_predictions, true_ratings):
       min_score = raw_predictions.min()
       max_score = raw_predictions.max()
       normalized = 1 + 4 * (raw_predictions - min_score) / (max_score - min_score)
       return np.sqrt(mean_squared_error(true_ratings, normalized))
    """.format(rmse_norm))


if __name__ == '__main__':
    demo_rmse_calculation()
