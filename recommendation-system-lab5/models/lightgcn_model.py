"""
LightGCN Model - Graph Convolutional Network for Recommendation System
Полная реализация LightGCN на PyTorch Geometric с метриками Precision@K, Recall@K, NDCG@K
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import coalesce, degree
from tqdm import tqdm
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================================
# 1. ПОДГОТОВКА ДАННЫХ (Pandas -> PyG Graph)
# ==========================================================

def prepare_lightgcn_data(ratings_df, min_user_interactions=0):
    """
    Преобразование Pandas DataFrame в граф для LightGCN.
    
    Работает со столбцами: 'userId', 'movieId' (или 'rating')
    
    Args:
        ratings_df: DataFrame с колонками 'userId', 'movieId'
        min_user_interactions: минимальное количество взаимодействий пользователя
    
    Returns:
        edge_index, n_users, n_items, user_inv, item_inv, gt_dict, user_map, item_map
    """
    # Переименовываем колонки если нужно (для совместимости)
    if 'user_id' in ratings_df.columns and 'userId' not in ratings_df.columns:
        ratings_df = ratings_df.rename(columns={'user_id': 'userId', 'movie_id': 'movieId'})
    
    # Фильтрируем пользователей с малым числом взаимодействий
    if min_user_interactions > 0:
        user_counts = ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        ratings_df = ratings_df[ratings_df['userId'].isin(valid_users)].copy()
    
    # Маппинг ID к непрерывным индексам (требование эмбеддингов)
    unique_users = ratings_df['userId'].unique()
    unique_items = ratings_df['movieId'].unique()
    
    user_map = {u: i for i, u in enumerate(unique_users)}
    item_map = {m: i for i, m in enumerate(unique_items)}
    
    n_users = len(user_map)
    n_items = len(item_map)
    
    # Обратные маппинги для интерпретации результатов
    user_inv = {v: k for k, v in user_map.items()}
    item_inv = {v: k for k, v in item_map.items()}
    
    # Создаём двудольный граф: 
    # user nodes: 0..U-1, item nodes: U..U+I-1
    src = ratings_df['userId'].map(user_map).values
    dst = ratings_df['movieId'].map(item_map).values + n_users
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # Делаем граф неориентированным (двусторонние рёбра)
    edge_index_undirected = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    edge_index_undirected, _ = coalesce(
        edge_index_undirected, 
        num_nodes=n_users + n_items
    )
    
    # Собираем ground-truth: {user_id: [item_ids]}
    gt_dict = ratings_df.groupby('userId')['movieId'].apply(list).to_dict()
    
    print(f"📊 Data prepared: {n_users} users, {n_items} items, "
          f"{edge_index.shape[1]} interactions")
    
    return edge_index_undirected, n_users, n_items, user_inv, item_inv, gt_dict, user_map, item_map


# ==========================================================
# 2. МОДЕЛЬ LIGHTGCN
# ==========================================================

class LightGCN(nn.Module):
    """
    LightGCN: Light Graph Convolutional Network
    
    Упрощённая GCN для рекомендаций с:
    - Эмбеддингами пользователей и фильмов
    - Графовыми сверточными слоями (GCN layers)
    - Усреднением эмбеддингов всех слоёв
    """
    
    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        
        # Инициализация эмбеддингов
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
    
    def forward(self, edge_index):
        """
        Forward pass LightGCN.
        
        Args:
            edge_index: граф (2, num_edges)
        
        Returns:
            user_embs: эмбеддинги пользователей (n_users, emb_dim)
            item_embs: эмбеддинги фильмов (n_items, emb_dim)
        """
        # Объединяем эмбеддинги (пользователи + фильмы)
        emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [emb]
        
        # Строим матрицу смежности с нормализацией
        row, col = edge_index
        deg = degree(col, emb.size(0), dtype=emb.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Нормализованные веса рёбер: D^(-1/2) @ A @ D^(-1/2)
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Разреженная матрица смежности
        adj = torch.sparse_coo_tensor(
            edge_index, 
            edge_weight, 
            (emb.size(0), emb.size(0))
        ).to(device)
        
        # K слоёв графовой свёртки
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(adj, emb)
            embs.append(emb)
        
        # Усреднение эмбеддингов всех слоёв (ключевая идея LightGCN)
        embs = torch.stack(embs, dim=0)  # (K+1, N, D)
        embs = torch.mean(embs, dim=0)   # (N, D)
        
        # Разделяем на пользователей и фильмы
        user_embs = embs[:self.n_users]
        item_embs = embs[self.n_users:]
        
        return user_embs, item_embs


# ==========================================================
# 3. ОБУЧЕНИЕ (BPR Loss + Negative Sampling)
# ==========================================================

def train_lightgcn(model, edge_index, gt_dict, n_users, n_items, 
                   user_map, item_map,
                   lr=0.001, epochs=20, batch_size=1024, neg_samples=1):
    """
    Обучение LightGCN с BPR (Bayesian Personalized Ranking) loss.
    
    Args:
        model: LightGCN модель
        edge_index: граф
        gt_dict: ground-truth взаимодействия {user_id: [item_ids]}
        n_users, n_items: размеры
        user_map, item_map: маппинги ID
        lr: learning rate
        epochs: количество эпох
        batch_size: размер батча
        neg_samples: количество негативных сэмплов
    
    Returns:
        user_embs, item_embs: обученные эмбеддинги
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    edge_index = edge_index.to(device)
    
    # Собираем валидные пользователи с positive items
    all_users_mapped = []
    all_pos = []
    
    for u_original_id in gt_dict.keys():
        if u_original_id in user_map and len(gt_dict[u_original_id]) > 0:
            all_users_mapped.append(u_original_id)
            all_pos.append(gt_dict[u_original_id])
    
    model.train()
    losses_history = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(
            range(0, len(all_users_mapped), batch_size),
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False
        )
        
        for start in pbar:
            optimizer.zero_grad()
            
            user_embs, item_embs = model(edge_index)
            
            end = min(start + batch_size, len(all_users_mapped))
            u_batch = all_users_mapped[start:end]
            
            # Маппируем пользователей
            u_idx = torch.tensor([user_map[u] for u in u_batch], device=device)
            
            # Positive items (случайно выбираем из известных)
            pos_items = []
            for i in range(len(u_batch)):
                pos_items.append(np.random.choice(all_pos[start + i]))
            pos_idx = torch.tensor([item_map[i] for i in pos_items], device=device)
            
            # Negative items (случайно из всех фильмов)
            neg_idx = torch.randint(0, n_items, (len(u_batch),), device=device)
            
            # Вычисляем скалярные произведения (scores)
            u_emb = user_embs[u_idx]
            pos_emb = item_embs[pos_idx]
            neg_emb = item_embs[neg_idx]
            
            pos_scores = (u_emb * pos_emb).sum(dim=1)
            neg_scores = (u_emb * neg_emb).sum(dim=1)
            
            # BPR Loss: log-sigmoid(pos - neg)
            batch_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        losses_history.append(avg_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    # Финальное вычисление эмбеддингов без градиентов
    model.eval()
    with torch.no_grad():
        user_embs, item_embs = model(edge_index)
    
    return user_embs, item_embs, losses_history


# ==========================================================
# 4. МЕТРИКИ (Precision@K, Recall@K, NDCG@K)
# ==========================================================

def evaluate_model(user_embs, item_embs, gt_dict, user_inv, item_inv, 
                   user_map, item_map, k=10):
    """
    Оценка модели по метрикам Precision@K, Recall@K, NDCG@K.
    
    Args:
        user_embs, item_embs: эмбеддинги из модели
        gt_dict: ground-truth {user_id: [item_ids]}
        user_inv, item_inv: обратные маппинги
        user_map, item_map: прямые маппинги
        k: количество рекомендаций
    
    Returns:
        dict с метриками
    """
    user_embs_cpu = user_embs.cpu().detach()
    item_embs_cpu = item_embs.cpu().detach()
    
    precisions, recalls, ndcgs = [], [], []
    
    for u_original_id in tqdm(gt_dict.keys(), desc=f"Evaluating @ K={k}", leave=False):
        # Проверяем, есть ли пользователь в маппинге
        if u_original_id not in user_map:
            continue
        
        u_mapped_idx = user_map[u_original_id]
        true_items = set(gt_dict[u_original_id])
        
        if len(true_items) == 0:
            continue
        
        # Вычисляем scores между пользователем и всеми фильмами
        u_emb = user_embs_cpu[u_mapped_idx]  # (emb_dim,)
        scores = u_emb @ item_embs_cpu.T     # (n_items,)
        
        # Исключаем уже взаимодействовавшие фильмы
        # Маппируем known items
        known_items_mapped = set()
        for item_id in true_items:
            if item_id in item_map:
                known_items_mapped.add(item_map[item_id])
        
        scores_copy = scores.clone()
        for item_idx in known_items_mapped:
            scores_copy[item_idx] = -1e9
        
        # Top-K рекомендации
        top_k_indices = torch.topk(scores_copy, k=min(k, len(scores))).indices.tolist()
        
        # Маппируем обратно в исходные ID
        recs_original = set([item_inv[i] for i in top_k_indices])
        
        # Метрики
        hits = len(true_items & recs_original)
        precision = hits / k
        recall = hits / len(true_items) if len(true_items) > 0 else 0
        
        # NDCG@K
        dcg = sum(
            1.0 / np.log2(i + 2) 
            for i, item_idx in enumerate(top_k_indices) 
            if item_inv[item_idx] in true_items
        )
        ideal_len = min(len(true_items), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
    
    return {
        "Precision@K": np.mean(precisions) if precisions else 0,
        "Recall@K": np.mean(recalls) if recalls else 0,
        "NDCG@K": np.mean(ndcgs) if ndcgs else 0,
        "n_evaluated_users": len(precisions)
    }


# ==========================================================
# 5. ГЛАВНАЯ ФУНКЦИЯ - ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ==========================================================

def main():
    """
    Полный пайплайн: загрузка -> подготовка -> обучение -> оценка
    """
    # Пути по умолчанию
    data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    
    print("=" * 60)
    print("🚀 LightGCN Recommendation System")
    print("=" * 60)
    
    # Загрузка данных
    print("\n📥 Loading data...")
    ratings_path = data_dir / "ratings_clean.csv"
    movies_path = data_dir / "movies_clean.csv"
    
    if not ratings_path.exists():
        print(f"⚠️  {ratings_path} not found, trying ratings.csv")
        ratings_path = data_dir / "ratings.csv"
    
    if not movies_path.exists():
        print(f"⚠️  {movies_path} not found, trying movies.csv")
        movies_path = data_dir / "movies.csv"
    
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    print(f"✓ Loaded {len(ratings)} ratings, {len(movies)} movies")
    
    # Для LightGCN используем implicit feedback (просто взаимодействия)
    # Если есть рейтинги, можно отфильтровать: rating >= 4.0
    if 'rating' in ratings.columns:
        ratings_filtered = ratings[ratings['rating'] >= 4.0][['userId', 'movieId']].drop_duplicates()
        print(f"✓ Filtered to {len(ratings_filtered)} interactions (rating >= 4.0)")
    else:
        ratings_filtered = ratings[['userId', 'movieId']].drop_duplicates()
        print(f"✓ Using {len(ratings_filtered)} unique interactions")
    
    # Подготовка данных
    print("\n🔧 Preparing data...")
    edge_index, n_users, n_items, user_inv, item_inv, gt_dict, user_map, item_map = \
        prepare_lightgcn_data(ratings_filtered, min_user_interactions=2)
    
    # Инициализация модели
    print("\n🏗️  Building LightGCN model...")
    model = LightGCN(
        n_users=n_users,
        n_items=n_items,
        emb_dim=64,      # размер эмбеддинга
        n_layers=3       # количество GCN слоёв
    )
    print(f"✓ Model ready: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Обучение
    print("\n🎓 Training model...")
    user_embs, item_embs, losses = train_lightgcn(
        model,
        edge_index,
        gt_dict,
        n_users,
        n_items,
        user_map,
        item_map,
        lr=0.001,
        epochs=20,
        batch_size=1024
    )
    
    # Оценка
    print("\n📊 Evaluating model...")
    metrics_k10 = evaluate_model(user_embs, item_embs, gt_dict, user_inv, item_inv,
                                  user_map, item_map, k=10)
    metrics_k20 = evaluate_model(user_embs, item_embs, gt_dict, user_inv, item_inv,
                                  user_map, item_map, k=20)
    
    print("\n" + "=" * 60)
    print("📈 LightGCN EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n✓ Evaluated {metrics_k10['n_evaluated_users']} users")
    print("\n🎯 Metrics @ K=10:")
    for metric, value in metrics_k10.items():
        if metric != 'n_evaluated_users':
            print(f"  {metric:15s}: {value:.4f}")
    
    print("\n🎯 Metrics @ K=20:")
    for metric, value in metrics_k20.items():
        if metric != 'n_evaluated_users':
            print(f"  {metric:15s}: {value:.4f}")
    
    print("\n" + "=" * 60)
    
    return model, user_embs, item_embs, metrics_k10, metrics_k20


if __name__ == "__main__":
    model, user_embs, item_embs, metrics_k10, metrics_k20 = main()
