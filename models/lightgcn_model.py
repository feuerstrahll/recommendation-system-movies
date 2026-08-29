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
    
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    
    # Делаем граф неориентированным (двусторонние рёбра)
    edge_index_undirected = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    edge_index_undirected = coalesce(
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

        # Cache for the normalized sparse adjacency matrix (see _get_adj):
        # it depends only on graph structure, which is fixed for the whole
        # training run, so it should be built once rather than on every
        # forward call.
        self._adj_cache = None
        self._adj_cache_edge_index = None

    def _get_adj(self, edge_index):
        """
        Normalized sparse adjacency (D^(-1/2) @ A @ D^(-1/2)), cached by
        edge_index identity. edge_index never changes during training (only
        the embeddings do), so rebuilding deg/deg_inv_sqrt/the sparse tensor
        on every forward call — every batch, every epoch — was pure repeated
        work with no effect on the result.
        """
        if self._adj_cache is not None and self._adj_cache_edge_index is edge_index:
            return self._adj_cache

        n_nodes = self.n_users + self.n_items
        row, col = edge_index
        deg = degree(col, n_nodes, dtype=self.user_emb.weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(
            edge_index,
            edge_weight,
            (n_nodes, n_nodes)
        ).coalesce().to(device)

        self._adj_cache = adj
        self._adj_cache_edge_index = edge_index
        return adj

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

        adj = self._get_adj(edge_index)

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
                   lr=0.001, epochs=20, batch_size=1024, neg_samples=1, seed=42):
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
        seed: seed для локального RNG (positive/negative sampling), для воспроизводимости

    Returns:
        user_embs, item_embs: обученные эмбеддинги
    """
    rng = np.random.RandomState(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    edge_index = edge_index.to(device)

    # Собираем валидные пользователи с positive items (сразу в виде эмбеддинг-индексов)
    all_users_mapped = []
    all_pos = []          # оригинальные item id, нужны для rng.choice ниже
    all_pos_idx_sets = [] # те же позитивы, но как set индексов эмбеддингов — для фильтрации негативов

    for u_original_id in gt_dict.keys():
        if u_original_id in user_map and len(gt_dict[u_original_id]) > 0:
            pos_ids = [i for i in gt_dict[u_original_id] if i in item_map]
            if not pos_ids:
                continue
            all_users_mapped.append(u_original_id)
            all_pos.append(pos_ids)
            all_pos_idx_sets.append({item_map[i] for i in pos_ids})

    model.train()
    losses_history = []

    # Training exposure per epoch is equalized against LightFM: LightFM's
    # "one epoch" is one WARP pass over EVERY positive interaction, so one
    # (user, pos) triple per user per epoch here (regardless of how many
    # positives that user has) gave LightGCN far fewer gradient-relevant
    # examples per epoch at the same epochs= value — the "10 epochs vs 10
    # epochs" comparison in the results table was misleading. Instead,
    # sample neg_samples negatives for EVERY positive interaction of every
    # user each epoch, so the number of training examples scales with the
    # actual interaction count, same as LightFM.
    u_idx_list = []
    pos_idx_list = []
    pos_idx_set_per_example = []
    for i, u_original_id in enumerate(all_users_mapped):
        u_internal = user_map[u_original_id]
        for pos_item in all_pos[i]:
            u_idx_list.append(u_internal)
            pos_idx_list.append(item_map[pos_item])
            pos_idx_set_per_example.append(all_pos_idx_sets[i])

    for epoch in range(epochs):
        neg_idx_list = []
        for _ in range(neg_samples):
            for pos_idx_set in pos_idx_set_per_example:
                while True:
                    candidate = rng.randint(0, n_items)
                    if candidate not in pos_idx_set:
                        neg_idx_list.append(candidate)
                        break

        u_idx = torch.tensor(u_idx_list * neg_samples, device=device)
        pos_idx = torch.tensor(pos_idx_list * neg_samples, device=device)
        neg_idx = torch.tensor(neg_idx_list, device=device)

        # optimizer.step() PER BATCH, not per epoch (a step-per-epoch design
        # gave far too few real gradient updates to move the BPR loss off
        # its random-init value of ln(2) ~= 0.693). Each batch recomputes
        # the graph forward pass since parameters (and therefore user_embs/
        # item_embs) change after every optimizer.step().
        total_loss = 0.0
        n_batches = 0
        perm = torch.randperm(len(u_idx), device=device)
        pbar = tqdm(
            range(0, len(u_idx), batch_size),
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False
        )
        for start in pbar:
            end = min(start + batch_size, len(u_idx))
            idx = perm[start:end]

            user_embs, item_embs = model(edge_index)

            u_emb = user_embs[u_idx[idx]]
            pos_emb = item_embs[pos_idx[idx]]
            neg_emb = item_embs[neg_idx[idx]]

            pos_scores = (u_emb * pos_emb).sum(dim=1)
            neg_scores = (u_emb * neg_emb).sum(dim=1)

            batch_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            n_batches += 1
            if n_batches % 20 == 0:
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
                   user_map, item_map, k=10, seen_items=None):
    """
    Оценка модели по метрикам Precision@K, Recall@K, NDCG@K.

    Args:
        user_embs, item_embs: эмбеддинги из модели
        gt_dict: ground-truth {user_id: [item_ids]} — должен быть held-out (test), не train
        user_inv, item_inv: обратные маппинги
        user_map, item_map: прямые маппинги
        k: количество рекомендаций
        seen_items: optional {user_id: [item_ids]} — train-взаимодействия, исключаются
            из top-K отдельно от gt_dict, чтобы модель не могла "порекомендовать"
            фильм, который пользователь уже видел при обучении

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

        # Исключаем из ранжирования фильмы, уже виденные в train (не должны попасть в рекомендации)
        exclude_items = set(seen_items.get(u_original_id, [])) if seen_items else set()
        known_items_mapped = set()
        for item_id in exclude_items:
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
    Полный пайплайн: загрузка -> train/test split -> подготовка -> обучение -> оценка на held-out данных
    """
    # Пути по умолчанию
    data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"

    print("=" * 60)
    print("🚀 LightGCN Recommendation System")
    print("=" * 60)

    # Загрузка данных. ratings.csv — канонический файл проекта, ключ movie_id = TMDB id.
    # ratings_clean.csv используется только как запасной вариант (raw MovieLens id).
    print("\n📥 Loading data...")
    ratings_path = data_dir / "ratings.csv"
    id_cols = {"user": "user_id", "movie": "movie_id"}

    if not ratings_path.exists():
        print(f"⚠️  {ratings_path} not found, trying ratings_clean.csv")
        ratings_path = data_dir / "ratings_clean.csv"
        id_cols = {"user": "userId", "movie": "movieId"}

    ratings = pd.read_csv(ratings_path)
    print(f"✓ Loaded {len(ratings)} ratings from {ratings_path.name}")

    # Нормализуем к userId/movieId, как ожидает prepare_lightgcn_data
    ratings = ratings.rename(columns={id_cols["user"]: "userId", id_cols["movie"]: "movieId"})

    # Implicit feedback: оставляем только положительные взаимодействия (rating >= 4.0)
    if 'rating' in ratings.columns:
        ratings_filtered = ratings[ratings['rating'] >= 4.0][['userId', 'movieId']].drop_duplicates()
        print(f"✓ Filtered to {len(ratings_filtered)} interactions (rating >= 4.0)")
    else:
        ratings_filtered = ratings[['userId', 'movieId']].drop_duplicates()
        print(f"✓ Using {len(ratings_filtered)} unique interactions")

    # Держим только пользователей с достаточным числом взаимодействий,
    # чтобы после разбиения у каждого было что-то и в train, и в test
    user_counts = ratings_filtered['userId'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    ratings_filtered = ratings_filtered[ratings_filtered['userId'].isin(valid_users)].copy()
    print(f"✓ Kept {len(ratings_filtered)} interactions from {ratings_filtered['userId'].nunique()} users with >= 5 interactions")

    # 80/20 train/test split по каждому пользователю (не по всей таблице сразу),
    # чтобы каждый пользователь остался и в train, и в test
    print("\n✂️  Splitting train/test (80/20 per user)...")
    train_rows, test_rows = [], []
    rng = np.random.RandomState(42)
    for _, udf in ratings_filtered.groupby('userId'):
        shuffled = udf.sample(frac=1, random_state=rng)
        n_train = max(1, int(len(shuffled) * 0.8))
        train_rows.append(shuffled.iloc[:n_train])
        test_rows.append(shuffled.iloc[n_train:])
    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)
    print(f"✓ Train: {len(train_df)} interactions | Test: {len(test_df)} interactions")

    # Подготовка графа СТРОГО на train, чтобы тестовые взаимодействия не просачивались в обучение
    print("\n🔧 Preparing data...")
    edge_index, n_users, n_items, user_inv, item_inv, train_gt_dict, user_map, item_map = \
        prepare_lightgcn_data(train_df, min_user_interactions=0)

    # Ground truth для оценки — это TEST-взаимодействия (held-out), а не train.
    # Фильмы, которых не было в train, не имеют item embedding и физически не могут
    # быть рекомендованы (cold-start item) — такие test-взаимодействия исключаем явно,
    # иначе они искусственно занижают Recall/NDCG независимо от качества модели.
    test_df_known_items = test_df[test_df['movieId'].isin(item_map.keys())]
    n_dropped = len(test_df) - len(test_df_known_items)
    if n_dropped > 0:
        print(f"⚠️  Dropped {n_dropped} test interactions on items unseen in train (cold-start items)")
    test_gt_dict = test_df_known_items.groupby('userId')['movieId'].apply(list).to_dict()

    # Инициализация модели
    print("\n🏗️  Building LightGCN model...")
    model = LightGCN(
        n_users=n_users,
        n_items=n_items,
        emb_dim=64,      # размер эмбеддинга
        n_layers=3       # количество GCN слоёв
    )
    print(f"✓ Model ready: {sum(p.numel() for p in model.parameters())} parameters")

    # Обучение — только на train_gt_dict / edge_index из train
    print("\n🎓 Training model...")
    user_embs, item_embs, losses = train_lightgcn(
        model,
        edge_index,
        train_gt_dict,
        n_users,
        n_items,
        user_map,
        item_map,
        lr=0.001,
        epochs=20,
        batch_size=1024,
        seed=42
    )

    # Оценка на held-out test-взаимодействиях; seen_items=train_gt_dict исключает
    # из top-K фильмы, уже виденные пользователем в train.
    print("\n📊 Evaluating model on held-out test set...")
    metrics_k10 = evaluate_model(user_embs, item_embs, test_gt_dict, user_inv, item_inv,
                                  user_map, item_map, k=10, seen_items=train_gt_dict)
    metrics_k20 = evaluate_model(user_embs, item_embs, test_gt_dict, user_inv, item_inv,
                                  user_map, item_map, k=20, seen_items=train_gt_dict)

    print("\n" + "=" * 60)
    print("📈 LightGCN EVALUATION RESULTS (held-out test set)")
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
