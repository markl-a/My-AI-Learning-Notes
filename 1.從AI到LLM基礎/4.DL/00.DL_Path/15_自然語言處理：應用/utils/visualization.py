"""
可視化工具模組
Visualization Utilities for NLP

提供訓練曲線、混淆矩陣、注意力可視化等功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch


# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_curve(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
):
    """
    繪製訓練曲線

    Args:
        history: 訓練歷史 {'train_loss': [...], 'val_loss': [...], ...}
        metrics: 要繪製的指標列表
        figsize: 圖形大小
        save_path: 保存路徑

    Example:
        >>> history = {
        ...     'train_loss': [0.5, 0.4, 0.3],
        ...     'val_loss': [0.6, 0.5, 0.4],
        ...     'train_acc': [0.7, 0.8, 0.9],
        ...     'val_acc': [0.6, 0.7, 0.8],
        ... }
        >>> plot_training_curve(history)
    """
    if metrics is None:
        # 自動檢測指標
        metrics_set = set()
        for key in history.keys():
            if key.startswith('train_'):
                metric_name = key.replace('train_', '')
                metrics_set.add(metric_name)
        metrics = sorted(list(metrics_set))

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        if train_key in history:
            ax.plot(history[train_key], label=f'Train {metric}', marker='o')

        if val_key in history:
            ax.plot(history[val_key], label=f'Val {metric}', marker='s')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 圖形已保存到 {save_path}")

    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
):
    """
    繪製混淆矩陣

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        labels: 類別標籤列表
        normalize: 是否歸一化
        figsize: 圖形大小
        save_path: 保存路徑

    Example:
        >>> y_true = np.array([0, 1, 0, 1, 0])
        >>> y_pred = np.array([0, 1, 1, 1, 0])
        >>> plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive'])
    """
    from sklearn.metrics import confusion_matrix

    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    # 繪製
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 混淆矩陣已保存到 {save_path}")

    plt.show()


def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    繪製分類報告熱力圖

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        labels: 類別標籤列表
        figsize: 圖形大小
        save_path: 保存路徑
    """
    from sklearn.metrics import classification_report
    import pandas as pd

    # 生成報告
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

    # 轉換為 DataFrame
    df = pd.DataFrame(report).transpose()
    df = df[['precision', 'recall', 'f1-score', 'support']]

    # 繪製
    plt.figure(figsize=figsize)
    sns.heatmap(
        df.iloc[:-3, :-1],  # 排除 accuracy, macro avg, weighted avg 和 support
        annot=True,
        fmt='.2f',
        cmap='YlGnBu',
        cbar_kws={'label': 'Score'}
    )

    plt.title('Classification Report')
    plt.ylabel('Class')
    plt.xlabel('Metrics')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 分類報告已保存到 {save_path}")

    plt.show()


def visualize_attention(
    attention_weights: np.ndarray,
    input_tokens: List[str],
    output_tokens: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
):
    """
    可視化注意力權重

    Args:
        attention_weights: 注意力權重矩陣 shape: (output_len, input_len)
        input_tokens: 輸入詞元列表
        output_tokens: 輸出詞元列表 (可選)
        figsize: 圖形大小
        save_path: 保存路徑

    Example:
        >>> attention = np.random.rand(5, 7)
        >>> input_tokens = ['I', 'love', 'natural', 'language', 'processing', '.']
        >>> output_tokens = ['我', '愛', '自然', '語言', '處理']
        >>> visualize_attention(attention, input_tokens, output_tokens)
    """
    if output_tokens is None:
        output_tokens = input_tokens

    plt.figure(figsize=figsize)
    sns.heatmap(
        attention_weights,
        xticklabels=input_tokens,
        yticklabels=output_tokens,
        cmap='viridis',
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Attention Weight'}
    )

    plt.title('Attention Weights Visualization')
    plt.xlabel('Input Tokens')
    plt.ylabel('Output Tokens')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 注意力圖已保存到 {save_path}")

    plt.show()


def plot_embeddings(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    method: str = 'tsne',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
):
    """
    可視化詞向量 (降維到2D)

    Args:
        embeddings: 詞向量矩陣 shape: (vocab_size, embedding_dim)
        labels: 詞元標籤列表
        method: 降維方法 ('tsne', 'pca', 'umap')
        figsize: 圖形大小
        save_path: 保存路徑

    Example:
        >>> embeddings = np.random.randn(100, 50)
        >>> labels = [f'word_{i}' for i in range(100)]
        >>> plot_embeddings(embeddings, labels, method='tsne')
    """
    # 降維
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        except ImportError:
            print("UMAP not installed. Falling back to t-SNE.")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    embeddings_2d = reducer.fit_transform(embeddings)

    # 繪製
    plt.figure(figsize=figsize)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

    # 添加標籤
    if labels:
        for i, label in enumerate(labels):
            plt.annotate(
                label,
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=8,
                alpha=0.7
            )

    plt.title(f'Word Embeddings Visualization ({method.upper()})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 詞向量可視化已保存到 {save_path}")

    plt.show()


def plot_word_cloud(
    text: str,
    max_words: int = 100,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
):
    """
    繪製詞雲

    Args:
        text: 文本字符串
        max_words: 最大詞數
        figsize: 圖形大小
        save_path: 保存路徑

    Example:
        >>> text = "natural language processing machine learning deep learning"
        >>> plot_word_cloud(text)
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("wordcloud not installed. Install with: pip install wordcloud")
        return

    # 生成詞雲
    wordcloud = WordCloud(
        width=800,
        height=400,
        max_words=max_words,
        background_color='white',
        colormap='viridis'
    ).generate(text)

    # 繪製
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud', fontsize=16, pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 詞雲已保存到 {save_path}")

    plt.show()


def plot_learning_rate(
    learning_rates: List[float],
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
):
    """
    繪製學習率變化曲線

    Args:
        learning_rates: 學習率列表
        figsize: 圖形大小
        save_path: 保存路徑
    """
    plt.figure(figsize=figsize)
    plt.plot(learning_rates, marker='o')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 學習率曲線已保存到 {save_path}")

    plt.show()


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
):
    """
    繪製預測概率分布

    Args:
        y_true: 真實標籤
        y_pred_proba: 預測概率 shape: (n_samples, n_classes)
        class_names: 類別名稱列表
        figsize: 圖形大小
        save_path: 保存路徑
    """
    n_classes = y_pred_proba.shape[1]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 左圖: 預測概率分布
    for i in range(n_classes):
        axes[0].hist(
            y_pred_proba[:, i],
            bins=50,
            alpha=0.5,
            label=class_names[i]
        )
    axes[0].set_xlabel('Prediction Probability')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Prediction Probability Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右圖: 預測置信度 vs 正確性
    y_pred = np.argmax(y_pred_proba, axis=1)
    max_proba = np.max(y_pred_proba, axis=1)
    correct = (y_pred == y_true)

    axes[1].scatter(
        max_proba[correct],
        np.ones(correct.sum()),
        alpha=0.5,
        label='Correct',
        c='green'
    )
    axes[1].scatter(
        max_proba[~correct],
        np.zeros((~correct).sum()),
        alpha=0.5,
        label='Incorrect',
        c='red'
    )
    axes[1].set_xlabel('Max Prediction Probability')
    axes[1].set_ylabel('Correctness')
    axes[1].set_title('Confidence vs Correctness')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 預測分布圖已保存到 {save_path}")

    plt.show()


if __name__ == '__main__':
    print("=" * 50)
    print("可視化工具測試")
    print("=" * 50)

    # 測試訓練曲線
    print("\n1. 測試訓練曲線...")
    history = {
        'train_loss': [0.8, 0.6, 0.4, 0.3, 0.2],
        'val_loss': [0.9, 0.7, 0.5, 0.4, 0.35],
        'train_acc': [0.6, 0.7, 0.8, 0.85, 0.9],
        'val_acc': [0.55, 0.65, 0.75, 0.8, 0.82],
    }
    plot_training_curve(history)

    # 測試混淆矩陣
    print("\n2. 測試混淆矩陣...")
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive'])

    # 測試注意力可視化
    print("\n3. 測試注意力可視化...")
    attention = np.random.rand(5, 7)
    input_tokens = ['I', 'love', 'NLP', '!']
    output_tokens = ['NLP', 'is', 'great']
    # visualize_attention(attention[:3, :4], input_tokens, output_tokens)

    print("\n✅ 所有測試完成！")
