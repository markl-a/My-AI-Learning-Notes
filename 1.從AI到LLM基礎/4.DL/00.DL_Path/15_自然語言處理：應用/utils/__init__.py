"""
NLP 應用實用工具包
Utilities for NLP Applications

提供數據預處理、訓練輔助、可視化、AI 輔助等功能
"""

__version__ = '2.0.0'
__author__ = 'NLP Applications Team'

# 數據預處理
from .preprocessing import (
    TextCleaner,
    Vocabulary,
    clean_text,
    tokenize_english,
    tokenize_chinese,
    pad_sequences,
    create_ngrams,
    augment_text,
    create_vocab,
)

# 訓練輔助
from .training import (
    Trainer,
    EarlyStopping,
    ModelCheckpoint,
    MetricsTracker,
    LearningRateScheduler,
    compute_accuracy,
    compute_f1_score,
    set_seed,
)

# 可視化
from .visualization import (
    plot_training_curve,
    plot_confusion_matrix,
    plot_classification_report,
    visualize_attention,
    plot_embeddings,
    plot_word_cloud,
    plot_learning_rate,
    plot_prediction_distribution,
)

# AI 輔助工具
from .ai_tools import (
    OpenAIAssistant,
    AnthropicAssistant,
    generate_with_gpt,
    review_code_with_ai,
    generate_test_cases,
    explain_error,
    generate_docstring,
    brainstorm_improvements,
    ai_augment_data,
    PromptTemplate,
)

__all__ = [
    # 預處理
    'TextCleaner',
    'Vocabulary',
    'clean_text',
    'tokenize_english',
    'tokenize_chinese',
    'pad_sequences',
    'create_ngrams',
    'augment_text',
    'create_vocab',

    # 訓練
    'Trainer',
    'EarlyStopping',
    'ModelCheckpoint',
    'MetricsTracker',
    'LearningRateScheduler',
    'compute_accuracy',
    'compute_f1_score',
    'set_seed',

    # 可視化
    'plot_training_curve',
    'plot_confusion_matrix',
    'plot_classification_report',
    'visualize_attention',
    'plot_embeddings',
    'plot_word_cloud',
    'plot_learning_rate',
    'plot_prediction_distribution',

    # AI 輔助
    'OpenAIAssistant',
    'AnthropicAssistant',
    'generate_with_gpt',
    'review_code_with_ai',
    'generate_test_cases',
    'explain_error',
    'generate_docstring',
    'brainstorm_improvements',
    'ai_augment_data',
    'PromptTemplate',
]


def print_utils_info():
    """打印工具包信息"""
    print("=" * 60)
    print(f"NLP 應用實用工具包 v{__version__}")
    print("=" * 60)
    print("\n📦 可用模組:")
    print("  1. preprocessing - 數據預處理工具")
    print("  2. training      - 訓練輔助工具")
    print("  3. visualization - 可視化工具")
    print("  4. ai_tools      - AI 輔助開發工具")
    print("\n💡 快速開始:")
    print("  from utils import clean_text, Trainer, plot_training_curve")
    print("\n📚 查看文檔:")
    print("  help(utils.preprocessing)")
    print("  help(utils.training)")
    print("=" * 60)


# 版本檢查
def check_dependencies():
    """檢查依賴包"""
    import sys

    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
    }

    optional = {
        'jieba': '中文分詞 (jieba)',
        'transformers': 'Hugging Face Transformers',
        'openai': 'OpenAI API',
        'anthropic': 'Anthropic API',
        'seaborn': 'Seaborn (可視化增強)',
        'wordcloud': 'WordCloud (詞雲)',
    }

    print("🔍 檢查依賴包...")
    print("\n必需依賴:")
    for pkg, name in required.items():
        try:
            __import__(pkg)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - 請安裝: pip install {pkg}")

    print("\n可選依賴:")
    for pkg, name in optional.items():
        try:
            __import__(pkg)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️ {name} - 可選安裝: pip install {pkg}")


if __name__ == '__main__':
    print_utils_info()
    print("\n")
    check_dependencies()
