"""
AI 輔助開發工具模組
AI-Assisted Development Tools

提供與 GPT、Claude 等 AI 模型集成的工具
"""

import os
from typing import List, Dict, Optional, Callable
import time


class AIAssistant:
    """AI 助手基類"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Args:
            api_key: API 密鑰
            model: 模型名稱
        """
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        raise NotImplementedError


class OpenAIAssistant(AIAssistant):
    """OpenAI GPT 助手"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Args:
            api_key: OpenAI API 密鑰 (如果為 None，從環境變量獲取)
            model: 模型名稱 (推薦: gpt-4o-mini, gpt-4o)
        """
        super().__init__(api_key, model)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self._client = None

        if not self.api_key:
            print("⚠️ 警告: 未設置 OPENAI_API_KEY")
            print("請設置環境變量: export OPENAI_API_KEY='your-key'")
            print("或直接傳入 api_key 參數")

    @property
    def client(self):
        """延遲初始化 OpenAI 客戶端"""
        if self._client is None and self.api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> str:
        """
        使用 GPT 生成文本

        Args:
            prompt: 提示詞
            temperature: 溫度參數 (0-2)
            max_tokens: 最大token數
            **kwargs: 其他參數

        Returns:
            生成的文本

        Example:
            >>> assistant = OpenAIAssistant()
            >>> response = assistant.generate("解釋什麼是自然語言處理")
            >>> print(response)
        """
        if not self.api_key:
            return "❌ 錯誤: 未設置 API 密鑰"

        try:
            if self.client is None:
                return "❌ 錯誤: 無法初始化 OpenAI 客戶端"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            return response.choices[0].message.content.strip()

        except ImportError:
            return "❌ 錯誤: 請安裝 openai 套件 (pip install openai>=1.0)"
        except Exception as e:
            return f"❌ 錯誤: {str(e)}"


class AnthropicAssistant(AIAssistant):
    """Anthropic Claude 助手"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Args:
            api_key: Anthropic API 密鑰
            model: 模型名稱
        """
        super().__init__(api_key, model)
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')

        if not self.api_key:
            print("⚠️ 警告: 未設置 ANTHROPIC_API_KEY")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> str:
        """使用 Claude 生成文本"""
        if not self.api_key:
            return "❌ 錯誤: 未設置 API 密鑰"

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )

            return message.content[0].text

        except ImportError:
            return "❌ 錯誤: 請安裝 anthropic 套件 (pip install anthropic)"
        except Exception as e:
            return f"❌ 錯誤: {str(e)}"


def generate_with_gpt(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> str:
    """
    便捷函數: 使用 GPT 生成文本

    Args:
        prompt: 提示詞
        api_key: API 密鑰
        model: 模型名稱
        temperature: 溫度參數

    Returns:
        生成的文本

    Example:
        >>> response = generate_with_gpt("寫一個情感分析的測試用例")
        >>> print(response)
    """
    assistant = OpenAIAssistant(api_key=api_key, model=model)
    return assistant.generate(prompt, temperature=temperature)


def review_code_with_ai(
    code: str,
    language: str = "python",
    api_key: Optional[str] = None,
) -> str:
    """
    使用 AI 審查代碼

    Args:
        code: 代碼字符串
        language: 編程語言
        api_key: API 密鑰

    Returns:
        審查建議

    Example:
        >>> code = '''
        ... def train(model, data):
        ...     for x, y in data:
        ...         loss = model(x, y)
        ...         loss.backward()
        ... '''
        >>> review = review_code_with_ai(code)
        >>> print(review)
    """
    prompt = f"""
請審查以下 {language} 代碼，提供改進建議：

```{language}
{code}
```

請關注：
1. 代碼質量和可讀性
2. 潛在的 bug 或錯誤
3. 性能優化建議
4. 最佳實踐
5. 安全問題
"""

    return generate_with_gpt(prompt, api_key=api_key, temperature=0.3)


def generate_test_cases(
    function_code: str,
    num_cases: int = 5,
    api_key: Optional[str] = None,
) -> str:
    """
    使用 AI 生成測試用例

    Args:
        function_code: 函數代碼
        num_cases: 測試用例數量
        api_key: API 密鑰

    Returns:
        測試代碼

    Example:
        >>> function_code = '''
        ... def tokenize(text):
        ...     return text.lower().split()
        ... '''
        >>> tests = generate_test_cases(function_code)
        >>> print(tests)
    """
    prompt = f"""
請為以下函數生成 {num_cases} 個測試用例（使用 pytest）：

```python
{function_code}
```

生成的測試應該：
1. 測試正常情況
2. 測試邊界情況
3. 測試異常情況
4. 使用清晰的測試名稱
"""

    return generate_with_gpt(prompt, api_key=api_key, temperature=0.5)


def explain_error(
    error_message: str,
    code_context: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    使用 AI 解釋錯誤

    Args:
        error_message: 錯誤信息
        code_context: 相關代碼（可選）
        api_key: API 密鑰

    Returns:
        錯誤解釋和解決方案

    Example:
        >>> error = "RuntimeError: CUDA out of memory"
        >>> explanation = explain_error(error)
        >>> print(explanation)
    """
    prompt = f"""
請解釋以下錯誤並提供解決方案：

錯誤信息:
{error_message}
"""

    if code_context:
        prompt += f"""

相關代碼:
```python
{code_context}
```
"""

    prompt += """

請提供：
1. 錯誤原因解釋
2. 可能的解決方案（按優先級排序）
3. 預防措施
"""

    return generate_with_gpt(prompt, api_key=api_key, temperature=0.3)


def generate_docstring(
    function_code: str,
    style: str = "google",
    api_key: Optional[str] = None,
) -> str:
    """
    使用 AI 生成文檔字符串

    Args:
        function_code: 函數代碼
        style: 文檔風格 ('google', 'numpy', 'sphinx')
        api_key: API 密鑰

    Returns:
        文檔字符串

    Example:
        >>> function_code = '''
        ... def preprocess(text, lowercase=True):
        ...     if lowercase:
        ...         text = text.lower()
        ...     return text.split()
        ... '''
        >>> docstring = generate_docstring(function_code)
        >>> print(docstring)
    """
    prompt = f"""
請為以下函數生成 {style} 風格的文檔字符串（docstring）：

```python
{function_code}
```

文檔字符串應該包含：
1. 函數功能描述
2. 參數說明（類型和用途）
3. 返回值說明
4. 使用示例（如果適用）
5. 注意事項（如果有）
"""

    return generate_with_gpt(prompt, api_key=api_key, temperature=0.3)


def brainstorm_improvements(
    task_description: str,
    current_approach: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    使用 AI 腦暴改進方案

    Args:
        task_description: 任務描述
        current_approach: 當前方法（可選）
        api_key: API 密鑰

    Returns:
        改進建議

    Example:
        >>> task = "構建一個情感分析系統"
        >>> current = "使用 LSTM + GloVe"
        >>> suggestions = brainstorm_improvements(task, current)
        >>> print(suggestions)
    """
    prompt = f"""
任務: {task_description}
"""

    if current_approach:
        prompt += f"\n當前方法: {current_approach}\n"

    prompt += """
請提供改進建議，包括：
1. 模型架構改進
2. 數據處理優化
3. 訓練技巧
4. 評估方法
5. 部署優化
6. 最新的研究方向

請針對每個建議說明優缺點。
"""

    return generate_with_gpt(prompt, api_key=api_key, temperature=0.8)


class PromptTemplate:
    """Prompt 模板類"""

    # 常用 Prompt 模板
    TEMPLATES = {
        'code_review': """
請審查以下 {language} 代碼：

```{language}
{code}
```

關注點: {focus_areas}
""",

        'bug_fix': """
這段代碼有問題：

```{language}
{code}
```

錯誤信息: {error}

請找出問題並提供修復方案。
""",

        'optimization': """
請優化以下代碼的性能：

```{language}
{code}
```

優化目標: {objectives}
""",

        'explanation': """
請解釋以下代碼的工作原理：

```{language}
{code}
```

解釋重點: {focus}
""",

        'data_augmentation': """
為 NLP 任務生成數據增強樣本：

原始文本: {text}
任務類型: {task_type}
增強數量: {num_samples}

生成多樣化的訓練樣本。
""",
    }

    @classmethod
    def get(cls, template_name: str, **kwargs) -> str:
        """獲取並填充模板"""
        if template_name not in cls.TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")

        template = cls.TEMPLATES[template_name]
        return template.format(**kwargs)


# AI 輔助數據增強
def ai_augment_data(
    texts: List[str],
    task_type: str = "sentiment_analysis",
    num_aug_per_text: int = 3,
    api_key: Optional[str] = None,
) -> List[str]:
    """
    使用 AI 進行數據增強

    Args:
        texts: 原始文本列表
        task_type: 任務類型
        num_aug_per_text: 每個文本生成的增強樣本數
        api_key: API 密鑰

    Returns:
        增強後的文本列表

    Example:
        >>> texts = ["This movie is great!", "I love this product"]
        >>> augmented = ai_augment_data(texts, num_aug_per_text=2)
        >>> print(augmented)
    """
    augmented = []

    for text in texts:
        prompt = PromptTemplate.get(
            'data_augmentation',
            text=text,
            task_type=task_type,
            num_samples=num_aug_per_text
        )

        response = generate_with_gpt(prompt, api_key=api_key, temperature=0.9)
        # 簡單解析響應（實際使用時可能需要更複雜的解析）
        aug_texts = [line.strip() for line in response.split('\n') if line.strip()]
        augmented.extend(aug_texts[:num_aug_per_text])

    return augmented


if __name__ == '__main__':
    print("=" * 50)
    print("AI 輔助工具測試")
    print("=" * 50)

    print("\n📝 注意: 需要設置 API 密鑰才能實際使用")
    print("export OPENAI_API_KEY='your-key'")
    print("export ANTHROPIC_API_KEY='your-key'")

    # 測試 Prompt 模板
    print("\n" + "=" * 50)
    print("測試 Prompt 模板")
    print("=" * 50)

    prompt = PromptTemplate.get(
        'code_review',
        language='python',
        code='def train(model, data):\n    pass',
        focus_areas='代碼質量, 性能'
    )
    print(prompt)

    # 測試代碼審查（模擬）
    print("\n" + "=" * 50)
    print("代碼審查功能（需要 API 密鑰）")
    print("=" * 50)

    code = """
def train_model(model, train_loader, epochs):
    for epoch in range(epochs):
        for batch in train_loader:
            x, y = batch
            loss = model(x, y)
            loss.backward()
    """

    print(f"代碼:\n{code}")
    print("\n💡 使用 review_code_with_ai(code) 獲取審查建議")

    print("\n✅ 測試完成！")
    print("\n💡 提示: 這些工具可以大大提高開發效率")
    print("   - 代碼審查")
    print("   - 測試生成")
    print("   - 錯誤解釋")
    print("   - 文檔生成")
    print("   - 數據增強")
