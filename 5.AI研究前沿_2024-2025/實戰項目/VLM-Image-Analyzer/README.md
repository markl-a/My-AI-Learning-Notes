# VLM 圖像分析系統實戰專案

> 使用視覺語言模型 (Vision-Language Model) 建構智能圖像分析應用

## 📋 專案概述

本專案實作一個基於 VLM 的多功能圖像分析系統，支援：
- 圖像描述與標籤生成
- 視覺問答 (VQA)
- 文件/發票 OCR 分析
- 圖表數據提取
- 多圖像比較分析

## 🎯 學習目標

完成本專案後，你將掌握：
- VLM API 整合 (GPT-4V, Claude Vision, Gemini)
- 圖像預處理與最佳化
- 多模態提示工程
- 批量處理與效能優化
- 生產環境部署

## 🏗️ 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                    VLM 圖像分析系統                          │
├─────────────────────────────────────────────────────────────┤
│  輸入層                                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ 單張圖像 │  │ 多張圖像 │  │ PDF文件 │  │ 視頻幀  │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       └────────────┴────────────┴────────────┘             │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              預處理模組                              │   │
│  │  • 圖像壓縮/調整大小  • 格式轉換  • Base64 編碼     │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              VLM 引擎                                │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │   │
│  │  │ GPT-4V  │  │ Claude  │  │ Gemini  │  ← 可切換   │   │
│  │  └─────────┘  └─────────┘  └─────────┘             │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              後處理模組                              │   │
│  │  • 結構化輸出  • 資料驗證  • 結果快取               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 📦 專案結構

```
VLM-Image-Analyzer/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── vlm_client.py      # VLM 客戶端抽象
│   │   ├── image_processor.py  # 圖像預處理
│   │   └── prompt_templates.py # 提示模板
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── image_captioner.py  # 圖像描述
│   │   ├── document_analyzer.py # 文件分析
│   │   ├── chart_extractor.py  # 圖表數據提取
│   │   └── vqa_engine.py       # 視覺問答
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # FastAPI 路由
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── test_vlm_client.py
│   ├── test_analyzers.py
│   └── sample_images/
├── examples/
│   ├── 01_basic_usage.py
│   ├── 02_document_ocr.py
│   ├── 03_chart_analysis.py
│   └── 04_batch_processing.py
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

## 🚀 快速開始

### 1. 環境設定

```bash
# 克隆專案
git clone <repository-url>
cd VLM-Image-Analyzer

# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
cp .env.example .env
# 編輯 .env 填入 API 金鑰
```

### 2. 基本使用

```python
from src.core.vlm_client import VLMClient
from src.analyzers.image_captioner import ImageCaptioner

# 初始化客戶端
client = VLMClient(provider="openai")  # 或 "anthropic", "google"

# 圖像描述
captioner = ImageCaptioner(client)
result = captioner.caption("path/to/image.jpg")
print(result.description)
print(result.tags)
```

## 💻 核心程式碼

### VLM 客戶端抽象

```python
# src/core/vlm_client.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from pathlib import Path
import base64
from PIL import Image
import io

class VLMProvider(ABC):
    """VLM 提供者抽象基類"""

    @abstractmethod
    def analyze_image(
        self,
        image: Union[str, bytes, Path],
        prompt: str,
        max_tokens: int = 1024
    ) -> str:
        """分析單張圖像"""
        pass

    @abstractmethod
    def analyze_multiple_images(
        self,
        images: List[Union[str, bytes, Path]],
        prompt: str,
        max_tokens: int = 2048
    ) -> str:
        """分析多張圖像"""
        pass


class OpenAIVLM(VLMProvider):
    """OpenAI GPT-4V 實作"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _encode_image(self, image: Union[str, bytes, Path]) -> str:
        """將圖像編碼為 base64"""
        if isinstance(image, bytes):
            return base64.b64encode(image).decode('utf-8')
        elif isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        raise ValueError("不支援的圖像格式")

    def analyze_image(
        self,
        image: Union[str, bytes, Path],
        prompt: str,
        max_tokens: int = 1024
    ) -> str:
        base64_image = self._encode_image(image)

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
        )

        return response.choices[0].message.content

    def analyze_multiple_images(
        self,
        images: List[Union[str, bytes, Path]],
        prompt: str,
        max_tokens: int = 2048
    ) -> str:
        content = [{"type": "text", "text": prompt}]

        for i, image in enumerate(images):
            base64_image = self._encode_image(image)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            })

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}]
        )

        return response.choices[0].message.content


class AnthropicVLM(VLMProvider):
    """Anthropic Claude Vision 實作"""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _encode_image(self, image: Union[str, bytes, Path]) -> tuple:
        """編碼圖像並檢測類型"""
        if isinstance(image, bytes):
            data = image
        else:
            with open(image, "rb") as f:
                data = f.read()

        # 檢測圖像類型
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            media_type = "image/png"
        elif data[:2] == b'\xff\xd8':
            media_type = "image/jpeg"
        else:
            media_type = "image/jpeg"  # 預設

        return base64.b64encode(data).decode('utf-8'), media_type

    def analyze_image(
        self,
        image: Union[str, bytes, Path],
        prompt: str,
        max_tokens: int = 1024
    ) -> str:
        base64_data, media_type = self._encode_image(image)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )

        return response.content[0].text

    def analyze_multiple_images(
        self,
        images: List[Union[str, bytes, Path]],
        prompt: str,
        max_tokens: int = 2048
    ) -> str:
        content = []

        for image in images:
            base64_data, media_type = self._encode_image(image)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data
                }
            })

        content.append({"type": "text", "text": prompt})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}]
        )

        return response.content[0].text


class VLMClient:
    """VLM 客戶端工廠"""

    PROVIDERS = {
        "openai": OpenAIVLM,
        "anthropic": AnthropicVLM,
    }

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        **kwargs
    ):
        import os

        if provider not in self.PROVIDERS:
            raise ValueError(f"不支援的提供者: {provider}")

        if api_key is None:
            env_key = f"{provider.upper()}_API_KEY"
            api_key = os.getenv(env_key)

        self.provider = self.PROVIDERS[provider](api_key, **kwargs)

    def analyze(
        self,
        image: Union[str, bytes, Path, List],
        prompt: str,
        **kwargs
    ) -> str:
        if isinstance(image, list):
            return self.provider.analyze_multiple_images(image, prompt, **kwargs)
        return self.provider.analyze_image(image, prompt, **kwargs)
```

### 圖像預處理

```python
# src/core/image_processor.py
from PIL import Image
from pathlib import Path
from typing import Union, Tuple
import io

class ImageProcessor:
    """圖像預處理器"""

    def __init__(
        self,
        max_size: Tuple[int, int] = (2048, 2048),
        quality: int = 85,
        format: str = "JPEG"
    ):
        self.max_size = max_size
        self.quality = quality
        self.format = format

    def process(
        self,
        image: Union[str, Path, bytes, Image.Image]
    ) -> bytes:
        """處理圖像：調整大小、壓縮、格式轉換"""
        # 載入圖像
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError("不支援的圖像類型")

        # 轉換為 RGB（處理 RGBA）
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        # 調整大小（保持比例）
        img.thumbnail(self.max_size, Image.Resampling.LANCZOS)

        # 輸出為 bytes
        buffer = io.BytesIO()
        img.save(buffer, format=self.format, quality=self.quality)
        return buffer.getvalue()

    def get_image_info(
        self,
        image: Union[str, Path, bytes]
    ) -> dict:
        """獲取圖像資訊"""
        if isinstance(image, (str, Path)):
            img = Image.open(image)
            file_size = Path(image).stat().st_size
        else:
            img = Image.open(io.BytesIO(image))
            file_size = len(image)

        return {
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": img.format,
            "file_size_kb": file_size / 1024
        }
```

### 文件分析器

```python
# src/analyzers/document_analyzer.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

from ..core.vlm_client import VLMClient
from ..core.prompt_templates import DOCUMENT_ANALYSIS_PROMPT

@dataclass
class DocumentAnalysisResult:
    """文件分析結果"""
    document_type: str
    extracted_text: str
    structured_data: Dict
    confidence: float
    language: str

class DocumentAnalyzer:
    """文件/發票分析器"""

    SUPPORTED_TYPES = [
        "invoice", "receipt", "contract",
        "id_card", "business_card", "form"
    ]

    def __init__(self, vlm_client: VLMClient):
        self.client = vlm_client

    def analyze(
        self,
        image_path: str,
        document_type: Optional[str] = None
    ) -> DocumentAnalysisResult:
        """分析文件圖像"""
        prompt = self._build_prompt(document_type)

        response = self.client.analyze(image_path, prompt)

        # 解析結構化輸出
        result = self._parse_response(response)

        return DocumentAnalysisResult(**result)

    def _build_prompt(self, document_type: Optional[str]) -> str:
        """建構分析提示"""
        if document_type:
            type_specific = f"這是一張{document_type}的圖像。"
        else:
            type_specific = "請先判斷這是什麼類型的文件。"

        return f"""
        {type_specific}

        請仔細分析這張文件圖像，並提取以下資訊：

        1. 文件類型（如：發票、收據、合約、表格等）
        2. 完整的文字內容（OCR）
        3. 結構化資料（JSON 格式）：
           - 對於發票/收據：日期、金額、供應商、項目明細
           - 對於合約：當事人、日期、主要條款
           - 對於表格：表格結構和內容

        請以以下 JSON 格式回覆：
        {{
            "document_type": "類型",
            "extracted_text": "完整文字內容",
            "structured_data": {{...}},
            "confidence": 0.95,
            "language": "zh-TW"
        }}

        只返回 JSON，不要其他內容。
        """

    def _parse_response(self, response: str) -> Dict:
        """解析 VLM 回應"""
        # 嘗試提取 JSON
        try:
            # 移除可能的 markdown 標記
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {
                "document_type": "unknown",
                "extracted_text": response,
                "structured_data": {},
                "confidence": 0.5,
                "language": "unknown"
            }

    def batch_analyze(
        self,
        image_paths: List[str],
        document_type: Optional[str] = None
    ) -> List[DocumentAnalysisResult]:
        """批量分析文件"""
        results = []
        for path in image_paths:
            result = self.analyze(path, document_type)
            results.append(result)
        return results
```

### 圖表數據提取

```python
# src/analyzers/chart_extractor.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

from ..core.vlm_client import VLMClient

@dataclass
class ChartData:
    """圖表數據"""
    chart_type: str
    title: Optional[str]
    x_axis: Optional[str]
    y_axis: Optional[str]
    data_points: List[Dict]
    summary: str

class ChartExtractor:
    """圖表數據提取器"""

    CHART_TYPES = [
        "bar", "line", "pie", "scatter",
        "area", "histogram", "box", "heatmap"
    ]

    def __init__(self, vlm_client: VLMClient):
        self.client = vlm_client

    def extract(self, image_path: str) -> ChartData:
        """從圖表圖像提取數據"""
        prompt = """
        分析這張圖表圖像，提取所有可見的數據。

        請提供：
        1. 圖表類型（如：折線圖、長條圖、圓餅圖等）
        2. 標題（如果有）
        3. X 軸和 Y 軸標籤
        4. 所有可讀取的數據點
        5. 圖表的主要發現或趨勢總結

        以 JSON 格式回覆：
        {
            "chart_type": "類型",
            "title": "標題",
            "x_axis": "X軸標籤",
            "y_axis": "Y軸標籤",
            "data_points": [
                {"category": "類別1", "value": 100},
                {"category": "類別2", "value": 200}
            ],
            "summary": "主要發現總結"
        }

        只返回 JSON。
        """

        response = self.client.analyze(image_path, prompt)
        data = self._parse_response(response)

        return ChartData(**data)

    def compare_charts(
        self,
        image_paths: List[str]
    ) -> Dict:
        """比較多張圖表"""
        prompt = """
        比較這些圖表，分析它們之間的關係和差異。

        對於每張圖表，提取關鍵數據點。
        然後進行比較分析：
        1. 相同之處
        2. 差異之處
        3. 趨勢對比
        4. 整體結論

        以 JSON 格式回覆。
        """

        response = self.client.analyze(image_paths, prompt)
        return json.loads(response)

    def _parse_response(self, response: str) -> Dict:
        """解析回應"""
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            return json.loads(response.strip())
        except:
            return {
                "chart_type": "unknown",
                "title": None,
                "x_axis": None,
                "y_axis": None,
                "data_points": [],
                "summary": response
            }
```

## 🧪 測試

```python
# tests/test_vlm_client.py
import pytest
from src.core.vlm_client import VLMClient
from src.analyzers.image_captioner import ImageCaptioner

class TestVLMClient:

    @pytest.fixture
    def client(self):
        return VLMClient(provider="openai")

    def test_single_image_analysis(self, client):
        result = client.analyze(
            "tests/sample_images/cat.jpg",
            "描述這張圖片的內容"
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_multiple_images(self, client):
        images = [
            "tests/sample_images/cat.jpg",
            "tests/sample_images/dog.jpg"
        ]
        result = client.analyze(
            images,
            "比較這兩張圖片的差異"
        )
        assert isinstance(result, str)

class TestDocumentAnalyzer:

    @pytest.fixture
    def analyzer(self):
        client = VLMClient(provider="openai")
        return DocumentAnalyzer(client)

    def test_invoice_analysis(self, analyzer):
        result = analyzer.analyze(
            "tests/sample_images/invoice.jpg",
            document_type="invoice"
        )
        assert result.document_type == "invoice"
        assert "total" in result.structured_data or "金額" in str(result.structured_data)
```

## 🐳 Docker 部署

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  vlm-analyzer:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ../uploads:/app/uploads
    restart: unless-stopped
```

## 📊 效能優化

### 1. 圖像壓縮
- 自動調整圖像大小至最佳解析度
- 使用適當的壓縮品質

### 2. 批量處理
- 使用非同步處理多張圖像
- 實作請求佇列和速率限制

### 3. 快取策略
- 快取重複分析結果
- 使用內容雜湊作為快取鍵

## 📚 延伸閱讀

- [OpenAI Vision 文件](https://platform.openai.com/docs/guides/vision)
- [Claude Vision 文件](https://docs.anthropic.com/en/docs/vision)
- [視覺語言模型入門](../../../3.LLM應用工程/10.多模態生成/5.視覺語言模型/)

---

*本專案持續更新中，歡迎貢獻改進建議。*
