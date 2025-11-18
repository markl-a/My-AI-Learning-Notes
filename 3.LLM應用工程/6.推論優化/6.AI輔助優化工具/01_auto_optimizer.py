"""
AI 輔助的自動優化器
使用 AI 分析場景並生成優化建議
"""

import os
import json
import torch
import psutil
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class OptimizationProvider(Enum):
    """AI 提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # 使用本地規則引擎


@dataclass
class ScenarioInfo:
    """場景資訊"""
    model_name: str
    use_case: str  # chatbot, batch_processing, real_time, etc.
    gpu_memory_gb: Optional[float] = None
    cpu_cores: Optional[int] = None
    expected_qps: Optional[int] = None  # Queries per second
    avg_input_tokens: Optional[int] = None
    avg_output_tokens: Optional[int] = None
    latency_requirement: str = "medium"  # low/medium/high
    quality_priority: str = "high"  # low/medium/high
    budget_constraint: Optional[str] = None


@dataclass
class OptimizationRecommendation:
    """優化建議"""
    quantization_method: str
    quantization_bits: int
    use_kv_cache: bool
    batch_size: int
    inference_engine: str
    additional_optimizations: List[str]
    reasoning: str
    expected_memory_gb: float
    expected_latency_ms: float
    expected_throughput: float
    confidence: float  # 0-1


class SystemAnalyzer:
    """系統分析器"""

    @staticmethod
    def get_system_info() -> Dict:
        """獲取系統資訊"""
        info = {
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_available": torch.cuda.is_available()
        }

        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return info

    @staticmethod
    def estimate_model_size(model_name: str, precision: str = "fp16") -> float:
        """
        估算模型大小（GB）

        Args:
            model_name: 模型名稱
            precision: 精度（fp32, fp16, int8, int4）

        Returns:
            模型大小（GB）
        """
        # 從模型名稱推斷參數量
        if "7b" in model_name.lower() or "7-b" in model_name.lower():
            params = 7_000_000_000
        elif "13b" in model_name.lower() or "13-b" in model_name.lower():
            params = 13_000_000_000
        elif "70b" in model_name.lower() or "70-b" in model_name.lower():
            params = 70_000_000_000
        elif "gpt2" in model_name.lower():
            if "large" in model_name.lower():
                params = 774_000_000
            elif "medium" in model_name.lower():
                params = 345_000_000
            else:
                params = 124_000_000
        else:
            # 預設假設為中型模型
            params = 1_000_000_000

        # 計算大小
        bytes_per_param = {
            "fp32": 4,
            "fp16": 2,
            "int8": 1,
            "int4": 0.5
        }

        size_bytes = params * bytes_per_param.get(precision.lower(), 2)
        size_gb = size_bytes / (1024**3)

        return size_gb


class RuleBasedOptimizer:
    """基於規則的優化器（不需要 AI API）"""

    def __init__(self):
        self.system_info = SystemAnalyzer.get_system_info()

    def recommend(self, scenario: ScenarioInfo) -> OptimizationRecommendation:
        """
        基於規則生成優化建議

        Args:
            scenario: 場景資訊

        Returns:
            優化建議
        """
        print("\n🔍 分析場景...")
        print(f"模型: {scenario.model_name}")
        print(f"使用案例: {scenario.use_case}")
        print(f"可用 GPU 記憶體: {scenario.gpu_memory_gb or self.system_info.get('gpu_memory_gb', 0):.1f} GB")

        # 估算模型大小
        model_size_fp16 = SystemAnalyzer.estimate_model_size(scenario.model_name, "fp16")
        print(f"估算模型大小 (FP16): {model_size_fp16:.2f} GB")

        # 決策邏輯
        gpu_memory = scenario.gpu_memory_gb or self.system_info.get("gpu_memory_gb", 0)
        has_gpu = self.system_info["gpu_available"]

        # 1. 選擇量化方法
        quantization, bits = self._select_quantization(
            gpu_memory, model_size_fp16, scenario.quality_priority
        )

        # 2. KV Cache
        use_kv_cache = self._should_use_kv_cache(
            scenario.use_case,
            scenario.avg_output_tokens
        )

        # 3. 批次大小
        batch_size = self._calculate_batch_size(
            gpu_memory, model_size_fp16, scenario.use_case
        )

        # 4. 推論引擎
        engine = self._select_inference_engine(
            has_gpu, scenario.use_case, scenario.latency_requirement
        )

        # 5. 額外優化
        additional = self._select_additional_optimizations(
            scenario, has_gpu
        )

        # 6. 估算性能
        expected_memory = self._estimate_memory(
            model_size_fp16, bits, batch_size
        )
        expected_latency = self._estimate_latency(
            model_size_fp16, bits, scenario.avg_output_tokens or 50
        )
        expected_throughput = self._estimate_throughput(
            batch_size, expected_latency
        )

        # 7. 生成推理
        reasoning = self._generate_reasoning(
            quantization, bits, use_kv_cache, batch_size,
            engine, scenario
        )

        return OptimizationRecommendation(
            quantization_method=quantization,
            quantization_bits=bits,
            use_kv_cache=use_kv_cache,
            batch_size=batch_size,
            inference_engine=engine,
            additional_optimizations=additional,
            reasoning=reasoning,
            expected_memory_gb=expected_memory,
            expected_latency_ms=expected_latency,
            expected_throughput=expected_throughput,
            confidence=0.85
        )

    def _select_quantization(
        self,
        gpu_memory: float,
        model_size: float,
        quality_priority: str
    ) -> tuple:
        """選擇量化方法"""
        if gpu_memory < model_size * 0.6:
            # 記憶體嚴重不足，需要激進量化
            if quality_priority == "high":
                return "AWQ", 4
            else:
                return "GPTQ", 4
        elif gpu_memory < model_size * 1.2:
            # 記憶體適中
            return "INT8", 8
        else:
            # 記憶體充足
            return "FP16", 16

    def _should_use_kv_cache(
        self,
        use_case: str,
        avg_output_tokens: Optional[int]
    ) -> bool:
        """判斷是否使用 KV Cache"""
        # 短文本生成可能不需要
        if avg_output_tokens and avg_output_tokens < 20:
            return False

        # 對話和長文本生成應該使用
        if use_case in ["chatbot", "conversation", "text_generation"]:
            return True

        return True  # 預設啟用

    def _calculate_batch_size(
        self,
        gpu_memory: float,
        model_size: float,
        use_case: str
    ) -> int:
        """計算最優批次大小"""
        if use_case == "real_time" or use_case == "chatbot":
            # 實時場景優先延遲
            return 1
        elif use_case == "batch_processing":
            # 批次處理優先吞吐量
            available = gpu_memory - model_size * 1.2
            if available > 10:
                return 32
            elif available > 5:
                return 16
            else:
                return 8
        else:
            # 預設
            return 4

    def _select_inference_engine(
        self,
        has_gpu: bool,
        use_case: str,
        latency_requirement: str
    ) -> str:
        """選擇推論引擎"""
        if not has_gpu:
            return "llama.cpp"  # CPU 優化

        if use_case == "chatbot" and latency_requirement == "low":
            return "vLLM"  # 支援 continuous batching
        elif use_case == "batch_processing":
            return "vLLM"  # 高吞吐量
        else:
            return "transformers"  # 通用

    def _select_additional_optimizations(
        self,
        scenario: ScenarioInfo,
        has_gpu: bool
    ) -> List[str]:
        """選擇額外優化"""
        optimizations = []

        if has_gpu:
            optimizations.append("使用 Flash Attention 2")

        if scenario.use_case == "chatbot":
            optimizations.append("啟用 continuous batching")

        if scenario.latency_requirement == "low":
            optimizations.append("預熱模型")
            optimizations.append("使用 JIT 編譯")

        return optimizations

    def _estimate_memory(
        self,
        model_size_fp16: float,
        bits: int,
        batch_size: int
    ) -> float:
        """估算記憶體使用"""
        # 模型權重
        model_memory = model_size_fp16 * (bits / 16)

        # KV Cache（粗略估算）
        kv_cache = model_size_fp16 * 0.1 * batch_size

        # 激活值和其他開銷
        overhead = model_size_fp16 * 0.2

        return model_memory + kv_cache + overhead

    def _estimate_latency(
        self,
        model_size: float,
        bits: int,
        output_tokens: int
    ) -> float:
        """估算延遲（ms）"""
        # 非常粗略的估算
        base_latency = model_size * 10  # 每 GB 約 10ms
        quantization_speedup = 16 / bits
        per_token = (base_latency / quantization_speedup) / 10

        return per_token * output_tokens

    def _estimate_throughput(
        self,
        batch_size: int,
        latency: float
    ) -> float:
        """估算吞吐量（請求/秒）"""
        if latency == 0:
            return 0
        return (batch_size * 1000) / latency

    def _generate_reasoning(
        self,
        quantization: str,
        bits: int,
        use_kv_cache: bool,
        batch_size: int,
        engine: str,
        scenario: ScenarioInfo
    ) -> str:
        """生成推理說明"""
        reasoning = []

        # 量化
        reasoning.append(
            f"1. 量化方法: {quantization} {bits}-bit\n"
            f"   原因: 根據可用記憶體和精度要求，{quantization} 提供最佳平衡"
        )

        # KV Cache
        cache_status = "啟用" if use_kv_cache else "禁用"
        reasoning.append(
            f"2. KV Cache: {cache_status}\n"
            f"   原因: 對於 {scenario.use_case} 場景，"
            f"{'KV Cache 可以顯著加速' if use_kv_cache else '短序列不需要 Cache'}"
        )

        # 批次大小
        reasoning.append(
            f"3. 批次大小: {batch_size}\n"
            f"   原因: 根據使用案例 ({scenario.use_case}) 和資源限制優化"
        )

        # 推論引擎
        reasoning.append(
            f"4. 推論引擎: {engine}\n"
            f"   原因: {engine} 最適合此場景的性能需求"
        )

        return "\n\n".join(reasoning)


class LLMOptimizationAdvisor:
    """LLM 優化顧問（主類）"""

    def __init__(
        self,
        provider: OptimizationProvider = OptimizationProvider.LOCAL,
        api_key: Optional[str] = None
    ):
        """
        初始化

        Args:
            provider: AI 提供商
            api_key: API 密鑰（如果使用外部 AI）
        """
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

        if provider != OptimizationProvider.LOCAL and not self.api_key:
            print("⚠️  未提供 API Key，使用本地規則引擎")
            self.provider = OptimizationProvider.LOCAL

        self.rule_optimizer = RuleBasedOptimizer()

    def analyze_and_recommend(
        self,
        scenario: ScenarioInfo
    ) -> OptimizationRecommendation:
        """
        分析場景並生成建議

        Args:
            scenario: 場景資訊

        Returns:
            優化建議
        """
        print("\n" + "="*70)
        print("🤖 LLM 推論優化顧問")
        print("="*70)

        # 使用規則引擎（總是可用）
        if self.provider == OptimizationProvider.LOCAL:
            return self.rule_optimizer.recommend(scenario)

        # TODO: 實現外部 AI API 調用
        # elif self.provider == OptimizationProvider.OPENAI:
        #     return self._get_openai_recommendation(scenario)
        # elif self.provider == OptimizationProvider.ANTHROPIC:
        #     return self._get_anthropic_recommendation(scenario)

        return self.rule_optimizer.recommend(scenario)

    def print_recommendation(self, rec: OptimizationRecommendation):
        """打印建議"""
        print("\n" + "="*70)
        print("📋 優化建議")
        print("="*70)

        print(f"\n✨ 量化策略:")
        print(f"   方法: {rec.quantization_method}")
        print(f"   位元數: {rec.quantization_bits}-bit")

        print(f"\n💾 記憶體管理:")
        print(f"   KV Cache: {'啟用' if rec.use_kv_cache else '禁用'}")
        print(f"   預期記憶體: {rec.expected_memory_gb:.2f} GB")

        print(f"\n⚡ 執行配置:")
        print(f"   批次大小: {rec.batch_size}")
        print(f"   推論引擎: {rec.inference_engine}")

        if rec.additional_optimizations:
            print(f"\n🔧 額外優化:")
            for opt in rec.additional_optimizations:
                print(f"   • {opt}")

        print(f"\n📊 預期效能:")
        print(f"   延遲: {rec.expected_latency_ms:.0f} ms")
        print(f"   吞吐量: {rec.expected_throughput:.1f} 請求/秒")

        print(f"\n💡 推理:")
        print(rec.reasoning)

        print(f"\n📈 信心度: {rec.confidence*100:.0f}%")

    def generate_code(self, rec: OptimizationRecommendation, model_name: str) -> str:
        """生成可執行代碼"""
        code = f'''"""
自動生成的優化代碼
模型: {model_name}
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 載入模型配置
model_name = "{model_name}"
'''

        if rec.quantization_bits == 4:
            code += f'''
# 4-bit 量化配置
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
'''
        elif rec.quantization_bits == 8:
            code += f'''
# 8-bit 量化配置
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
'''
        else:
            code += f'''
# FP16 模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
'''

        code += f'''
# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 配置生成參數
model.config.use_cache = {rec.use_kv_cache}

# 生成函數
def generate(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache={rec.use_kv_cache},
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 測試
if __name__ == "__main__":
    prompt = "The future of AI is"
    result = generate(prompt)
    print(result)
'''

        return code


def demo_scenarios():
    """演示不同場景"""
    advisor = LLMOptimizationAdvisor()

    scenarios = [
        ScenarioInfo(
            model_name="meta-llama/Llama-2-7b-hf",
            use_case="chatbot",
            gpu_memory_gb=16,
            expected_qps=10,
            avg_output_tokens=100,
            latency_requirement="low",
            quality_priority="high"
        ),
        ScenarioInfo(
            model_name="gpt2-large",
            use_case="batch_processing",
            gpu_memory_gb=40,
            expected_qps=None,
            avg_output_tokens=200,
            latency_requirement="medium",
            quality_priority="medium"
        ),
        ScenarioInfo(
            model_name="facebook/opt-1.3b",
            use_case="real_time",
            gpu_memory_gb=8,
            expected_qps=50,
            avg_output_tokens=50,
            latency_requirement="low",
            quality_priority="medium"
        )
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'#'*70}")
        print(f"場景 {i}: {scenario.use_case}")
        print(f"{'#'*70}")

        recommendation = advisor.analyze_and_recommend(scenario)
        advisor.print_recommendation(recommendation)

        # 生成代碼
        print("\n📝 生成的代碼:")
        print("-" * 70)
        code = advisor.generate_code(recommendation, scenario.model_name)
        print(code[:500] + "...\n(完整代碼已省略)")

        if i < len(scenarios):
            input("\n按 Enter 繼續到下一個場景...")


def main():
    """主函數"""
    print("""
╔════════════════════════════════════════════════════════════╗
║            AI 輔助的 LLM 優化顧問                           ║
║                                                            ║
║  功能:                                                     ║
║  • 分析系統資源和使用場景                                   ║
║  • 智能推薦優化策略                                        ║
║  • 生成可執行的優化代碼                                     ║
║  • 預測優化效果                                            ║
║                                                            ║
║  模式:                                                     ║
║  1. 本地規則引擎（無需 API）                               ║
║  2. 外部 AI 增強（需要 API Key）                           ║
╚════════════════════════════════════════════════════════════╝
    """)

    print("\n選擇模式:")
    print("1. 演示不同場景")
    print("2. 自定義場景")

    choice = input("\n輸入選項 (預設 1): ").strip() or "1"

    if choice == "1":
        demo_scenarios()
    elif choice == "2":
        # 自定義場景
        model_name = input("模型名稱 (預設 gpt2): ").strip() or "gpt2"
        use_case = input("使用案例 (chatbot/batch_processing/real_time): ").strip() or "chatbot"

        scenario = ScenarioInfo(
            model_name=model_name,
            use_case=use_case,
            latency_requirement="medium",
            quality_priority="high"
        )

        advisor = LLMOptimizationAdvisor()
        recommendation = advisor.analyze_and_recommend(scenario)
        advisor.print_recommendation(recommendation)

        # 詢問是否生成代碼
        if input("\n生成代碼? (y/n): ").strip().lower() == 'y':
            code = advisor.generate_code(recommendation, model_name)
            output_file = "optimized_model.py"
            with open(output_file, "w") as f:
                f.write(code)
            print(f"\n✅ 代碼已儲存到: {output_file}")

    print("\n✅ 完成！")
    print("\n🔗 下一步:")
    print("  - 智能模型選擇: 02_model_selector.py")
    print("  - 效能分析: 03_performance_analyzer.py")


if __name__ == "__main__":
    main()
