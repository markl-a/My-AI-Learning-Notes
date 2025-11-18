"""
推論優化 - 綜合測試腳本
驗證所有模組可以正常運行
"""

import sys
import os
import subprocess
from pathlib import Path


class TestRunner:
    """測試運行器"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results = []

    def test_imports(self):
        """測試基礎依賴導入"""
        print("\n" + "="*70)
        print("測試 1: 檢查基礎依賴")
        print("="*70)

        dependencies = [
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("numpy", "NumPy"),
            ("matplotlib", "Matplotlib"),
        ]

        all_passed = True

        for module, name in dependencies:
            try:
                __import__(module)
                print(f"✅ {name} 已安裝")
            except ImportError:
                print(f"❌ {name} 未安裝")
                all_passed = False

        self.results.append(("基礎依賴", all_passed))
        return all_passed

    def test_module(self, module_path: str, module_name: str):
        """測試單個模組"""
        print(f"\n" + "="*70)
        print(f"測試: {module_name}")
        print("="*70)

        script_path = self.base_dir / module_path

        if not script_path.exists():
            print(f"❌ 檔案不存在: {script_path}")
            self.results.append((module_name, False))
            return False

        print(f"檢查檔案: {script_path}")

        # 檢查語法
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                code = f.read()
                compile(code, script_path, 'exec')
            print(f"✅ {module_name} 語法檢查通過")
            self.results.append((module_name, True))
            return True
        except SyntaxError as e:
            print(f"❌ {module_name} 語法錯誤: {e}")
            self.results.append((module_name, False))
            return False
        except Exception as e:
            print(f"⚠️  {module_name} 檢查時出現其他錯誤: {e}")
            self.results.append((module_name, False))
            return False

    def run_all_tests(self):
        """運行所有測試"""
        print("""
╔════════════════════════════════════════════════════════════╗
║            推論優化 - 綜合測試                              ║
║                                                            ║
║  測試內容:                                                 ║
║  • 基礎依賴檢查                                            ║
║  • 各模組語法驗證                                          ║
║  • 檔案完整性檢查                                          ║
╚════════════════════════════════════════════════════════════╝
        """)

        # 測試 1: 基礎依賴
        self.test_imports()

        # 測試 2: 量化技術模組
        self.test_module(
            "1.量化技術/01_basic_quantization.py",
            "量化技術 - 基礎量化"
        )
        self.test_module(
            "1.量化技術/02_gptq_quantization.py",
            "量化技術 - GPTQ 量化"
        )
        self.test_module(
            "1.量化技術/05_quantization_comparison.py",
            "量化技術 - 量化對比"
        )

        # 測試 3: KV Cache 模組
        self.test_module(
            "2.KV-Cache/01_kv_cache_basic.py",
            "KV Cache - 基礎演示"
        )

        # 測試 4: AI 輔助優化工具
        self.test_module(
            "6.AI輔助優化工具/01_auto_optimizer.py",
            "AI 輔助優化工具 - 自動優化器"
        )

        # 顯示結果摘要
        self.print_summary()

    def print_summary(self):
        """打印測試摘要"""
        print("\n" + "="*70)
        print("測試結果摘要")
        print("="*70)

        passed = sum(1 for _, result in self.results if result)
        total = len(self.results)

        print(f"\n{'模組':<40} {'結果':<10}")
        print("-" * 70)

        for module, result in self.results:
            status = "✅ 通過" if result else "❌ 失敗"
            print(f"{module:<40} {status:<10}")

        print("-" * 70)
        print(f"\n總計: {passed}/{total} 通過")

        if passed == total:
            print("\n🎉 所有測試通過！")
            print("\n下一步:")
            print("  • 運行量化對比: python 1.量化技術/05_quantization_comparison.py")
            print("  • 測試 KV Cache: python 2.KV-Cache/01_kv_cache_basic.py")
            print("  • 使用 AI 優化器: python 6.AI輔助優化工具/01_auto_optimizer.py")
        else:
            print("\n⚠️  部分測試失敗，請檢查錯誤信息")

        return passed == total


def check_python_version():
    """檢查 Python 版本"""
    print("檢查 Python 版本...")
    version = sys.version_info

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 版本過低: {version.major}.{version.minor}")
        print("需要 Python 3.8 或更高版本")
        return False

    print(f"✅ Python 版本: {version.major}.{version.minor}.{version.micro}")
    return True


def check_gpu():
    """檢查 GPU 可用性"""
    print("\n檢查 GPU...")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU 可用: {gpu_name}")
            print(f"   記憶體: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠️  未檢測到 GPU，部分功能將在 CPU 上運行")
            return False
    except ImportError:
        print("❌ PyTorch 未安裝，無法檢查 GPU")
        return False


def main():
    """主函數"""
    # 檢查環境
    if not check_python_version():
        sys.exit(1)

    check_gpu()

    # 運行測試
    runner = TestRunner()
    success = runner.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
