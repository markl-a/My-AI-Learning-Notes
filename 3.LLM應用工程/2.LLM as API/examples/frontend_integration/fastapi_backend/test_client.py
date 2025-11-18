"""
FastAPI 後端測試客戶端
測試所有 API 端點的功能
"""

import requests
import json
import time
from typing import Dict, Generator

# API 設定
BASE_URL = "http://localhost:8000"
API_KEY = "your-secret-key"  # 從環境變數獲取

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def test_health():
    """測試健康檢查端點"""
    print("\n=== 測試健康檢查 ===")

    response = requests.get(f"{BASE_URL}/health")

    print(f"狀態碼: {response.status_code}")
    print(f"回應: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

    return response.json()


def test_chat_basic(provider: str = "openai", model: str = None):
    """測試基本聊天功能"""
    print(f"\n=== 測試基本聊天 ({provider}) ===")

    payload = {
        "messages": [
            {"role": "system", "content": "你是一個專業的助理"},
            {"role": "user", "content": "用一句話解釋什麼是 API"}
        ],
        "provider": provider,
        "temperature": 0.7,
        "stream": False
    }

    if model:
        payload["model"] = model

    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/api/chat",
        headers=HEADERS,
        json=payload
    )
    duration = time.time() - start_time

    print(f"狀態碼: {response.status_code}")
    print(f"耗時: {duration:.2f}秒")

    if response.status_code == 200:
        result = response.json()
        print(f"\n提供商: {result['provider']}")
        print(f"模型: {result['model']}")
        print(f"回應: {result['message']}")
        print(f"\nToken 使用:")
        print(f"  - Prompt: {result['usage']['prompt_tokens']}")
        print(f"  - Completion: {result['usage']['completion_tokens']}")
        print(f"  - Total: {result['usage']['total_tokens']}")
    else:
        print(f"錯誤: {response.text}")

    return response.json() if response.status_code == 200 else None


def test_chat_stream(provider: str = "openai"):
    """測試串流聊天功能"""
    print(f"\n=== 測試串流聊天 ({provider}) ===")

    payload = {
        "messages": [
            {"role": "user", "content": "寫一個 Python 函數來計算階乘"}
        ],
        "provider": provider,
        "stream": True
    }

    print("AI 回應: ", end="", flush=True)

    response = requests.post(
        f"{BASE_URL}/api/chat",
        headers=HEADERS,
        json=payload,
        stream=True
    )

    if response.status_code == 200:
        full_response = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    if 'content' in data:
                        print(data['content'], end="", flush=True)
                        full_response += data['content']
                    elif 'done' in data and data['done']:
                        print(f"\n\n✅ 完成 (耗時: {data['duration']:.2f}秒)")

        return full_response
    else:
        print(f"\n❌ 錯誤: {response.text}")
        return None


def test_multi_turn_conversation(provider: str = "openai"):
    """測試多輪對話"""
    print(f"\n=== 測試多輪對話 ({provider}) ===")

    conversation = [
        {"role": "system", "content": "你是一個友善的 AI 助理"}
    ]

    # 第一輪
    conversation.append({"role": "user", "content": "我想學習 Python"})

    response1 = requests.post(
        f"{BASE_URL}/api/chat",
        headers=HEADERS,
        json={
            "messages": conversation,
            "provider": provider,
            "stream": False
        }
    )

    if response1.status_code == 200:
        result1 = response1.json()
        print(f"\n用戶: 我想學習 Python")
        print(f"AI: {result1['message'][:100]}...")

        conversation.append({"role": "assistant", "content": result1['message']})

        # 第二輪
        conversation.append({"role": "user", "content": "應該從哪裡開始？"})

        response2 = requests.post(
            f"{BASE_URL}/api/chat",
            headers=HEADERS,
            json={
                "messages": conversation,
                "provider": provider,
                "stream": False
            }
        )

        if response2.status_code == 200:
            result2 = response2.json()
            print(f"\n用戶: 應該從哪裡開始？")
            print(f"AI: {result2['message'][:100]}...")

            print(f"\n✅ 多輪對話成功")
            return True

    print(f"\n❌ 多輪對話失敗")
    return False


def test_error_handling():
    """測試錯誤處理"""
    print("\n=== 測試錯誤處理 ===")

    # 測試無效的 API Key
    print("\n1. 測試無效的 API Key")
    response = requests.post(
        f"{BASE_URL}/api/chat",
        headers={"Authorization": "Bearer invalid-key"},
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "provider": "openai"
        }
    )
    print(f"狀態碼: {response.status_code}")
    assert response.status_code == 401, "應該返回 401"
    print("✅ 正確處理無效的 API Key")

    # 測試無效的提供商
    print("\n2. 測試無效的提供商")
    response = requests.post(
        f"{BASE_URL}/api/chat",
        headers=HEADERS,
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "provider": "invalid_provider"
        }
    )
    print(f"狀態碼: {response.status_code}")
    assert response.status_code == 422, "應該返回 422"
    print("✅ 正確處理無效的提供商")

    # 測試空訊息
    print("\n3. 測試空訊息列表")
    response = requests.post(
        f"{BASE_URL}/api/chat",
        headers=HEADERS,
        json={
            "messages": [],
            "provider": "openai"
        }
    )
    print(f"狀態碼: {response.status_code}")
    print("✅ 正確處理空訊息")


def test_metrics():
    """測試指標端點"""
    print("\n=== 測試 Prometheus 指標 ===")

    response = requests.get(f"{BASE_URL}/metrics")

    print(f"狀態碼: {response.status_code}")

    if response.status_code == 200:
        metrics = response.text
        print(f"\n指標樣本（前 500 字符）:")
        print(metrics[:500])
        print("...")
        print("\n✅ 指標端點正常")
    else:
        print("❌ 指標端點異常")


def run_all_tests():
    """運行所有測試"""
    print("=" * 60)
    print("FastAPI 後端測試套件")
    print("=" * 60)

    try:
        # 1. 健康檢查
        health = test_health()

        # 2. 基本聊天測試
        test_chat_basic("openai")

        # 如果 Anthropic 可用
        if health.get('providers', {}).get('anthropic'):
            test_chat_basic("anthropic")

        # 3. 串流測試
        test_chat_stream("openai")

        # 4. 多輪對話測試
        test_multi_turn_conversation("openai")

        # 5. 錯誤處理測試
        test_error_handling()

        # 6. 指標測試
        test_metrics()

        print("\n" + "=" * 60)
        print("✅ 所有測試完成")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


def interactive_test():
    """互動式測試"""
    print("\n" + "=" * 60)
    print("互動式測試模式")
    print("=" * 60)

    while True:
        print("\n選擇測試:")
        print("1. 健康檢查")
        print("2. 基本聊天 (OpenAI)")
        print("3. 基本聊天 (Anthropic)")
        print("4. 串流聊天")
        print("5. 多輪對話")
        print("6. 自定義請求")
        print("7. 錯誤處理測試")
        print("8. 查看指標")
        print("9. 運行所有測試")
        print("0. 退出")

        choice = input("\n輸入選擇 (0-9): ").strip()

        if choice == '0':
            print("退出測試")
            break
        elif choice == '1':
            test_health()
        elif choice == '2':
            test_chat_basic("openai")
        elif choice == '3':
            test_chat_basic("anthropic")
        elif choice == '4':
            provider = input("選擇提供商 (openai/anthropic): ").strip()
            test_chat_stream(provider)
        elif choice == '5':
            provider = input("選擇提供商 (openai/anthropic): ").strip()
            test_multi_turn_conversation(provider)
        elif choice == '6':
            prompt = input("輸入提示: ").strip()
            provider = input("選擇提供商 (openai/anthropic): ").strip()
            test_chat_basic(provider)
        elif choice == '7':
            test_error_handling()
        elif choice == '8':
            test_metrics()
        elif choice == '9':
            run_all_tests()
        else:
            print("無效的選擇")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        run_all_tests()
