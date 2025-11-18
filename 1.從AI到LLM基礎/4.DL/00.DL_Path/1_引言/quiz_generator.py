#!/usr/bin/env python3
"""
智能測驗生成器 - 深度學習引言章節
自動生成複習題目，提供即時反饋與解釋
"""

import argparse
import random
import json
from typing import List, Dict, Tuple
from datetime import datetime

# 測驗題庫
QUIZ_BANK = {
    "easy": [
        {
            "question": "什麼是機器學習？",
            "type": "multiple_choice",
            "options": [
                "一種讓計算機從數據中學習而無需明確編程的技術",
                "一種編寫計算機程序的方法",
                "一種數據庫管理系統",
                "一種操作系統"
            ],
            "correct": 0,
            "explanation": "機器學習是一種讓計算機從數據和經驗中學習的技術，而不需要程序員明確編寫每一條規則。"
        },
        {
            "question": "以下哪個是深度學習的應用？",
            "type": "multiple_choice",
            "options": [
                "電子表格計算",
                "圖像識別",
                "文字處理",
                "文件壓縮"
            ],
            "correct": 1,
            "explanation": "圖像識別是深度學習的典型應用，如人臉識別、物體檢測等。其他選項是傳統軟件功能。"
        },
        {
            "question": "在機器學習中，模型的參數是通過什麼過程確定的？",
            "type": "multiple_choice",
            "options": [
                "程序員手動設置",
                "隨機生成",
                "訓練過程",
                "用戶輸入"
            ],
            "correct": 2,
            "explanation": "模型參數通過訓練過程自動學習和調整，這是機器學習的核心特點。"
        },
        {
            "question": "數據集在機器學習中的作用是什麼？",
            "type": "multiple_choice",
            "options": [
                "僅用於測試程序",
                "用於訓練和評估模型",
                "用於存儲結果",
                "用於備份數據"
            ],
            "correct": 1,
            "explanation": "數據集用於訓練模型（讓模型學習）和評估模型（測試性能），是機器學習的基礎。"
        },
        {
            "question": "以下哪個不是機器學習的主要類型？",
            "type": "multiple_choice",
            "options": [
                "監督學習",
                "無監督學習",
                "強化學習",
                "循環學習"
            ],
            "correct": 3,
            "explanation": "機器學習的主要類型包括監督學習、無監督學習和強化學習。循環學習不是標準的機器學習類型。"
        },
        {
            "question": "機器學習與傳統編程的主要區別是什麼？",
            "type": "multiple_choice",
            "options": [
                "使用不同的編程語言",
                "傳統編程明確規則，機器學習從數據學習規則",
                "運行速度不同",
                "使用不同的硬件"
            ],
            "correct": 1,
            "explanation": "傳統編程中程序員編寫明確的規則，而機器學習讓計算機從數據中自動學習規則和模式。"
        },
        {
            "question": "在深度學習中，'深度'指的是什麼？",
            "type": "multiple_choice",
            "options": [
                "數據的深度",
                "神經網絡的層數",
                "學習的時間長度",
                "問題的複雜度"
            ],
            "correct": 1,
            "explanation": "'深度'指的是神經網絡的層數。深度神經網絡通常有多個隱藏層，可以學習更複雜的表示。"
        },
        {
            "question": "以下哪個是監督學習的例子？",
            "type": "multiple_choice",
            "options": [
                "客戶分群",
                "垃圾郵件分類",
                "數據壓縮",
                "異常檢測（無標籤）"
            ],
            "correct": 1,
            "explanation": "垃圾郵件分類是監督學習，因為我們有標註的數據（郵件被標記為垃圾或正常）。客戶分群通常是無監督學習。"
        }
    ],

    "medium": [
        {
            "question": "在訓練模型時，為什麼要將數據分為訓練集和測試集？",
            "type": "multiple_choice",
            "options": [
                "為了節省計算時間",
                "為了評估模型在未見過數據上的表現",
                "為了增加數據量",
                "為了方便存儲"
            ],
            "correct": 1,
            "explanation": "將數據分為訓練集和測試集是為了評估模型的泛化能力，即在未見過的新數據上的表現，避免過擬合。"
        },
        {
            "question": "過擬合（Overfitting）是什麼意思？",
            "type": "multiple_choice",
            "options": [
                "模型在所有數據上表現都很差",
                "模型在訓練數據上表現好，但在測試數據上表現差",
                "模型訓練時間太長",
                "模型參數太少"
            ],
            "correct": 1,
            "explanation": "過擬合指模型過度學習訓練數據的細節和噪音，導致在訓練集表現很好，但在新數據上表現差。"
        },
        {
            "question": "什麼是損失函數（Loss Function）？",
            "type": "multiple_choice",
            "options": [
                "計算模型大小的函數",
                "測量模型預測與真實值差異的函數",
                "計算訓練時間的函數",
                "選擇最佳模型的函數"
            ],
            "correct": 1,
            "explanation": "損失函數測量模型預測與真實標籤之間的差異，訓練的目標就是最小化這個損失。"
        },
        {
            "question": "梯度下降（Gradient Descent）的作用是什麼？",
            "type": "multiple_choice",
            "options": [
                "增加模型複雜度",
                "優化模型參數以最小化損失",
                "增加訓練數據",
                "選擇最佳特徵"
            ],
            "correct": 1,
            "explanation": "梯度下降是一種優化算法，通過計算損失函數的梯度來迭代更新參數，使損失最小化。"
        },
        {
            "question": "什麼是特徵（Feature）？",
            "type": "multiple_choice",
            "options": [
                "模型的輸出",
                "描述數據的輸入變量",
                "模型的參數",
                "訓練的輪數"
            ],
            "correct": 1,
            "explanation": "特徵是描述數據樣本的輸入變量或屬性，如預測房價時的房屋面積、位置等。"
        },
        {
            "question": "無監督學習和監督學習的主要區別是什麼？",
            "type": "multiple_choice",
            "options": [
                "使用的算法不同",
                "無監督學習沒有標籤數據",
                "無監督學習更準確",
                "無監督學習速度更快"
            ],
            "correct": 1,
            "explanation": "監督學習使用帶標籤的數據（知道正確答案），而無監督學習處理無標籤數據，尋找數據的內在結構。"
        },
        {
            "question": "什麼是批次大小（Batch Size）？",
            "type": "multiple_choice",
            "options": [
                "訓練的總輪數",
                "每次更新參數時使用的樣本數量",
                "模型的層數",
                "測試集的大小"
            ],
            "correct": 1,
            "explanation": "批次大小是每次前向傳播和參數更新時使用的訓練樣本數量。較大的批次更穩定但需要更多內存。"
        },
        {
            "question": "為什麼需要激活函數（Activation Function）？",
            "type": "multiple_choice",
            "options": [
                "加快訓練速度",
                "引入非線性，使網絡能學習複雜模式",
                "減少參數數量",
                "防止過擬合"
            ],
            "correct": 1,
            "explanation": "激活函數引入非線性，使神經網絡能夠學習和表示複雜的非線性關係。沒有激活函數，多層網絡等同於單層線性模型。"
        }
    ],

    "hard": [
        {
            "question": "什麼是反向傳播（Backpropagation）？",
            "type": "multiple_choice",
            "options": [
                "一種前向計算方法",
                "計算損失函數關於參數梯度的算法",
                "一種數據預處理技術",
                "一種模型評估方法"
            ],
            "correct": 1,
            "explanation": "反向傳播是計算神經網絡中損失函數關於每個參數梯度的高效算法，通過鏈式法則從輸出層向輸入層反向計算。"
        },
        {
            "question": "在深度學習中，為什麼隨機初始化參數很重要？",
            "type": "multiple_choice",
            "options": [
                "為了加快訓練",
                "打破對稱性，使不同神經元學習不同特徵",
                "減少內存使用",
                "提高準確率"
            ],
            "correct": 1,
            "explanation": "隨機初始化打破對稱性。如果所有權重相同，所有神經元會學習相同的特徵，失去多層網絡的優勢。"
        },
        {
            "question": "什麼是學習率（Learning Rate）的作用？",
            "type": "multiple_choice",
            "options": [
                "控制模型複雜度",
                "控制參數更新的步長",
                "控制訓練時間",
                "控制批次大小"
            ],
            "correct": 1,
            "explanation": "學習率控制梯度下降時參數更新的步長。太大可能無法收斂，太小訓練會很慢。"
        },
        {
            "question": "正則化（Regularization）的主要目的是什麼？",
            "type": "multiple_choice",
            "options": [
                "加快訓練速度",
                "防止過擬合，提高泛化能力",
                "增加模型複雜度",
                "減少訓練數據需求"
            ],
            "correct": 1,
            "explanation": "正則化通過在損失函數中添加懲罰項（如 L1/L2）來限制模型複雜度，防止過擬合，提高在新數據上的表現。"
        },
        {
            "question": "Dropout 技術的工作原理是什麼？",
            "type": "multiple_choice",
            "options": [
                "刪除部分訓練數據",
                "訓練時隨機丟棄部分神經元",
                "減少網絡層數",
                "降低學習率"
            ],
            "correct": 1,
            "explanation": "Dropout 在訓練時隨機丟棄（暫時移除）部分神經元，防止神經元之間過度依賴，從而減少過擬合。"
        },
        {
            "question": "什麼是交叉驗證（Cross-Validation）？",
            "type": "multiple_choice",
            "options": [
                "一種損失函數",
                "一種評估模型性能的方法，將數據分為多個折疊",
                "一種優化算法",
                "一種特徵選擇方法"
            ],
            "correct": 1,
            "explanation": "交叉驗證將數據分為k個折疊，輪流使用其中一個作為驗證集，其餘作為訓練集，更可靠地評估模型性能。"
        },
        {
            "question": "什麼是遷移學習（Transfer Learning）？",
            "type": "multiple_choice",
            "options": [
                "將數據從一個系統轉移到另一個",
                "在新任務上使用預訓練模型的知識",
                "將模型從訓練模式轉為推理模式",
                "將數據從訓練集轉移到測試集"
            ],
            "correct": 1,
            "explanation": "遷移學習利用在大規模數據集上預訓練的模型，將學到的知識應用到新的相關任務上，大大減少所需的訓練數據和時間。"
        },
        {
            "question": "批標準化（Batch Normalization）的主要好處是什麼？",
            "type": "multiple_choice",
            "options": [
                "減少模型參數",
                "穩定訓練過程，加快收斂",
                "減少訓練數據需求",
                "自動選擇最佳架構"
            ],
            "correct": 1,
            "explanation": "批標準化通過標準化每層的輸入來穩定訓練，允許使用更大的學習率，加快收斂，並具有輕微的正則化效果。"
        }
    ],

    "true_false": [
        {
            "question": "深度學習是機器學習的一個子集。",
            "type": "true_false",
            "correct": True,
            "explanation": "正確！深度學習是機器學習的一個子集，特別使用深度神經網絡進行學習。"
        },
        {
            "question": "機器學習模型總是需要大量標註數據。",
            "type": "true_false",
            "correct": False,
            "explanation": "錯誤！只有監督學習需要標註數據。無監督學習、半監督學習和遷移學習可以利用較少的標註數據。"
        },
        {
            "question": "過擬合意味著模型在訓練集和測試集上都表現很差。",
            "type": "true_false",
            "correct": False,
            "explanation": "錯誤！過擬合是指模型在訓練集上表現好，但在測試集上表現差。兩者都差是欠擬合。"
        },
        {
            "question": "參數和超參數是同一個概念。",
            "type": "true_false",
            "correct": False,
            "explanation": "錯誤！參數是通過訓練學習的（如權重），超參數是在訓練前設置的（如學習率、層數）。"
        },
        {
            "question": "增加模型複雜度總是會提高性能。",
            "type": "true_false",
            "correct": False,
            "explanation": "錯誤！過於複雜的模型可能導致過擬合，在新數據上表現更差。需要找到合適的複雜度平衡。"
        }
    ]
}


class QuizSession:
    """測驗會話類"""

    def __init__(self, difficulty: str = "easy", count: int = 5):
        self.difficulty = difficulty
        self.count = count
        self.questions: List[Dict] = []
        self.answers: List[Tuple[int, bool]] = []  # (question_idx, is_correct)
        self.start_time = datetime.now()

    def generate_questions(self):
        """生成測驗題目"""
        if self.difficulty == "mixed":
            # 混合難度
            easy = random.sample(QUIZ_BANK["easy"], min(2, len(QUIZ_BANK["easy"])))
            medium = random.sample(QUIZ_BANK["medium"], min(2, len(QUIZ_BANK["medium"])))
            hard = random.sample(QUIZ_BANK["hard"], min(1, len(QUIZ_BANK["hard"])))
            self.questions = easy + medium + hard
        elif self.difficulty == "all":
            # 包含判斷題
            pool = (QUIZ_BANK["easy"] + QUIZ_BANK["medium"] +
                    QUIZ_BANK["hard"] + QUIZ_BANK["true_false"])
            self.questions = random.sample(pool, min(self.count, len(pool)))
        else:
            pool = QUIZ_BANK.get(self.difficulty, QUIZ_BANK["easy"])
            self.questions = random.sample(pool, min(self.count, len(pool)))

        random.shuffle(self.questions)

    def ask_question(self, idx: int) -> bool:
        """詢問單個問題"""
        q = self.questions[idx]

        print(f"\n{'='*60}")
        print(f"問題 {idx + 1}/{len(self.questions)}")
        print(f"{'='*60}")
        print(f"\n{q['question']}\n")

        if q['type'] == 'multiple_choice':
            for i, option in enumerate(q['options']):
                print(f"  {chr(65 + i)}. {option}")

            while True:
                answer = input("\n你的答案 (A/B/C/D): ").strip().upper()
                if answer in ['A', 'B', 'C', 'D']:
                    user_answer = ord(answer) - 65
                    break
                print("❌ 請輸入 A、B、C 或 D")

            is_correct = (user_answer == q['correct'])

        else:  # true_false
            while True:
                answer = input("\n你的答案 (T/F 或 正確/錯誤): ").strip().upper()
                if answer in ['T', 'F', '正確', '錯誤', 'TRUE', 'FALSE']:
                    user_answer = answer in ['T', '正確', 'TRUE']
                    break
                print("❌ 請輸入 T/F 或 正確/錯誤")

            is_correct = (user_answer == q['correct'])

        # 即時反饋
        if is_correct:
            print("\n✅ 正確！")
        else:
            print("\n❌ 錯誤！")
            if q['type'] == 'multiple_choice':
                print(f"正確答案是：{chr(65 + q['correct'])}. {q['options'][q['correct']]}")
            else:
                print(f"正確答案是：{'正確' if q['correct'] else '錯誤'}")

        print(f"\n💡 解釋：{q['explanation']}")

        self.answers.append((idx, is_correct))
        return is_correct

    def show_results(self):
        """顯示測驗結果"""
        correct_count = sum(1 for _, is_correct in self.answers if is_correct)
        total = len(self.answers)
        percentage = (correct_count / total * 100) if total > 0 else 0
        duration = (datetime.now() - self.start_time).total_seconds()

        print(f"\n\n{'='*60}")
        print("📊 測驗結果")
        print(f"{'='*60}\n")

        print(f"✓ 正確：{correct_count}/{total}")
        print(f"📈 正確率：{percentage:.1f}%")
        print(f"⏱️  用時：{int(duration // 60)} 分 {int(duration % 60)} 秒\n")

        # 評級
        if percentage >= 90:
            grade = "優秀！🌟"
            comment = "你對這些概念掌握得非常好！"
        elif percentage >= 75:
            grade = "良好！👍"
            comment = "大部分概念都掌握了，繼續加油！"
        elif percentage >= 60:
            grade = "及格 ✓"
            comment = "基本概念已經了解，但還需要加強。"
        else:
            grade = "需要努力 💪"
            comment = "建議重新學習相關章節，鞏固基礎。"

        print(f"評級：{grade}")
        print(f"評語：{comment}\n")

        # 錯題分析
        wrong_questions = [(idx, is_correct) for idx, is_correct in self.answers if not is_correct]
        if wrong_questions:
            print(f"{'─'*60}")
            print(f"❌ 錯題回顧（共 {len(wrong_questions)} 題）\n")
            for idx, _ in wrong_questions:
                q = self.questions[idx]
                print(f"• {q['question']}")
                print(f"  💡 {q['explanation']}\n")

        # 學習建議
        print(f"{'─'*60}")
        print("📚 學習建議：\n")
        if percentage < 60:
            print("1. 重新閱讀 index.ipynb 的相關章節")
            print("2. 使用 ai_learning_assistant.py 獲取概念解釋")
            print("3. 運行實踐 notebook 加深理解")
            print("4. 一週後再次測試")
        elif percentage < 80:
            print("1. 針對錯題相關的概念進行複習")
            print("2. 完成更多練習題")
            print("3. 嘗試用自己的話解釋這些概念")
        else:
            print("1. 可以進入下一章節學習")
            print("2. 定期複習以保持記憶")
            print("3. 嘗試應用到實際項目中")

        print(f"\n{'='*60}\n")

    def save_results(self, filename: str = None):
        """保存測驗結果"""
        if filename is None:
            filename = f"quiz_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        correct_count = sum(1 for _, is_correct in self.answers if is_correct)
        total = len(self.answers)

        results = {
            "timestamp": self.start_time.isoformat(),
            "difficulty": self.difficulty,
            "total_questions": total,
            "correct_answers": correct_count,
            "percentage": (correct_count / total * 100) if total > 0 else 0,
            "questions": [
                {
                    "question": self.questions[idx]['question'],
                    "correct": is_correct
                }
                for idx, is_correct in self.answers
            ]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"✓ 結果已保存到：{filename}")


def interactive_quiz():
    """交互式測驗模式"""
    print("\n" + "="*60)
    print("📝 智能測驗生成器")
    print("="*60)

    # 選擇難度
    print("\n請選擇難度：")
    print("  1. 簡單 (easy)")
    print("  2. 中等 (medium)")
    print("  3. 困難 (hard)")
    print("  4. 混合 (mixed)")
    print("  5. 全部題型 (all)")

    while True:
        choice = input("\n輸入數字 (1-5): ").strip()
        difficulty_map = {
            '1': 'easy',
            '2': 'medium',
            '3': 'hard',
            '4': 'mixed',
            '5': 'all'
        }
        if choice in difficulty_map:
            difficulty = difficulty_map[choice]
            break
        print("❌ 請輸入 1-5 之間的數字")

    # 選擇題目數量
    while True:
        try:
            count = int(input("\n題目數量 (建議 5-10): ").strip())
            if 1 <= count <= 50:
                break
            print("❌ 請輸入 1-50 之間的數字")
        except ValueError:
            print("❌ 請輸入有效數字")

    # 開始測驗
    session = QuizSession(difficulty, count)
    session.generate_questions()

    print(f"\n{'='*60}")
    print(f"開始測驗！共 {len(session.questions)} 題")
    print(f"{'='*60}")

    input("\n按 Enter 開始...")

    # 逐題作答
    for i in range(len(session.questions)):
        session.ask_question(i)

        if i < len(session.questions) - 1:
            input("\n按 Enter 繼續下一題...")

    # 顯示結果
    session.show_results()

    # 保存結果
    save = input("\n是否保存結果？(y/n): ").strip().lower()
    if save == 'y':
        session.save_results()


def main():
    parser = argparse.ArgumentParser(
        description="智能測驗生成器 - 深度學習引言章節",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 生成5道簡單題目
  python quiz_generator.py --difficulty easy --count 5

  # 交互式模式
  python quiz_generator.py --interactive

  # 生成混合難度題目
  python quiz_generator.py --difficulty mixed --count 10
        """
    )

    parser.add_argument('--difficulty', '-d',
                        choices=['easy', 'medium', 'hard', 'mixed', 'all'],
                        default='easy',
                        help='題目難度')
    parser.add_argument('--count', '-c', type=int, default=5,
                        help='題目數量')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='交互式模式')

    args = parser.parse_args()

    if args.interactive:
        interactive_quiz()
    else:
        session = QuizSession(args.difficulty, args.count)
        session.generate_questions()

        print(f"\n{'='*60}")
        print(f"測驗開始！難度：{args.difficulty}，共 {len(session.questions)} 題")
        print(f"{'='*60}")

        for i in range(len(session.questions)):
            session.ask_question(i)
            if i < len(session.questions) - 1:
                input("\n按 Enter 繼續...")

        session.show_results()

        save = input("\n是否保存結果？(y/n): ").strip().lower()
        if save == 'y':
            session.save_results()


if __name__ == "__main__":
    main()
