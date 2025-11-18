"""
學習進度追蹤器
============

追蹤學習進度，分析薄弱環節，提供個性化學習建議。

使用方法：
    python progress_tracker.py --update --topic ndarray --score 85
    python progress_tracker.py --report
    python progress_tracker.py --suggest

作者：AI Learning Community
版本：v1.0
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np


class ProgressTracker:
    """學習進度追蹤器"""

    def __init__(self, data_file='./ai_tools/progress_data.json'):
        self.data_file = data_file
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, Any]:
        """加載進度數據"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'user_id': 'learner_001',
            'start_date': datetime.now().isoformat(),
            'topics': {
                'ndarray': {'score': 0, 'time_spent': 0, 'exercises_completed': 0, 'last_update': None},
                'pandas': {'score': 0, 'time_spent': 0, 'exercises_completed': 0, 'last_update': None},
                'linear_algebra': {'score': 0, 'time_spent': 0, 'exercises_completed': 0, 'last_update': None},
                'calculus': {'score': 0, 'time_spent': 0, 'exercises_completed': 0, 'last_update': None},
                'autograd': {'score': 0, 'time_spent': 0, 'exercises_completed': 0, 'last_update': None},
                'probability': {'score': 0, 'time_spent': 0, 'exercises_completed': 0, 'last_update': None},
            },
            'total_time': 0,
            'milestones': [],
            'notes': []
        }

    def _save_data(self):
        """保存進度數據"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print("✅ 進度已保存")

    def update_progress(self, topic: str, score: int = None, time_spent: int = None,
                       exercises: int = None):
        """更新學習進度"""
        if topic not in self.data['topics']:
            print(f"❌ 未知主題: {topic}")
            return

        topic_data = self.data['topics'][topic]

        if score is not None:
            topic_data['score'] = max(topic_data['score'], score)  # 保留最高分
        if time_spent is not None:
            topic_data['time_spent'] += time_spent
            self.data['total_time'] += time_spent
        if exercises is not None:
            topic_data['exercises_completed'] += exercises

        topic_data['last_update'] = datetime.now().isoformat()

        # 檢查里程碑
        self._check_milestones(topic, score)

        self._save_data()
        print(f"✅ {topic} 進度已更新")

    def _check_milestones(self, topic: str, score: int):
        """檢查是否達成里程碑"""
        milestones = [
            (60, "入門"),
            (75, "熟練"),
            (90, "精通"),
            (100, "大師")
        ]

        for threshold, level in milestones:
            if score >= threshold:
                milestone = {
                    'topic': topic,
                    'level': level,
                    'score': score,
                    'achieved_at': datetime.now().isoformat()
                }
                # 避免重複添加
                if not any(m['topic'] == topic and m['level'] == level for m in self.data['milestones']):
                    self.data['milestones'].append(milestone)
                    print(f"🎉 恭喜！你在 {topic} 達到了 {level} 水平！")

    def generate_report(self):
        """生成學習報告"""
        print(f"\n{'='*80}")
        print(f"📊 學習進度報告")
        print(f"{'='*80}\n")

        print(f"用戶ID: {self.data['user_id']}")
        print(f"開始日期: {self.data['start_date'][:10]}")
        print(f"總學習時間: {self.data['total_time']} 小時\n")

        print(f"{'主題':<20} {'分數':<10} {'時間(h)':<10} {'練習數':<10} {'最後更新':<20}")
        print(f"{'-'*80}")

        for topic, data in self.data['topics'].items():
            last_update = data['last_update'][:10] if data['last_update'] else 'N/A'
            print(f"{topic:<20} {data['score']:<10} {data['time_spent']:<10} "
                  f"{data['exercises_completed']:<10} {last_update:<20}")

        # 計算總體進度
        avg_score = np.mean([d['score'] for d in self.data['topics'].values()])
        print(f"\n平均分數: {avg_score:.1f}")

        # 里程碑
        if self.data['milestones']:
            print(f"\n🏆 已達成的里程碑:")
            for m in self.data['milestones']:
                print(f"  - {m['topic']}: {m['level']} (分數: {m['score']})")

        # 可視化
        self._visualize_progress()

    def _visualize_progress(self):
        """可視化學習進度"""
        topics = list(self.data['topics'].keys())
        scores = [self.data['topics'][t]['score'] for t in topics]
        times = [self.data['topics'][t]['time_spent'] for t in topics]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 分數雷達圖
        angles = np.linspace(0, 2 * np.pi, len(topics), endpoint=False).tolist()
        scores_plot = scores + [scores[0]]
        angles += angles[:1]

        ax1 = plt.subplot(121, projection='polar')
        ax1.plot(angles, scores_plot, 'o-', linewidth=2, color='royalblue')
        ax1.fill(angles, scores_plot, alpha=0.25, color='royalblue')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(topics, size=10)
        ax1.set_ylim(0, 100)
        ax1.set_title('各主題掌握程度', size=14, fontweight='bold', pad=20)
        ax1.grid(True)

        # 學習時間柱狀圖
        ax2 = plt.subplot(122)
        colors = plt.cm.viridis(np.linspace(0, 1, len(topics)))
        bars = ax2.bar(topics, times, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('學習時間 (小時)', fontsize=12)
        ax2.set_title('各主題學習時間', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # 添加數值標籤
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}h',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('./ai_tools/learning_progress.png', dpi=150, bbox_inches='tight')
        print("\n✅ 進度圖表已保存至 learning_progress.png")
        plt.show()

    def get_suggestions(self):
        """獲取個性化學習建議"""
        print(f"\n{'='*80}")
        print(f"💡 個性化學習建議")
        print(f"{'='*80}\n")

        suggestions = []

        # 分析薄弱環節
        weak_topics = [(topic, data['score']) for topic, data in self.data['topics'].items()
                      if data['score'] < 70]

        if weak_topics:
            weak_topics.sort(key=lambda x: x[1])
            print("📌 需要加強的主題:")
            for topic, score in weak_topics:
                print(f"  - {topic} (當前分數: {score})")
                suggestions.append(f"建議複習 {topic}，目標提升至 75 分以上")

        # 推薦學習順序
        print("\n📚 推薦學習順序:")
        topic_order = ['ndarray', 'pandas', 'linear_algebra', 'calculus', 'autograd', 'probability']
        for i, topic in enumerate(topic_order, 1):
            status = "✅" if self.data['topics'][topic]['score'] >= 75 else "⬜"
            print(f"  {i}. {status} {topic}")

        # 時間分配建議
        print("\n⏰ 時間分配建議:")
        for topic in weak_topics[:3]:  # 前三個薄弱主題
            recommended_time = max(5, (75 - topic[1]) // 10)  # 根據分數差距推薦時間
            print(f"  - {topic[0]}: 建議再投入 {recommended_time} 小時")

        # 學習策略建議
        print("\n🎯 學習策略建議:")
        avg_score = np.mean([d['score'] for d in self.data['topics'].values()])

        if avg_score < 60:
            print("  - 當前處於入門階段，建議:")
            print("    1. 按順序完成每個 notebook")
            print("    2. 務必完成所有練習題")
            print("    3. 使用 AI 輔助工具生成額外練習")
        elif avg_score < 80:
            print("  - 當前處於進階階段，建議:")
            print("    1. 深入理解數學原理")
            print("    2. 完成實踐項目")
            print("    3. 嘗試實現一些算法")
        else:
            print("  - 當前處於精通階段，建議:")
            print("    1. 閱讀相關論文")
            print("    2. 參與開源項目")
            print("    3. 分享學習心得，教授他人")

        return suggestions

    def add_note(self, topic: str, note: str):
        """添加學習筆記"""
        self.data['notes'].append({
            'topic': topic,
            'content': note,
            'created_at': datetime.now().isoformat()
        })
        self._save_data()
        print(f"✅ 筆記已添加到 {topic}")


def main():
    parser = argparse.ArgumentParser(description='學習進度追蹤器')
    parser.add_argument('--update', action='store_true', help='更新進度')
    parser.add_argument('--topic', type=str, help='主題名稱')
    parser.add_argument('--score', type=int, help='分數 (0-100)')
    parser.add_argument('--time', type=int, help='學習時間（小時）')
    parser.add_argument('--exercises', type=int, help='完成的練習數')
    parser.add_argument('--report', action='store_true', help='生成學習報告')
    parser.add_argument('--suggest', action='store_true', help='獲取學習建議')
    parser.add_argument('--note', type=str, help='添加學習筆記')

    args = parser.parse_args()

    tracker = ProgressTracker()

    if args.update:
        if not args.topic:
            print("❌ 請指定主題 (--topic)")
            return
        tracker.update_progress(args.topic, args.score, args.time, args.exercises)

    if args.report:
        tracker.generate_report()

    if args.suggest:
        tracker.get_suggestions()

    if args.note and args.topic:
        tracker.add_note(args.topic, args.note)


if __name__ == '__main__':
    main()
