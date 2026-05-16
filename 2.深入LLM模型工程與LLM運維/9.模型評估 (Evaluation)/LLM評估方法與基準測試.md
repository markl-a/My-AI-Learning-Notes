# LLM 評估方法與基準測試完整指南

## 目錄
1. [評估的重要性與挑戰](#1-評估的重要性與挑戰)
2. [傳統自動化指標](#2-傳統自動化指標)
3. [通用評估基準](#3-通用評估基準)
4. [任務特定評估](#4-任務特定評估)
5. [人類評估方法](#5-人類評估方法)
6. [實作範例](#6-實作範例)

---

## 1. 評估的重要性與挑戰

### 1.1 為什麼需要評估？

**模型開發週期中的評估**：
```
預訓練 → 評估（困惑度、下游任務）
   ↓
微調 → 評估（任務準確率、BLEU等）
   ↓
對齊 → 評估（有用性、無害性、誠實性）
   ↓
部署 → 評估（用戶滿意度、業務指標）
```

### 1.2 評估的挑戰

1. **開放式生成難以量化**
   - 多個正確答案
   - 創意性無法衡量
   - 風格偏好主觀

2. **自動指標與人類判斷不一致**
   - BLEU 高但質量差
   - 困惑度低但不實用

3. **評估成本高**
   - 人類評估昂貴
   - 全面測試耗時

---

## 2. 傳統自動化指標

### 2.1 困惑度 (Perplexity)

**定義**：模型預測測試集的平均不確定性

```python
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity(model, tokenizer, text):
    """計算困惑度"""
    encodings = tokenizer(text, return_tensors='pt')

    max_length = model.config.n_positions
    stride = 512

    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i

        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

# 使用
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "The quick brown fox jumps over the lazy dog."
ppl = calculate_perplexity(model, tokenizer, text)
print(f"Perplexity: {ppl:.2f}")
```

**優點**：
- 快速計算
- 可比較不同模型

**缺點**：
- 與生成品質相關性弱
- 不反映實用性

### 2.2 BLEU (機器翻譯)

**定義**：n-gram 精確度的加權幾何平均

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

def calculate_bleu(reference, candidate):
    """計算 BLEU 分數"""
    # reference: 參考翻譯（可多個）
    # candidate: 候選翻譯

    reference_tokens = [ref.split() for ref in reference]
    candidate_tokens = candidate.split()

    # BLEU-4 (1-4 gram)
    score = sentence_bleu(reference_tokens, candidate_tokens)
    return score

# 示例
references = ["這是一個測試句子", "這是測試語句"]
candidate = "這是一個測試語句"

score = calculate_bleu(references, candidate)
print(f"BLEU: {score:.4f}")
```

### 2.3 ROUGE (文字摘要)

**定義**：召回率導向的 n-gram 重疊

```python
from rouge import Rouge

def calculate_rouge(reference, candidate):
    """計算 ROUGE 分數"""
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)[0]

    return {
        'rouge-1': scores['rouge-1']['f'],
        'rouge-2': scores['rouge-2']['f'],
        'rouge-l': scores['rouge-l']['f']
    }

# 使用
reference = "人工智慧正在改變世界"
candidate = "AI 正在改變世界"

scores = calculate_rouge(reference, candidate)
print(f"ROUGE-1: {scores['rouge-1']:.4f}")
print(f"ROUGE-2: {scores['rouge-2']:.4f}")
print(f"ROUGE-L: {scores['rouge-l']:.4f}")
```

### 2.4 傳統指標的局限性

**問題範例**：
```python
reference = "貓坐在墊子上"
candidate1 = "貓坐在墊子上"  # BLEU=1.0, 完美但無用（重複）
candidate2 = "一隻貓舒適地坐在柔軟的墊子上"  # BLEU較低, 但更好

# 傳統指標偏好字面匹配，忽略語義相似性
```

---

## 3. 通用評估基準

### 3.1 Open LLM Leaderboard

**Hugging Face 官方排行榜**

**評估任務**：
```yaml
ARC (AI2 Reasoning Challenge):
  - 科學問題推理
  - 25-shot
  - 難度：中等偏難

HellaSwag:
  - 常識推理
  - 10-shot
  - 完形填空風格

MMLU (Massive Multitask Language Understanding):
  - 57 個學科
  - 5-shot
  - 涵蓋 STEM、人文、社會科學

TruthfulQA:
  - 事實準確性
  - 0-shot
  - 避免常見謬誤

Winogrande:
  - 常識推理
  - 5-shot
  - 代詞消歧
```

**評估腳本**：
```python
from lm_eval import evaluator, tasks

# 評估模型
results = evaluator.simple_evaluate(
    model="hf-causal",
    model_args="pretrained=gpt2",
    tasks=["arc_challenge", "hellaswag", "mmlu", "truthfulqa_mc"],
    num_fewshot=5,
    batch_size=8
)

# 查看結果
for task, result in results["results"].items():
    print(f"{task}: {result['acc']:.4f}")
```

### 3.2 HELM (Holistic Evaluation of Language Models)

**更全面的評估框架**

**評估維度**：
1. **準確性** (Accuracy)
2. **校準度** (Calibration)
3. **魯棒性** (Robustness)
4. **公平性** (Fairness)
5. **偏見** (Bias)
6. **毒性** (Toxicity)
7. **效率** (Efficiency)

### 3.3 BIG-Bench

**204+ 個多樣化任務**

**任務類別**：
```yaml
語言理解:
  - 閱讀理解
  - 語言學任務
  - 翻譯

推理:
  - 邏輯推理
  - 數學推理
  - 常識推理

知識:
  - 事實性問答
  - 領域知識
  - 世界知識
```

---

## 4. 任務特定評估

### 4.1 問答系統 (QA)

**指標**：

```python
def evaluate_qa(predictions, references):
    """評估問答系統"""
    from sklearn.metrics import f1_score

    # Exact Match (EM)
    em = sum(p.strip() == r.strip() for p, r in zip(predictions, references)) / len(predictions)

    # F1 Score (token level)
    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        common = set(pred_tokens) & set(ref_tokens)
        if len(common) == 0:
            f1_scores.append(0)
        else:
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)

    return {
        'exact_match': em,
        'f1': sum(f1_scores) / len(f1_scores)
    }
```

### 4.2 程式碼生成

**HumanEval 基準**：

```python
# 評估程式碼生成
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

def generate_code_solutions(model, problems):
    """生成程式碼解決方案"""
    solutions = []

    for task_id, problem in problems.items():
        prompt = problem['prompt']

        # 生成程式碼
        generated = model.generate(prompt, max_tokens=512)

        solutions.append({
            'task_id': task_id,
            'completion': generated
        })

    return solutions

# 評估
problems = read_problems()
solutions = generate_code_solutions(model, problems)

# 寫入並評估
write_jsonl("samples.jsonl", solutions)
results = evaluate_functional_correctness("samples.jsonl")

print(f"Pass@1: {results['pass@1']:.2%}")
print(f"Pass@10: {results['pass@10']:.2%}")
```

### 4.3 長文字理解

**LongBench 評估**：

```python
# 評估長上下文能力
tasks = {
    'narrative_qa': '書籍問答',
    'qasper': '學術論文問答',
    'multifieldqa': '多領域問答',
    'hotpotqa': '多跳推理',
}

def evaluate_long_context(model, max_length=32000):
    """評估長上下文性能"""
    results = {}

    for task_name, task_desc in tasks.items():
        # 載入資料
        dataset = load_dataset(f"longbench/{task_name}")

        correct = 0
        total = 0

        for example in dataset:
            context = example['context'][:max_length]
            question = example['question']
            answer = example['answer']

            # 生成答案
            pred = model.generate(f"{context}\n\nQ: {question}\nA:")

            # 評估
            if pred.strip().lower() == answer.strip().lower():
                correct += 1
            total += 1

        results[task_name] = correct / total

    return results
```

---

## 5. 人類評估方法

### 5.1 Pairwise Comparison (成對比較)

**實施步驟**：

```python
import random

def create_comparison_task(prompt, response_a, response_b):
    """建立比較任務"""
    # 隨機順序避免位置偏見
    if random.random() < 0.5:
        response_a, response_b = response_b, response_a
        true_order = 'swapped'
    else:
        true_order = 'original'

    return {
        'prompt': prompt,
        'response_a': response_a,
        'response_b': response_b,
        'true_order': true_order
    }

def calculate_elo_rating(ratings, k=32):
    """計算 Elo 評分"""
    # Elo rating system
    for match in comparisons:
        model_a = match['model_a']
        model_b = match['model_b']
        winner = match['winner']

        # 預期得分
        expected_a = 1 / (1 + 10**((ratings[model_b] - ratings[model_a]) / 400))
        expected_b = 1 - expected_a

        # 實際得分
        if winner == 'a':
            actual_a, actual_b = 1, 0
        elif winner == 'b':
            actual_a, actual_b = 0, 1
        else:
            actual_a, actual_b = 0.5, 0.5

        # 更新評分
        ratings[model_a] += k * (actual_a - expected_a)
        ratings[model_b] += k * (actual_b - expected_b)

    return ratings
```

### 5.2 評估維度設計

**Chatbot Arena 式評估**：

```python
evaluation_criteria = {
    'helpfulness': {
        'description': '回答是否有助於解決用戶問題',
        'scale': '1-5',
        'examples': {
            1: '完全無關或有害',
            3: '部分有用但不完整',
            5: '完美解決問題'
        }
    },
    'accuracy': {
        'description': '資訊的準確性',
        'scale': '1-5'
    },
    'clarity': {
        'description': '表達的清晰度',
        'scale': '1-5'
    },
    'safety': {
        'description': '是否包含有害內容',
        'scale': 'binary'
    }
}
```

### 5.3 MT-Bench (多輪對話評估)

```python
# MT-Bench: 80 個多輪對話問題
categories = [
    'writing', 'roleplay', 'extraction',
    'reasoning', 'math', 'coding',
    'knowledge', 'common-sense'
]

def mt_bench_evaluate(model, judge_model='gpt-4'):
    """使用 GPT-4 作為評判"""
    results = {}

    for category in categories:
        questions = load_mt_bench_questions(category)
        scores = []

        for q in questions:
            # 第一輪
            response_1 = model.generate(q['turn_1'])

            # 第二輪
            response_2 = model.generate(q['turn_2'], history=response_1)

            # 使用 GPT-4 評分
            score = judge_model.rate(
                question_1=q['turn_1'],
                answer_1=response_1,
                question_2=q['turn_2'],
                answer_2=response_2
            )

            scores.append(score)

        results[category] = sum(scores) / len(scores)

    return results
```

---

## 6. 實作範例

### 6.1 完整評估流程

```python
class LLMEvaluator:
    """LLM 綜合評估器"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_perplexity(self, test_data):
        """評估困惑度"""
        from tqdm import tqdm

        total_loss = 0
        total_tokens = 0

        for text in tqdm(test_data, desc="Calculating PPL"):
            encodings = self.tokenizer(text, return_tensors='pt')

            with torch.no_grad():
                outputs = self.model(**encodings, labels=encodings.input_ids)
                loss = outputs.loss

            total_loss += loss.item() * encodings.input_ids.size(1)
            total_tokens += encodings.input_ids.size(1)

        ppl = np.exp(total_loss / total_tokens)
        return ppl

    def evaluate_generation_quality(self, prompts, references):
        """評估生成品質"""
        from rouge import Rouge

        rouge = Rouge()
        generated_texts = []

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors='pt')
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(text)

        # ROUGE 分數
        scores = rouge.get_scores(generated_texts, references, avg=True)

        return {
            'rouge-1': scores['rouge-1']['f'],
            'rouge-2': scores['rouge-2']['f'],
            'rouge-l': scores['rouge-l']['f']
        }

    def evaluate_few_shot(self, task, num_shots=5):
        """Few-shot 評估"""
        from datasets import load_dataset

        dataset = load_dataset(task)

        # 構建 few-shot prompt
        examples = dataset['train'][:num_shots]
        test_examples = dataset['test'][:100]

        few_shot_prompt = "\n\n".join([
            f"Q: {ex['question']}\nA: {ex['answer']}"
            for ex in examples
        ])

        correct = 0
        for test_ex in test_examples:
            prompt = few_shot_prompt + f"\n\nQ: {test_ex['question']}\nA:"

            # 生成答案
            inputs = self.tokenizer(prompt, return_tensors='pt')
            outputs = self.model.generate(**inputs, max_new_tokens=50)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 檢查正確性
            if answer.strip().lower() == test_ex['answer'].strip().lower():
                correct += 1

        accuracy = correct / len(test_examples)
        return accuracy

    def comprehensive_eval(self):
        """綜合評估"""
        results = {}

        # 1. 困惑度
        test_texts = [...]  # 你的測試資料
        results['perplexity'] = self.evaluate_perplexity(test_texts)

        # 2. 生成品質
        prompts = [...]
        references = [...]
        results['generation'] = self.evaluate_generation_quality(prompts, references)

        # 3. Few-shot 任務
        results['arc'] = self.evaluate_few_shot('arc', num_shots=25)
        results['hellaswag'] = self.evaluate_few_shot('hellaswag', num_shots=10)

        return results

# 使用
evaluator = LLMEvaluator(model, tokenizer)
results = evaluator.comprehensive_eval()

print("評估結果:")
for metric, value in results.items():
    print(f"{metric}: {value}")
```

### 6.2 自動化評估管道

```python
# evaluation_pipeline.py

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime

class EvaluationPipeline:
    """自動化評估管道"""

    def __init__(self, model_path, output_dir='./eval_results'):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    def run_benchmark(self, benchmark_name):
        """運行基準測試"""
        if benchmark_name == 'mmlu':
            return self.eval_mmlu()
        elif benchmark_name == 'humaneval':
            return self.eval_humaneval()
        # ...更多基準

    def eval_mmlu(self):
        """評估 MMLU"""
        dataset = load_dataset('cais/mmlu', 'all')

        results_by_subject = {}

        for subject in dataset.keys():
            correct = 0
            total = 0

            for example in dataset[subject]['test']:
                # 構建選擇題 prompt
                question = example['question']
                choices = example['choices']
                answer = example['answer']

                prompt = f"{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nAnswer:"

                # 生成
                pred = self.generate(prompt)

                # 評估
                if pred.strip().upper() == answer:
                    correct += 1
                total += 1

            results_by_subject[subject] = correct / total

        avg_acc = sum(results_by_subject.values()) / len(results_by_subject)

        return {
            'average': avg_acc,
            'by_subject': results_by_subject
        }

    def generate_report(self, results):
        """生成評估報告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"{self.output_dir}/report_{timestamp}.json"

        report = {
            'timestamp': timestamp,
            'model': self.model.config._name_or_path,
            'results': results
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"報告已保存: {report_path}")
        return report_path

# 使用
pipeline = EvaluationPipeline('gpt2')
results = pipeline.run_benchmark('mmlu')
pipeline.generate_report(results)
```

---

## 總結

### 評估策略建議

**研發階段**：
```
✓ 困惑度：快速迭代
✓ 自動指標：BLEU, ROUGE
✓ 小規模基準：Arc, HellaSwag
```

**模型對比**：
```
✓ 通用基準：MMLU, BBH
✓ 任務特定：HumanEval, TruthfulQA
✓ 人類評估：Pairwise comparison
```

**生產部署**：
```
✓ A/B 測試：真實用戶反饋
✓ 業務指標：轉化率、滿意度
✓ 安全性監控：毒性、偏見檢測
```

### 關鍵要點

1. **沒有完美的指標** - 組合使用自動和人類評估
2. **任務相關性** - 選擇與應用場景匹配的評估
3. **持續評估** - 部署後持續監控
4. **成本平衡** - 人類評估昂貴但必要

### 資源

- **Leaderboards**: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- **HELM**: https://crfm.stanford.edu/helm/
- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness
- **MT-Bench**: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
