# 模型評估 (Evaluation)

## 概述

大型語言模型 (LLM) 的評估是一個多維度、多層次的複雜任務。與傳統機器學習模型不同，LLM 的能力涵蓋廣泛（從文本生成到推理），因此需要結合自動化指標、人類評估和任務特定基準來全面衡量模型性能。

**評估的主要目的**：
- 衡量模型在特定任務上的表現
- 比較不同模型或訓練方法的優劣
- 發現模型的能力邊界和弱點
- 指導模型改進和迭代方向
- 確保模型符合安全性和對齊要求

**評估的挑戰**：
- 自動化指標與人類判斷存在差距
- 不同任務需要不同的評估方法
- 評估成本高（尤其是人類評估）
- 評估基準可能存在數據洩漏風險
- 模型的創造性和多樣性難以量化

---

## 9.1 傳統指標 (Perplexity、BLEU) 與其限制

### 9.1.1 Perplexity (困惑度)

**定義**：
Perplexity 衡量語言模型預測文本序列的"困惑程度"，反映模型對測試數據的不確定性。

$$
\text{PPL}(X) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(x_i|x_{<i})\right)
$$

其中 $N$ 是 token 數量，$P(x_i|x_{<i})$ 是模型對第 $i$ 個 token 的預測概率。

**解釋**：
- PPL 越低，模型越"確定"，預測越準確
- PPL = 1 表示完美預測
- PPL = k 表示模型在每個位置平均有 k 種選擇的困惑度

**計算範例**：

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity(model, tokenizer, text):
    """計算給定文本的 Perplexity"""
    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    # 計算 log-likelihood
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss 是平均 negative log-likelihood
        neg_log_likelihood = outputs.loss

    # PPL = exp(average negative log-likelihood)
    ppl = torch.exp(neg_log_likelihood)
    return ppl.item()

# 範例使用
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

test_text = "The quick brown fox jumps over the lazy dog."
ppl = calculate_perplexity(model, tokenizer, test_text)
print(f"Perplexity: {ppl:.2f}")
```

**批次計算 Perplexity**：

```python
def calculate_perplexity_batch(model, tokenizer, texts, max_length=512):
    """批次計算多個文本的 Perplexity"""
    model.eval()
    total_loss = 0
    total_length = 0

    for text in texts:
        encodings = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encodings.input_ids

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            # loss 是平均到每個 token 的
            total_loss += outputs.loss.item() * input_ids.size(1)
            total_length += input_ids.size(1)

    avg_loss = total_loss / total_length
    ppl = torch.exp(torch.tensor(avg_loss))
    return ppl.item()

# 範例
test_texts = [
    "Artificial intelligence is transforming the world.",
    "Machine learning models require large datasets.",
    "Natural language processing enables human-computer interaction."
]
ppl = calculate_perplexity_batch(model, tokenizer, test_texts)
print(f"Average Perplexity: {ppl:.2f}")
```

**優點**：
- 易於計算，適合快速評估
- 適合預訓練階段監控模型收斂
- 與訓練目標（負對數似然）直接對應

**限制**：
- **不反映實際任務性能**：低 PPL 不等於高質量生成
- **對 tokenizer 敏感**：不同 tokenizer 的 PPL 不可直接比較
- **無法評估創造性**：只衡量"預測準確度"，不衡量生成質量
- **不適合開放式生成**：更適合語言建模，不適合對話或摘要

---

### 9.1.2 BLEU (Bilingual Evaluation Understudy)

**定義**：
BLEU 最初用於機器翻譯評估，通過計算生成文本與參考文本之間的 n-gram 重疊度來衡量質量。

$$
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N}w_n\log p_n\right)
$$

其中：
- $p_n$ 是 n-gram 精確度
- $w_n$ 是權重（通常均勻分配）
- BP (Brevity Penalty) 是長度懲罰，避免生成過短文本

**計算範例**：

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

def calculate_bleu(reference, candidate):
    """計算單個句子的 BLEU 分數"""
    # reference 應為 list of lists (支持多個參考譯文)
    # candidate 應為 list of tokens

    # Tokenize
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()

    # 計算 BLEU-4 (1-gram 到 4-gram)
    smoothie = SmoothingFunction().method4  # 平滑處理零計數
    bleu = sentence_bleu(
        reference_tokens,
        candidate_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),  # BLEU-4
        smoothing_function=smoothie
    )
    return bleu

# 範例
reference = "The cat is on the mat"
candidate = "The cat sits on the mat"
bleu = calculate_bleu(reference, candidate)
print(f"BLEU Score: {bleu:.4f}")
```

**批次 BLEU 評估**：

```python
def evaluate_model_bleu(model, tokenizer, test_pairs, max_length=50):
    """評估模型在測試集上的 BLEU 分數"""
    from tqdm import tqdm

    references = []
    candidates = []

    model.eval()
    for source, target in tqdm(test_pairs):
        # 生成翻譯
        inputs = tokenizer(source, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        references.append([target.split()])
        candidates.append(generated.split())

    # 計算 corpus-level BLEU
    bleu = corpus_bleu(references, candidates)
    return bleu

# 範例（假設有翻譯對）
test_pairs = [
    ("Hello world", "你好世界"),
    ("Good morning", "早上好"),
]
# bleu_score = evaluate_model_bleu(model, tokenizer, test_pairs)
```

**變體指標**：

```python
def calculate_multiple_bleu_scores(reference, candidate):
    """計算不同 n-gram 的 BLEU 分數"""
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    smoothie = SmoothingFunction().method4

    scores = {
        'BLEU-1': sentence_bleu(reference_tokens, candidate_tokens,
                                weights=(1, 0, 0, 0), smoothing_function=smoothie),
        'BLEU-2': sentence_bleu(reference_tokens, candidate_tokens,
                                weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
        'BLEU-3': sentence_bleu(reference_tokens, candidate_tokens,
                                weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie),
        'BLEU-4': sentence_bleu(reference_tokens, candidate_tokens,
                                weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie),
    }
    return scores

# 範例
scores = calculate_multiple_bleu_scores(
    "The quick brown fox jumps over the lazy dog",
    "The fast brown fox jumps over a lazy dog"
)
for name, score in scores.items():
    print(f"{name}: {score:.4f}")
```

**優點**：
- 易於計算，可重複
- 適合有明確參考答案的任務（翻譯、摘要）
- 被廣泛接受和使用

**限制**：
- **過度依賴詞匯重疊**：無法捕捉語義相似性
- **對同義詞不敏感**："fast" vs "quick" 會被視為不同
- **懲罰創造性表達**：與參考文本不同的合理表達會被低估
- **不適合開放式生成**：對話、故事生成等沒有唯一正確答案

---

### 9.1.3 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**定義**：
ROUGE 主要用於摘要評估，與 BLEU 類似但更關注召回率 (Recall)。

**主要變體**：
- **ROUGE-N**: N-gram 召回率
- **ROUGE-L**: 最長公共子序列 (LCS)
- **ROUGE-S**: Skip-bigram 重疊

```python
from rouge_score import rouge_scorer

def calculate_rouge(reference, candidate):
    """計算 ROUGE 分數"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)

    results = {}
    for key, value in scores.items():
        results[key] = {
            'precision': value.precision,
            'recall': value.recall,
            'fmeasure': value.fmeasure
        }
    return results

# 範例
reference = "The quick brown fox jumps over the lazy dog"
candidate = "The fast brown fox leaps over a lazy dog"
scores = calculate_rouge(reference, candidate)

for metric, values in scores.items():
    print(f"{metric}:")
    print(f"  Precision: {values['precision']:.4f}")
    print(f"  Recall: {values['recall']:.4f}")
    print(f"  F1: {values['fmeasure']:.4f}")
```

**摘要評估**：

```python
def evaluate_summarization(model, tokenizer, articles, references):
    """評估摘要模型"""
    from tqdm import tqdm

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    model.eval()
    for article, reference in tqdm(zip(articles, references)):
        # 生成摘要
        inputs = tokenizer(article, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_length=150,
                min_length=40,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # 計算 ROUGE
        scores = scorer.score(reference, summary)
        for key in all_scores:
            all_scores[key].append(scores[key].fmeasure)

    # 計算平均分數
    avg_scores = {key: sum(values)/len(values) for key, values in all_scores.items()}
    return avg_scores

# 範例輸出
# avg_scores = evaluate_summarization(model, tokenizer, articles, references)
# print(f"ROUGE-1: {avg_scores['rouge1']:.4f}")
# print(f"ROUGE-2: {avg_scores['rouge2']:.4f}")
# print(f"ROUGE-L: {avg_scores['rougeL']:.4f}")
```

---

### 9.1.4 其他傳統指標

**METEOR (Metric for Evaluation of Translation with Explicit ORdering)**：

```python
from nltk.translate.meteor_score import meteor_score

def calculate_meteor(reference, candidate):
    """計算 METEOR 分數（考慮同義詞和詞幹）"""
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    return meteor_score([reference_tokens], candidate_tokens)

# 範例
reference = "The cat is on the mat"
candidate = "The feline sits on the rug"
meteor = calculate_meteor(reference, candidate)
print(f"METEOR: {meteor:.4f}")
```

**BERTScore (語義相似度)**：

```python
from bert_score import score

def calculate_bertscore(references, candidates, lang="en"):
    """計算 BERTScore（基於 BERT 嵌入的語義相似度）"""
    P, R, F1 = score(
        candidates,
        references,
        lang=lang,
        verbose=True,
        model_type="bert-base-uncased"
    )
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

# 範例
references = ["The cat is on the mat", "Hello world"]
candidates = ["The feline sits on the rug", "Hi there"]
scores = calculate_bertscore(references, candidates)
print(f"BERTScore F1: {scores['f1']:.4f}")
```

**傳統指標的總結比較**：

| 指標 | 適用任務 | 主要優點 | 主要限制 |
|------|----------|----------|----------|
| Perplexity | 語言建模 | 易計算、訓練監控 | 不反映生成質量 |
| BLEU | 翻譯、生成 | 可重複、廣泛接受 | 過度依賴詞匯重疊 |
| ROUGE | 摘要 | 注重召回率 | 不捕捉語義 |
| METEOR | 翻譯 | 考慮同義詞 | 計算複雜 |
| BERTScore | 通用 | 語義感知 | 計算成本高 |

---

## 9.2 通用評估基準 (Open LLM Leaderboard、BIG-Bench)

### 9.2.1 Open LLM Leaderboard

**概述**：
Hugging Face 的 Open LLM Leaderboard 是評估開源 LLM 的權威平台，提供標準化、可複製的評估流程。

**評估任務**：

1. **AI2 Reasoning Challenge (ARC)**
   - 科學推理多選題
   - 難度：Challenge 集 (困難題目)
   - 指標：準確率

2. **HellaSwag**
   - 常識推理和情境完成
   - 測試模型的常識理解能力
   - 指標：準確率（規範化）

3. **MMLU (Massive Multitask Language Understanding)**
   - 57 個學科的多選題
   - 涵蓋 STEM、人文、社會科學
   - 指標：準確率

4. **TruthfulQA**
   - 測試模型生成真實陳述的能力
   - 避免常見謬誤和錯誤信息
   - 指標：真實性分數

5. **Winogrande**
   - 常識推理（代詞消歧）
   - 測試語言理解能力
   - 指標：準確率

6. **GSM8K**
   - 小學數學應用題
   - 測試數學推理能力
   - 指標：準確率

**評估流程**：

```python
# 使用 lm-evaluation-harness 進行評估
from lm_eval import evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_on_leaderboard_tasks(model_name):
    """在 Open LLM Leaderboard 任務上評估模型"""
    # 加載模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 定義評估任務
    tasks = [
        "arc_challenge",
        "hellaswag",
        "mmlu",
        "truthfulqa_mc",
        "winogrande",
        "gsm8k"
    ]

    # 運行評估
    results = evaluator.simple_evaluate(
        model="hf-causal",
        model_args=f"pretrained={model_name}",
        tasks=tasks,
        num_fewshot=0,  # 0-shot 評估
        batch_size=8,
    )

    return results

# 範例（需要安裝 lm-evaluation-harness）
# results = evaluate_on_leaderboard_tasks("meta-llama/Llama-2-7b-hf")
# for task, metrics in results['results'].items():
#     print(f"{task}: {metrics['acc']:.4f}")
```

**自定義評估腳本**：

```python
import json
from tqdm import tqdm

def evaluate_mmlu(model, tokenizer, dataset_path, num_shots=5):
    """評估 MMLU 任務"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    correct = 0
    total = 0

    for item in tqdm(data):
        question = item['question']
        choices = item['choices']
        answer_idx = item['answer']

        # 構建 prompt
        prompt = f"Question: {question}\nChoices:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"

        # 生成答案
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False
            )

        predicted = tokenizer.decode(outputs[0][-1], skip_special_tokens=True).strip()

        # 檢查答案
        if predicted.upper() == chr(65 + answer_idx):
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy

# 範例
# accuracy = evaluate_mmlu(model, tokenizer, "mmlu_test.json")
# print(f"MMLU Accuracy: {accuracy:.4f}")
```

**Leaderboard 提交**：

```bash
# 安裝 lm-evaluation-harness
pip install lm-eval

# 運行完整評估
lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks arc_challenge,hellaswag,mmlu,truthfulqa_mc,winogrande,gsm8k \
    --device cuda:0 \
    --batch_size 8 \
    --output_path ./results/

# 查看結果
cat results/results.json
```

---

### 9.2.2 BIG-Bench (Beyond the Imitation Game Benchmark)

**概述**：
BIG-Bench 是一個包含 200+ 任務的大規模基準，旨在測試 LLM 在多種能力上的表現。

**任務類別**：

1. **語言理解**：語法、語義、語用
2. **知識推理**：常識、科學、數學
3. **符號操作**：邏輯、代數、編程
4. **世界知識**：歷史、地理、文化
5. **社會推理**：心理、倫理、社會規範

**BIG-Bench Lite**：
精選的 24 個代表性任務，適合快速評估。

```python
from bigbench import api, tasks

def evaluate_bigbench_lite(model, tokenizer):
    """評估 BIG-Bench Lite 任務"""
    # BIG-Bench Lite 任務列表
    lite_tasks = [
        'causal_judgement',
        'date_understanding',
        'disambiguation_qa',
        'geometric_shapes',
        'logical_deduction_five_objects',
        'movie_recommendation',
        'navigate',
        'ruin_names',
        'sports_understanding',
        'tracking_shuffled_objects_five_objects',
        'web_of_lies',
        # ... 更多任務
    ]

    results = {}
    for task_name in lite_tasks:
        task = tasks.get_task(task_name)
        score = api.evaluate_task(
            task=task,
            model=model,
            tokenizer=tokenizer,
            num_examples=100
        )
        results[task_name] = score

    return results

# 範例
# results = evaluate_bigbench_lite(model, tokenizer)
# avg_score = sum(results.values()) / len(results)
# print(f"BIG-Bench Lite Average: {avg_score:.4f}")
```

**任務範例 - 因果判斷**：

```python
def evaluate_causal_judgement(model, tokenizer):
    """評估因果推理能力"""
    examples = [
        {
            "scenario": "John forgot to water his plants. Two weeks later, they died.",
            "question": "Did John's forgetting cause the plants to die?",
            "answer": "Yes"
        },
        {
            "scenario": "Mary wore a red shirt. It rained that day.",
            "question": "Did Mary's red shirt cause the rain?",
            "answer": "No"
        }
    ]

    correct = 0
    for ex in examples:
        prompt = f"Scenario: {ex['scenario']}\nQuestion: {ex['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if ex['answer'].lower() in response.lower():
            correct += 1

    return correct / len(examples)

# 範例
# accuracy = evaluate_causal_judgement(model, tokenizer)
# print(f"Causal Judgement Accuracy: {accuracy:.4f}")
```

---

### 9.2.3 其他重要基準

**HELM (Holistic Evaluation of Language Models)**：

```python
# HELM 評估七個核心場景
# 1. Question answering
# 2. Information retrieval
# 3. Summarization
# 4. Sentiment analysis
# 5. Toxicity detection
# 6. Reasoning
# 7. Code generation

def evaluate_helm_scenario(model, tokenizer, scenario="question_answering"):
    """評估 HELM 場景"""
    # 加載場景數據
    from helm.benchmark.scenarios import get_scenario

    scenario_obj = get_scenario(scenario)
    instances = scenario_obj.get_instances()

    results = []
    for instance in instances:
        # 運行推理
        prompt = instance.input.text
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 評估指標（依場景而定）
        score = scenario_obj.evaluate(response, instance.references)
        results.append(score)

    return sum(results) / len(results)
```

**SuperGLUE**：

```python
from datasets import load_dataset

def evaluate_superglue(model, tokenizer, task="boolq"):
    """評估 SuperGLUE 任務"""
    dataset = load_dataset("super_glue", task)
    test_data = dataset['validation']

    correct = 0
    total = 0

    for example in test_data:
        if task == "boolq":
            prompt = f"Passage: {example['passage']}\nQuestion: {example['question']}\nAnswer (True/False):"
            true_answer = "True" if example['label'] == 1 else "False"

        # 生成預測
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5)

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if true_answer.lower() in prediction.lower():
            correct += 1
        total += 1

    return correct / total

# 範例
# accuracy = evaluate_superglue(model, tokenizer, task="boolq")
# print(f"BoolQ Accuracy: {accuracy:.4f}")
```

---

## 9.3 人類評估 (對話測試、Elo Rating)

### 9.3.1 人類評估的重要性

**為何需要人類評估**：
- 自動化指標無法捕捉細微的語義、流暢性、創造性
- 開放式生成任務（對話、創作）缺乏標準答案
- 安全性、倫理性、文化適應性需要人類判斷
- 用戶體驗最終由人類決定

**挑戰**：
- 成本高、耗時長
- 評估者間可能存在分歧
- 難以標準化和規模化
- 可能存在偏見

---

### 9.3.2 對話測試 (Chatbot Arena)

**Chatbot Arena 模式**：
- 用戶與兩個匿名模型對話
- 用戶選擇更好的回答
- 基於對戰結果計算 Elo 評分

**評估維度**：

```python
class ConversationEvaluator:
    """對話質量評估器"""

    def __init__(self):
        self.criteria = {
            'relevance': '回答是否切題',
            'coherence': '邏輯是否連貫',
            'fluency': '語言是否流暢',
            'informativeness': '信息量是否豐富',
            'safety': '是否安全無害',
            'engagement': '是否引人入勝'
        }

    def evaluate_response(self, prompt, response_a, response_b):
        """人類評估兩個回答"""
        print(f"Prompt: {prompt}\n")
        print(f"Response A: {response_a}\n")
        print(f"Response B: {response_b}\n")

        scores = {}
        for criterion, description in self.criteria.items():
            print(f"\n{criterion} ({description}):")
            print("1 - A 明顯更好")
            print("2 - A 略好")
            print("3 - 平手")
            print("4 - B 略好")
            print("5 - B 明顯更好")

            score = int(input("評分: "))
            scores[criterion] = score

        return scores

    def aggregate_scores(self, all_scores):
        """聚合多個評估者的評分"""
        aggregated = {}
        for criterion in self.criteria:
            criterion_scores = [s[criterion] for s in all_scores]
            aggregated[criterion] = {
                'mean': sum(criterion_scores) / len(criterion_scores),
                'std': self._std(criterion_scores),
                'agreement': self._agreement(criterion_scores)
            }
        return aggregated

    def _std(self, values):
        """計算標準差"""
        import math
        mean = sum(values) / len(values)
        return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))

    def _agreement(self, values):
        """計算評估者一致性（簡化版）"""
        from collections import Counter
        counts = Counter(values)
        most_common_count = counts.most_common(1)[0][1]
        return most_common_count / len(values)

# 範例使用
evaluator = ConversationEvaluator()

# 多輪對話評估
conversations = [
    {
        "prompt": "What's the capital of France?",
        "response_a": "Paris",
        "response_b": "The capital of France is Paris, a beautiful city known for its art and culture."
    }
]

# scores = evaluator.evaluate_response(**conversations[0])
```

**批次人類評估流程**：

```python
import json
from datetime import datetime

class HumanEvaluationPlatform:
    """人類評估平台"""

    def __init__(self):
        self.evaluations = []

    def create_evaluation_task(self, model_a, model_b, test_prompts):
        """創建評估任務"""
        task = {
            'task_id': f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'model_a': model_a,
            'model_b': model_b,
            'prompts': test_prompts,
            'evaluations': []
        }
        return task

    def collect_human_feedback(self, task, evaluator_id):
        """收集人類反饋"""
        for i, prompt in enumerate(task['prompts']):
            print(f"\n[{i+1}/{len(task['prompts'])}] {prompt}")

            # 顯示兩個模型的回答（匿名）
            print("\nResponse A:")
            print(task['responses_a'][i])
            print("\nResponse B:")
            print(task['responses_b'][i])

            # 收集評分
            winner = input("\n選擇更好的回答 (A/B/Tie): ").upper()
            confidence = int(input("信心水平 (1-5): "))

            task['evaluations'].append({
                'prompt_idx': i,
                'winner': winner,
                'confidence': confidence,
                'evaluator_id': evaluator_id,
                'timestamp': datetime.now().isoformat()
            })

        return task

    def calculate_win_rate(self, task):
        """計算勝率"""
        wins_a = sum(1 for e in task['evaluations'] if e['winner'] == 'A')
        wins_b = sum(1 for e in task['evaluations'] if e['winner'] == 'B')
        ties = sum(1 for e in task['evaluations'] if e['winner'] == 'TIE')

        total = len(task['evaluations'])

        return {
            'model_a_win_rate': wins_a / total,
            'model_b_win_rate': wins_b / total,
            'tie_rate': ties / total,
            'confidence_weighted_score': self._weighted_score(task)
        }

    def _weighted_score(self, task):
        """加權評分（考慮信心水平）"""
        score_a = 0
        score_b = 0
        total_weight = 0

        for e in task['evaluations']:
            weight = e['confidence']
            if e['winner'] == 'A':
                score_a += weight
            elif e['winner'] == 'B':
                score_b += weight
            else:  # Tie
                score_a += weight * 0.5
                score_b += weight * 0.5
            total_weight += weight

        return {
            'model_a': score_a / total_weight,
            'model_b': score_b / total_weight
        }

# 範例使用
platform = HumanEvaluationPlatform()

test_prompts = [
    "Explain quantum computing to a 10-year-old.",
    "Write a creative short story about a time-traveling cat.",
    "What are the ethical implications of AI?"
]

# task = platform.create_evaluation_task("gpt-3.5-turbo", "claude-2", test_prompts)
# task = platform.collect_human_feedback(task, evaluator_id="evaluator_001")
# results = platform.calculate_win_rate(task)
```

---

### 9.3.3 Elo Rating 系統

**Elo Rating 原理**：
- 源自國際象棋評分系統
- 通過對戰結果動態更新評分
- 預期勝率基於評分差距

**計算公式**：

$$
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
$$

$$
R_A^{\text{new}} = R_A + K \cdot (S_A - E_A)
$$

其中：
- $R_A$, $R_B$ 是當前 Elo 評分
- $E_A$ 是 A 的預期勝率
- $S_A$ 是實際結果（1=勝，0.5=平，0=負）
- $K$ 是更新係數（通常 32）

**實現 Elo 評分系統**：

```python
import math
from collections import defaultdict

class EloRatingSystem:
    """Elo 評分系統"""

    def __init__(self, k=32, initial_rating=1500):
        self.k = k
        self.initial_rating = initial_rating
        self.ratings = defaultdict(lambda: initial_rating)
        self.match_history = []

    def expected_score(self, rating_a, rating_b):
        """計算預期勝率"""
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))

    def update_ratings(self, model_a, model_b, result):
        """
        更新 Elo 評分
        result: 1 (A 勝), 0.5 (平手), 0 (B 勝)
        """
        rating_a = self.ratings[model_a]
        rating_b = self.ratings[model_b]

        # 計算預期勝率
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)

        # 更新評分
        new_rating_a = rating_a + self.k * (result - expected_a)
        new_rating_b = rating_b + self.k * ((1 - result) - expected_b)

        self.ratings[model_a] = new_rating_a
        self.ratings[model_b] = new_rating_b

        # 記錄歷史
        self.match_history.append({
            'model_a': model_a,
            'model_b': model_b,
            'result': result,
            'rating_a_before': rating_a,
            'rating_b_before': rating_b,
            'rating_a_after': new_rating_a,
            'rating_b_after': new_rating_b
        })

        return new_rating_a, new_rating_b

    def get_leaderboard(self):
        """獲取排行榜"""
        sorted_models = sorted(
            self.ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_models

    def simulate_matches(self, models, num_rounds=100):
        """模擬多輪對戰"""
        import random

        for _ in range(num_rounds):
            # 隨機選擇兩個模型
            model_a, model_b = random.sample(models, 2)

            # 模擬人類評估結果（實際應用中應由真人評估）
            result = self._simulate_match_result(model_a, model_b)

            # 更新評分
            self.update_ratings(model_a, model_b, result)

    def _simulate_match_result(self, model_a, model_b):
        """模擬對戰結果（僅用於演示）"""
        import random
        # 實際應用中，這應該是真實的人類評估結果
        return random.choice([0, 0.5, 1])

# 範例使用
elo = EloRatingSystem(k=32, initial_rating=1500)

# 模擬對戰
models = ["GPT-4", "Claude-3", "Llama-2-70B", "Mistral-7B", "Gemini-Pro"]

# 記錄對戰結果
matches = [
    ("GPT-4", "Claude-3", 0.5),  # 平手
    ("GPT-4", "Llama-2-70B", 1),  # GPT-4 勝
    ("Claude-3", "Mistral-7B", 1),  # Claude-3 勝
    ("Llama-2-70B", "Gemini-Pro", 0),  # Gemini-Pro 勝
    ("GPT-4", "Gemini-Pro", 1),  # GPT-4 勝
]

for model_a, model_b, result in matches:
    new_a, new_b = elo.update_ratings(model_a, model_b, result)
    print(f"{model_a} vs {model_b}: {result}")
    print(f"  {model_a}: {new_a:.1f}, {model_b}: {new_b:.1f}")

# 顯示排行榜
print("\n=== Leaderboard ===")
for rank, (model, rating) in enumerate(elo.get_leaderboard(), 1):
    print(f"{rank}. {model}: {rating:.1f}")
```

**LMSYS Chatbot Arena 實現**：

```python
class ChatbotArena:
    """Chatbot Arena 風格的評估平台"""

    def __init__(self):
        self.elo_system = EloRatingSystem(k=32, initial_rating=1000)
        self.battles = []

    def blind_battle(self, user_prompt, model_a_name, model_b_name,
                     model_a_response, model_b_response):
        """盲測對戰"""
        import random

        # 隨機化順序（避免位置偏見）
        if random.random() > 0.5:
            shown_first = ('A', model_a_name, model_a_response)
            shown_second = ('B', model_b_name, model_b_response)
        else:
            shown_first = ('B', model_b_name, model_b_response)
            shown_second = ('A', model_a_name, model_a_response)

        print(f"Prompt: {user_prompt}\n")
        print(f"Model {shown_first[0]}: {shown_first[2]}\n")
        print(f"Model {shown_second[0]}: {shown_second[2]}\n")

        # 用戶投票
        vote = input("Which is better? (A/B/Tie): ").upper()

        # 轉換為實際模型名稱
        if vote == shown_first[0]:
            winner = shown_first[1]
            result = 1 if shown_first[1] == model_a_name else 0
        elif vote == shown_second[0]:
            winner = shown_second[1]
            result = 0 if shown_second[1] == model_a_name else 1
        else:
            winner = "Tie"
            result = 0.5

        # 更新 Elo 評分
        self.elo_system.update_ratings(model_a_name, model_b_name, result)

        # 記錄對戰
        self.battles.append({
            'prompt': user_prompt,
            'model_a': model_a_name,
            'model_b': model_b_name,
            'winner': winner,
            'result': result
        })

        return winner

    def get_current_rankings(self):
        """獲取當前排名"""
        return self.elo_system.get_leaderboard()

    def statistics(self):
        """統計數據"""
        total_battles = len(self.battles)
        model_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0})

        for battle in self.battles:
            if battle['result'] == 1:
                model_stats[battle['model_a']]['wins'] += 1
                model_stats[battle['model_b']]['losses'] += 1
            elif battle['result'] == 0:
                model_stats[battle['model_a']]['losses'] += 1
                model_stats[battle['model_b']]['wins'] += 1
            else:
                model_stats[battle['model_a']]['ties'] += 1
                model_stats[battle['model_b']]['ties'] += 1

        return {
            'total_battles': total_battles,
            'model_stats': dict(model_stats)
        }

# 範例使用
arena = ChatbotArena()

# 模擬對戰
# winner = arena.blind_battle(
#     user_prompt="Explain machine learning",
#     model_a_name="GPT-4",
#     model_b_name="Claude-3",
#     model_a_response="Machine learning is...",
#     model_b_response="Machine learning enables..."
# )

# 查看排名
# rankings = arena.get_current_rankings()
# for rank, (model, rating) in enumerate(rankings, 1):
#     print(f"{rank}. {model}: {rating:.1f}")
```

**評估者間一致性 (Inter-Annotator Agreement)**：

```python
def calculate_inter_annotator_agreement(annotations):
    """
    計算評估者間一致性 (Fleiss' Kappa)
    annotations: list of dicts, each with evaluator votes
    """
    from sklearn.metrics import cohen_kappa_score
    import numpy as np

    # 簡化版：計算兩兩評估者的 Cohen's Kappa 平均值
    evaluators = list(annotations[0].keys())
    kappa_scores = []

    for i in range(len(evaluators)):
        for j in range(i+1, len(evaluators)):
            eval_i = [ann[evaluators[i]] for ann in annotations]
            eval_j = [ann[evaluators[j]] for ann in annotations]
            kappa = cohen_kappa_score(eval_i, eval_j)
            kappa_scores.append(kappa)

    avg_kappa = np.mean(kappa_scores)

    # Kappa 解釋
    if avg_kappa < 0:
        agreement = "No agreement"
    elif avg_kappa < 0.20:
        agreement = "Slight agreement"
    elif avg_kappa < 0.40:
        agreement = "Fair agreement"
    elif avg_kappa < 0.60:
        agreement = "Moderate agreement"
    elif avg_kappa < 0.80:
        agreement = "Substantial agreement"
    else:
        agreement = "Almost perfect agreement"

    return {
        'kappa': avg_kappa,
        'interpretation': agreement
    }

# 範例
annotations = [
    {'evaluator_1': 'A', 'evaluator_2': 'A', 'evaluator_3': 'B'},
    {'evaluator_1': 'B', 'evaluator_2': 'B', 'evaluator_3': 'B'},
    {'evaluator_1': 'A', 'evaluator_2': 'A', 'evaluator_3': 'A'},
]

# result = calculate_inter_annotator_agreement(annotations)
# print(f"Kappa: {result['kappa']:.3f} ({result['interpretation']})")
```

---

## 9.4 任務特定基準 (問答、翻譯、摘要、領域特化 QA)

### 9.4.1 問答 (Question Answering)

**SQuAD (Stanford Question Answering Dataset)**：

```python
from datasets import load_dataset
from transformers import pipeline

def evaluate_squad(model, tokenizer, num_samples=100):
    """評估 SQuAD 問答任務"""
    dataset = load_dataset("squad", split="validation")

    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    exact_match = 0
    f1_scores = []

    for i, example in enumerate(dataset):
        if i >= num_samples:
            break

        context = example['context']
        question = example['question']
        true_answer = example['answers']['text'][0]

        # 預測答案
        result = qa_pipeline(question=question, context=context)
        predicted_answer = result['answer']

        # 計算 Exact Match
        if predicted_answer.strip().lower() == true_answer.strip().lower():
            exact_match += 1

        # 計算 F1
        f1 = compute_f1(predicted_answer, true_answer)
        f1_scores.append(f1)

    return {
        'exact_match': exact_match / num_samples,
        'f1': sum(f1_scores) / len(f1_scores)
    }

def compute_f1(prediction, ground_truth):
    """計算 F1 分數"""
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()

    common = set(pred_tokens) & set(truth_tokens)

    if len(common) == 0:
        return 0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# 範例
# results = evaluate_squad(model, tokenizer, num_samples=100)
# print(f"Exact Match: {results['exact_match']:.4f}")
# print(f"F1 Score: {results['f1']:.4f}")
```

**Natural Questions (Google)**：

```python
def evaluate_natural_questions(model, tokenizer, dataset_path):
    """評估 Natural Questions 數據集"""
    import json

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    correct_short = 0
    correct_long = 0
    total = 0

    for item in data:
        question = item['question']
        short_answer = item['short_answer']
        long_answer = item['long_answer']

        # 生成答案
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)

        predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 檢查短答案
        if short_answer and short_answer.lower() in predicted.lower():
            correct_short += 1

        # 檢查長答案
        if long_answer and any(word in predicted.lower() for word in long_answer.lower().split()):
            correct_long += 1

        total += 1

    return {
        'short_answer_accuracy': correct_short / total,
        'long_answer_recall': correct_long / total
    }
```

---

### 9.4.2 翻譯 (Machine Translation)

**WMT (Workshop on Machine Translation)**：

```python
def evaluate_translation(model, tokenizer, src_lang="en", tgt_lang="de"):
    """評估機器翻譯"""
    from datasets import load_dataset
    from sacrebleu import corpus_bleu

    dataset = load_dataset("wmt14", f"{src_lang}-{tgt_lang}", split="test")

    references = []
    hypotheses = []

    for example in dataset:
        source = example['translation'][src_lang]
        reference = example['translation'][tgt_lang]

        # 翻譯
        prompt = f"Translate from {src_lang} to {tgt_lang}: {source}\nTranslation:"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)

        hypothesis = tokenizer.decode(outputs[0], skip_special_tokens=True)

        references.append([reference])
        hypotheses.append(hypothesis)

    # 計算 BLEU
    bleu = corpus_bleu(hypotheses, references)

    return {
        'bleu': bleu.score,
        'bp': bleu.bp,
        'precisions': bleu.precisions
    }

# 範例
# results = evaluate_translation(model, tokenizer, src_lang="en", tgt_lang="de")
# print(f"BLEU Score: {results['bleu']:.2f}")
```

**FLORES (Facebook Low Resource Translation)**：

```python
def evaluate_flores(model, tokenizer, lang_pair=("eng", "fra")):
    """評估低資源語言翻譯"""
    from datasets import load_dataset

    dataset = load_dataset("facebook/flores", f"{lang_pair[0]}_Latn-{lang_pair[1]}_Latn")

    bleu_scores = []

    for example in dataset['devtest']:
        source = example['sentence_eng_Latn']
        reference = example['sentence_fra_Latn']

        # 生成翻譯
        prompt = f"Translate to French: {source}"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)

        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 計算 BLEU
        bleu = sentence_bleu([reference.split()], translation.split())
        bleu_scores.append(bleu)

    return sum(bleu_scores) / len(bleu_scores)
```

---

### 9.4.3 摘要 (Summarization)

**CNN/DailyMail**：

```python
def evaluate_summarization_cnn_dm(model, tokenizer, num_samples=100):
    """評估 CNN/DailyMail 摘要任務"""
    from datasets import load_dataset
    from rouge_score import rouge_scorer

    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for i, example in enumerate(dataset):
        if i >= num_samples:
            break

        article = example['article']
        reference_summary = example['highlights']

        # 生成摘要
        prompt = f"Summarize the following article:\n\n{article}\n\nSummary:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_length=150,
                min_length=40,
                num_beams=4,
                length_penalty=2.0
            )

        generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # 計算 ROUGE
        scores = scorer.score(reference_summary, generated_summary)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    return {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores),
        'rouge2': sum(rouge2_scores) / len(rouge2_scores),
        'rougeL': sum(rougeL_scores) / len(rougeL_scores)
    }

# 範例
# results = evaluate_summarization_cnn_dm(model, tokenizer, num_samples=100)
# print(f"ROUGE-1: {results['rouge1']:.4f}")
# print(f"ROUGE-2: {results['rouge2']:.4f}")
# print(f"ROUGE-L: {results['rougeL']:.4f}")
```

**XSum (Extreme Summarization)**：

```python
def evaluate_xsum(model, tokenizer, num_samples=50):
    """評估 XSum 極端摘要任務"""
    from datasets import load_dataset

    dataset = load_dataset("xsum", split="test")

    rouge_scores = []

    for i, example in enumerate(dataset):
        if i >= num_samples:
            break

        document = example['document']
        reference = example['summary']

        # 生成單句摘要
        prompt = f"Summarize in one sentence:\n\n{document}\n\nSummary:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50, num_beams=4)

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 計算 ROUGE
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        score = scorer.score(reference, summary)['rougeL'].fmeasure
        rouge_scores.append(score)

    return sum(rouge_scores) / len(rouge_scores)
```

---

### 9.4.4 領域特化 QA (醫療、法律、金融)

**醫療領域 - MedQA**：

```python
def evaluate_medqa(model, tokenizer):
    """評估醫療問答能力"""
    from datasets import load_dataset

    dataset = load_dataset("bigbio/med_qa", split="test")

    correct = 0
    total = 0

    for example in dataset:
        question = example['question']
        options = example['options']
        answer_idx = example['answer_idx']

        # 構建 prompt
        prompt = f"Medical Question: {question}\n\nOptions:\n"
        for i, opt in enumerate(options):
            prompt += f"{chr(65+i)}. {opt}\n"
        prompt += "\nAnswer (letter only):"

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5)

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # 檢查答案
        if prediction.upper() == chr(65 + answer_idx):
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy

# 範例
# accuracy = evaluate_medqa(model, tokenizer)
# print(f"MedQA Accuracy: {accuracy:.4f}")
```

**法律領域 - LegalBench**：

```python
def evaluate_legal_reasoning(model, tokenizer):
    """評估法律推理能力"""

    test_cases = [
        {
            "scenario": "A contract was signed but one party was under duress.",
            "question": "Is the contract valid?",
            "answer": "No",
            "explanation": "Contracts signed under duress are voidable."
        },
        {
            "scenario": "An individual was arrested without a warrant during a felony in progress.",
            "question": "Was the arrest legal?",
            "answer": "Yes",
            "explanation": "Warrantless arrests are permitted during felonies in progress."
        }
    ]

    correct = 0

    for case in test_cases:
        prompt = f"Legal Scenario: {case['scenario']}\nQuestion: {case['question']}\nAnswer (Yes/No):"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if case['answer'].lower() in response.lower():
            correct += 1

    return correct / len(test_cases)

# 範例
# accuracy = evaluate_legal_reasoning(model, tokenizer)
# print(f"Legal Reasoning Accuracy: {accuracy:.4f}")
```

**金融領域 - FinQA**：

```python
def evaluate_finqa(model, tokenizer):
    """評估金融問答和數值推理"""
    from datasets import load_dataset

    dataset = load_dataset("dreamerdeo/finqa", split="test")

    correct = 0
    total = 0

    for example in dataset:
        context = example['pre_text'] + example['post_text']
        question = example['question']
        answer = example['answer']

        # 構建 prompt
        prompt = f"Financial Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 檢查數值答案
        if str(answer) in prediction:
            correct += 1
        total += 1

        if total >= 100:  # 限制測試樣本數
            break

    return correct / total

# 範例
# accuracy = evaluate_finqa(model, tokenizer)
# print(f"FinQA Accuracy: {accuracy:.4f}")
```

---

## 9.5 評估最佳實踐與完整流程

### 9.5.1 完整評估流程

```python
class ComprehensiveEvaluator:
    """綜合評估器"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}

    def run_all_evaluations(self):
        """運行所有評估"""
        print("開始綜合評估...")

        # 1. 基礎能力評估
        print("\n[1/5] 基礎能力評估...")
        self.results['perplexity'] = self.evaluate_perplexity()

        # 2. 通用基準評估
        print("\n[2/5] 通用基準評估...")
        self.results['mmlu'] = self.evaluate_mmlu_sample()
        self.results['hellaswag'] = self.evaluate_hellaswag_sample()

        # 3. 任務特定評估
        print("\n[3/5] 任務特定評估...")
        self.results['qa'] = self.evaluate_qa_sample()
        self.results['summarization'] = self.evaluate_summarization_sample()

        # 4. 安全性評估
        print("\n[4/5] 安全性評估...")
        self.results['safety'] = self.evaluate_safety()

        # 5. 人類評估（可選）
        print("\n[5/5] 人類評估...")
        self.results['human_eval'] = self.collect_human_feedback()

        return self.results

    def evaluate_perplexity(self):
        """評估困惑度"""
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning models require large datasets."
        ]
        return calculate_perplexity_batch(self.model, self.tokenizer, test_texts)

    def evaluate_mmlu_sample(self):
        """MMLU 樣本評估"""
        # 實現 MMLU 評估邏輯
        return 0.65  # 示例分數

    def evaluate_hellaswag_sample(self):
        """HellaSwag 樣本評估"""
        # 實現 HellaSwag 評估邏輯
        return 0.58  # 示例分數

    def evaluate_qa_sample(self):
        """問答樣本評估"""
        # 實現問答評估邏輯
        return {'exact_match': 0.72, 'f1': 0.81}

    def evaluate_summarization_sample(self):
        """摘要樣本評估"""
        # 實現摘要評估邏輯
        return {'rouge1': 0.42, 'rouge2': 0.20, 'rougeL': 0.38}

    def evaluate_safety(self):
        """安全性評估"""
        # 實現安全性檢查
        return {'toxic_rate': 0.02, 'bias_score': 0.15}

    def collect_human_feedback(self):
        """收集人類反饋"""
        # 實現人類評估流程
        return {'overall_quality': 4.2, 'helpfulness': 4.5}

    def generate_report(self):
        """生成評估報告"""
        report = "=" * 50 + "\n"
        report += "模型評估報告\n"
        report += "=" * 50 + "\n\n"

        report += f"困惑度: {self.results['perplexity']:.2f}\n\n"

        report += "通用基準:\n"
        report += f"  MMLU: {self.results['mmlu']:.2%}\n"
        report += f"  HellaSwag: {self.results['hellaswag']:.2%}\n\n"

        report += "任務特定:\n"
        report += f"  QA F1: {self.results['qa']['f1']:.2%}\n"
        report += f"  摘要 ROUGE-L: {self.results['summarization']['rougeL']:.2%}\n\n"

        report += "安全性:\n"
        report += f"  有害內容率: {self.results['safety']['toxic_rate']:.2%}\n"
        report += f"  偏見分數: {self.results['safety']['bias_score']:.2f}\n\n"

        report += "人類評估:\n"
        report += f"  整體質量: {self.results['human_eval']['overall_quality']:.1f}/5.0\n"
        report += f"  有用性: {self.results['human_eval']['helpfulness']:.1f}/5.0\n"

        return report

# 使用範例
# evaluator = ComprehensiveEvaluator(model, tokenizer)
# results = evaluator.run_all_evaluations()
# print(evaluator.generate_report())
```

### 9.5.2 評估最佳實踐

1. **組合多種評估方法**
   - 自動化指標 + 人類評估
   - 通用基準 + 任務特定基準
   - 定量指標 + 定性分析

2. **避免評估偏見**
   - 使用留出測試集（不用於訓練）
   - 防止數據洩漏
   - 多樣化評估者背景

3. **持續評估**
   - 在訓練過程中定期評估
   - 監控模型退化
   - A/B 測試新版本

4. **透明報告**
   - 公開評估方法和數據集
   - 報告置信區間
   - 披露限制和失敗案例

---

## 參考資源

- **評估工具**
  - [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
  - [HELM](https://github.com/stanford-crfm/helm)
  - [BIG-Bench](https://github.com/google/BIG-bench)

- **基準數據集**
  - [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
  - [MMLU](https://github.com/hendrycks/test)
  - [Chatbot Arena](https://chat.lmsys.org/)

- **論文**
  - "Language Models are Few-Shot Learners" (GPT-3)
  - "Holistic Evaluation of Language Models" (HELM)
  - "Beyond the Imitation Game" (BIG-Bench)
