Day04-1:

### LLM 架構概論

#### 1. **簡介**
大型語言模型（LLM）在自然語言處理（NLP）領域取得了顯著的進步。這些模型大多基於Transformer架構，特別是解碼器部分，如GPT模型系列。理解LLM的基本輸入（tokens 令牌）和輸出（logits）以及注意力機制對於掌握LLM的工作原理至關重要。

#### 2. **高階視圖**
LLM通常基於Transformer架構，其中特別採用了僅使用解碼器的設計（例如GPT系列）。這些模型使用自注意力機制來處理輸入並生成輸出。

**參考資料**：
- [LLM Foundations](https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/llm-foundations/): 包含詳細的模型架構介紹和理論背景。
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY): 提供從頭構建GPT模型的實作指南。
- [LLM-from-scratch.ipynb](https://colab.research.google.com/gist/iamaziz/171170dce60d9cd07fab221507fd1d52): 簡化版的GPT模型實作範例。
- [[1hr Talk] Intro to Large Language Models](https://youtu.be/zjkBMFhNj_g?si=VnNOE1gggtAhxTDn): 大型語言模型的簡介講座。

#### 3. **標記化 Tokenization**
將原始文字資料轉換為模型可以理解的格式，即token。這過程包括將文字拆分為標記（通常是單字或子單字）。

**參考資料**：
- [Let's build the GPT Tokenizer](https://youtu.be/zduSFxRajkE?si=IhIuECg3ZSGHRtWT): 解釋如何構建GPT分詞器。

#### 4. **注意力機制**
注意力機制是LLM的核心技術，它使得模型能夠在生成輸出時關注輸入的不同部分。這包括自注意力和縮放點積注意力機制。

**參考資料**：
- [動手深度學習-注意力機制](https://zh.d2l.ai/chapter_attention-mechanisms/index.html): 詳細介紹注意力機制的理論和實現。

#### 5. **文字生成**
模型使用不同的策略生成文字輸出。常見策略包括貪婪解碼、波束搜尋、top-k 採樣和核採樣。

**參考資料**：
- [如何生成文字: 通过 Transformers 用不同的解码方法生成文字](https://huggingface.co/blog/zh/how-to-generate): 介紹各種文字生成策略及其實現。

#### 6. **學習資源**
為了更深入理解LLM的架構和應用，可以參考以下資源：
- [HuggingFace的 Transformer教學](https://huggingface.co/docs/transformers/quicktour): 包含從導入、微調到部署的完整流程。
- [CS25: Transformers United V3](https://web.stanford.edu/class/cs25/): 史丹佛大學的Transformer課程，涵蓋架構及應用。
- [Stanford CS25: V1 I Transformers United: DL Models that have revolutionized NLP, CV, RL](https://www.youtube.com/watch?v=P127jhj-8-Y&list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM): 史丹佛大學Transformer課程的錄影。

#### 7. **深入閱讀**
以下是一些對LLM和Transformer架構深入理解的推薦閱讀材料：
- [Building LLMs from Scratch](https://youtu.be/UU1WVnMk4E8?si=Vn1IbHE5p5LUQmKi): 從零開始構建LLM的指南。
- [Transformer 插圖](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar: 直觀的Transformer模型解釋。
- [GPT-2圖解](https://jalammar.github.io/illustrated-gpt2/) by Jay Alammar: 專注於GPT架構的視覺化解釋。
- [LLM Visualization](https://bbycroft.net/llm) by Brendan Bycroft: 3D視覺化展示LLM內部運作。
- [nanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy: 從頭開始重新實現GPT的詳細教程。
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) by Lilian Weng: 對注意力機制必要性的正式介紹。
- [Decoding Strategies in LLMs](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html): 對各種解碼策略的圖像化介紹及程式碼實現。

這份概論提供了一個全面的學習框架，涵蓋了LLM的基本概念、核心技術以及實踐資源。希望這對你的學習有幫助！