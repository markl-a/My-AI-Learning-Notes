
### LLM 簡介與架構

#### 1. **簡介**
大型語言模型（LLM）在自然語言處理（NLP）領域取得了顯著的進步。這些模型大多基於Transformer架構，特別是解碼器部分，如GPT模型系列。理解LLM的基本輸入（tokens 令牌）和輸出（logits）以及注意力機制對於掌握LLM的工作原理至關重要。

詳細的LLM簡介可參照我翻譯的：[Stanford CS25-Apr 2024: V4 I Overview of Transformers - Transformers and LLMs: An Introduction(上)](https://ithelp.ithome.com.tw/articles/10343567)

[Stanford CS25-Apr 2024: V4 I Overview of Transformers - Transformers and LLMs: An Introduction(下)](https://ithelp.ithome.com.tw/articles/10343633)

看完上面的連結對目前大模型的狀況變可以了解一二。

**延伸閱讀與觀看**：
- [Hugging Face- NLP Course](https://huggingface.co/learn/nlp-course/zh-TW/chapter1/1): 難度較低的課程，適合快速入門和對 transformer 中的 NLP有一個了解。
- [[1hr Talk] Intro to Large Language Models](https://youtu.be/zjkBMFhNj_g?si=VnNOE1gggtAhxTDn): 大型語言模型的簡介講座。

- [CS25: Transformers United V4](https://web.stanford.edu/class/cs25/): 史丹佛大學的Transformer課程，涵蓋架構及應用，較為深入。

#### 2. **LLM 整體架構 - Transformer整體架構**

LLM通常基於Transformer架構，其中特別採用了僅使用解碼器的設計（例如GPT系列）。這些模型使用自注意力機制來處理輸入並生成輸出。其他的架構則之後會陸續介紹。

具體請參照：

1. [Let's build GPT: from scratch, in code, spelled out. 學習紀錄](https://ithelp.ithome.com.tw/articles/10343476)

2. 最原始的 transformer 版本圖文詳細敘述：[Transformer 運作原理圖解](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar

3. [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU?si=qwkdLAXfFlk_aRJp):跑完這個流程大概對程式碼跟模型的理解絕對會更深的多。

4. [nanoGPT 流程圖像化](https://bbycroft.net/llm) by Brendan Bycroft: 3D視覺化展示LLM內部運作。

**延伸閱讀與觀看**：
- [LLM Foundations](https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/llm-foundations/): 包含詳細的模型架構介紹和理論背景。-這個類似上面的內容，不過內容比較廣泛以及片介紹性質。

- [LLM-from-scratch.ipynb](https://colab.research.google.com/gist/iamaziz/171170dce60d9cd07fab221507fd1d52): 簡化版的GPT模型實作範例。

- [從編解碼和字詞嵌入開始，一步一步理解Transformer](https://www.youtube.com/watch?v=GGLr-TtKguA)

- [Hugging Face - transformers doc](https://huggingface.co/docs/transformers/v4.44.0/en/quicktour): Hugging Face 的 transformers庫的文件和教學，對transformer 的方方面面都有介紹到，英文的版本最為詳細。

- [Building LLMs from Scratch](https://youtu.be/UU1WVnMk4E8?si=Vn1IbHE5p5LUQmKi): 這個類似 Let's build GPT: from scratch，不過更為詳細。

- [GPT-2圖解](https://jalammar.github.io/illustrated-gpt2/) by Jay Alammar: 專注於GPT架構的視覺化解釋。

#### 3. **標記化 Tokenization**
將原始文字資料轉換為模型可以理解的格式，即token。這過程包括將文字拆分為標記（通常是單字或子單字）。

**具體請參考:**
- [Let's build the GPT Tokenizer](https://youtu.be/zduSFxRajkE?si=IhIuECg3ZSGHRtWT): 解釋如何構建GPT分詞器。

中文方面兩者擇一了解即可，找了很久沒找到繁中的，感覺可惜。

- [怎么让英文大预言模型支持中文？（一）构建自己的tokenization 
](https://www.cnblogs.com/xiximayou/p/17500806.html)

- [【中文编码】利用bert-base-chinese中的Tokenizer实现中文编码嵌入](https://blog.csdn.net/qq_43426908/article/details/134748200)



#### 4. **注意力機制**
注意力機制是LLM的核心技術，它使得模型能夠在生成輸出時關注輸入的不同部分。這包括自注意力和縮放點積注意力機制，相關的介紹其實在前面架構介紹的內容裡也有提到。

**延伸閱讀與觀看**：
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) by Lilian Weng: 對注意力機制必要性的正式介紹。
- [動手深度學習-注意力機制](https://zh.d2l.ai/chapter_attention-mechanisms/index.html): 詳細介紹注意力機制的理論和實現。

#### 5. **文字生成**
模型使用不同的策略生成文字輸出。常見策略包括貪婪解碼、波束搜尋、top-k 採樣和核採樣。
- [Decoding Strategies in LLMs](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html): 對各種解碼策略的圖像化介紹及程式碼實現。

**延伸閱讀與觀看**：
- [如何產生文字: 透過 Transformers 以不同的解碼方法產生文字](https://huggingface.co/blog/zh/how-to-generate): 介紹各種文字生成策略及其實現。

#### 6. **參考的流程跑通小專案**

下面的都是對岸的，沒辦法，因為流程跟繁中是最類似的，假如有繁中的話拜託讓我知道，萬分感謝。

1. [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)
2. [ChatLM-mini-Chinese](https://github.com/charent/ChatLM-mini-Chinese)

#### 7. **其他的模型架構或方法**

</br></n>

##### 7.1 **新的位置嵌入 Positional embeddings相關方法**: 

在了解原始 transformer 的 Positional embeddings方法後，就可看下不同的方法，像是[RoPE](https://arxiv.org/abs/2104.09864) 這樣的相對位置編碼方案。或實現 [YaRN](https://arxiv.org/abs/2309.00071) (通過溫度因子乘以注意力矩陣) 跟 [ALiBi](https://arxiv.org/abs/2108.12409) (基於token距離的注意力獎懲) 來擴展上下文長度。

- [Extending the RoPE](https://blog.eleuther.ai/yarn/) by EleutherAI: 總結不同位置編碼技術的文章.

- [Understanding YaRN](https://medium.com/@rcrajatchawla/understanding-yarn-extending-context-window-of-llms-3f21e3522465) by Rajat Chawla: 對YaRN的介紹.
  
##### 7.2 **Mamba ,RWKV , TTT等新模型**:

- Mamba 介紹：[一文讀懂Mamba：具有選擇狀態空間的線性時間序列建模](https://zhuanlan.zhihu.com/p/680846351)

- RWKV 介紹:[RWKV 模型解析](https://zhuanlan.zhihu.com/p/640050680)

- TTT 介紹:[Test-Time Training on Graphs with Large Language Models (LLMs)](https://arxiv.org/pdf/2404.13571)



##### 7.3 **模型融合 Model merging**: 

另外將以訓練的模型合併也是一個提升表先的方法，具體的可參考這個 [mergekit](https://github.com/cg123/mergekit) 庫，這個課實現了許多融合的方法，如 SLERP, [DARE](https://arxiv.org/abs/2311.03099), 和 [TIES](https://arxiv.org/abs/2311.03099)。

模型融合通常指的是將多個已訓練的模型合併成一個單一模型的過程。這不僅僅是用參數平均或投票決定輸出，而是在模型的權重和結構層面上進行合併。這個過程不需要再次訓練，可以通過數學操作（如球面線性內插（SLERP）或其他融合技術）將不同模型的知識整合起來。模型融合可用於建立一個表現更佳、更強大的模型，通常是將多個模型在特定任務上的優勢結合起來。

- [Merge LLMs with mergekit](https://mlabonne.github.io/blog/posts/2024-01-08_Merge_LLMs_with_mergekit.html): 關於使用mergekit進行模型融合的教程.

##### 7.4 **專家混合 Mixture of Experts**: 

[Mixtral](https://arxiv.org/abs/2401.04088) 因其卓越的性能而重新使MoE架構流行起來。 與此同時，開源社區出現了一種frankenMoE，通過融合像 [Phixtral](https://huggingface.co/mlabonne/phixtral-2x2_8)這樣的模型，這是一個更經濟且性能良好的選項。MoE是一種結構，它包含多個子模型或“專家”，每個專家專門處理不同的任務或資料子集。在MoE架構中，一個“gate”或調度器決定對於給定的輸入，哪個專家被使用。這是一種稀疏啟動方法，可以大幅提升模型的容量和效率，因為不是所有的專家都會對每個輸入進行響應。

- [Mixture of Experts Explained](https://huggingface.co/blog/moe) by Hugging Face: 關於MoE及其工作方式的詳盡指南.
  
##### 7.5 **多模態模型 Multimodal models**: 

這類模型像是（ [CLIP](https://openai.com/research/clip), [Stable Diffusion](https://stability.ai/stable-image), 或 [LLaVA](https://llava-vl.github.io/)) 能處理多種類型的輸入（文字、圖像、音頻等）以及具備了統一的嵌入空間，從而具備了強大的應用能力，如文字到圖像。
    
- [Large Multimodal Models](https://huyenchip.com/2023/10/10/multimodal.html) by Chip Huyen: 對多模態系統及其近期發展歷史的概述.
    
- [Sora可能架構的解析](https://blog.csdn.net/v_JULY_v/article/details/136143475?spm=1000.2115.3001.5927)

