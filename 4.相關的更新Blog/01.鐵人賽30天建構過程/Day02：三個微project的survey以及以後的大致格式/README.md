由於今天太晚才開始寫，所以目前大致上大概就把這30天要做的三個微project的survey跟進度,以及之後大致的內容大致的格式大概描述一下。

1.醫療助手：

目前的實際進度：[medical_chat](https://github.com/markl-a/My-AI-Engineer-s-Notes/blob/main/2.%E6%B7%B1%E5%85%A5LLM%E6%A8%A1%E5%9E%8B%E5%B7%A5%E7%A8%8B%E8%88%87LLM%E9%81%8B%E7%B6%AD/%E3%80%8C20240619_%E7%89%88%E6%9C%AC_medical_chat%E3%80%8D.ipynb)

已經找到資料集, 也弄到模型, 只是目前微調的不是太成功，在EOS後還是會有輸出。目前也沒有壓縮跟弄到gradio 或類似的介面上。

理想是能參考下面的模型以及使用gpt或python把英文,簡中的資料集轉成繁中後，漸漸地做到下面兩個項目的樣子。

   1. [MedicalGPT](https://github.com/shibing624/MedicalGPT)
   2. [Huatuo-Llama-Med-Chinese](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)

2.Code 助手:

目前是想做出能遵守公司內部的程式開發規範以及能呼叫公司內已開發程式碼的 codecopilot，目前可能會先理解開源的copilot以及devin項目。

   下面為參考的項目，只希望能做出我日常生活中能用到的版本就好。
   1. [fauxpilot](https://github.com/fauxpilot/fauxpilot)
   2. [picopilot](https://github.com/coder/picopilot)
   3. [OpenDevin](https://github.com/OpenDevin/OpenDevin)
   4. [awesome-devins](https://github.com/e2b-dev/awesome-devins)
   5. [auto-dev-vscode](https://github.com/unit-mesh/auto-dev-vscode)
   6. [gpt-pilot](https://github.com//Pythagora-io/gpt-pilot)
   7. [tabby](https://github.com/TabbyML/tabby)

3.RPA +LLM

這篇[csdn](https://blog.csdn.net/liangwqi/article/details/134399646)文章很好的敘述了大致的思路。

目前可能會先透過agent框架或是zapier,uipath這類的框架(儘管這兩類框架其實也不太一樣)先搭出一個可用的版本，之後再想要怎麼做。

   下方為參考的項目：
   1. [skyvern](https://github.com/Skyvern-AI/skyvern)
   2. [autoMate](https://github.com/yuruotong1/autoMate)
   3. [Autogen_GraphRAG_Ollama](https://github.com/karthik-codex/Autogen_GraphRAG_Ollama)

4.之後大致的內容跟格式

   1.後幾天基本是[LLM Course](https://github.com/mlabonne/llm-course) 中模型的部分，每天大概一個部分，盡可能包含實作（9天）
   2.之後就大概是 [LLM Course](https://github.com/mlabonne/llm-course)中應用的部分，也是依樣每天一個部分，也盡可能包含實作（7~8天）
   
   剩下大概12天
   3. 音樂,語音,圖像,影片生成,具身智能的內容,模型,框架跟實作(5天)
   4. 論文+code兩篇 (2天)
   5. 
       5.1: ML (pandas,numpy,sk-learn)
       5.2: DL (pytorch ,tensorflow2 ,keras3,optuna)中的內容跟實作(1天)(要提前準備)
   6.醫療助手  一天(要提前準備)
   7.code助手  一天(要提前準備)
   8.RPA+LLM  一天(要提前準備)
   9.目前較新或較為熱門的生成式AI項目介紹,使用跟原理解讀
   上面的內容除了微項目外大多會轉到我在github上面的[筆記](https://github.com/markl-a/My-AI-Engineer-s-Notes)中，當然這也是目的之一就是了
    
這樣大概週末真的要屯稿就是了，要不然大概做不完。