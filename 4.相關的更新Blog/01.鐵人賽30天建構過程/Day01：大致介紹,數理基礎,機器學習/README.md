# Day 01: 大致介紹,數理跟ＭＬ基礎

# 目錄

- [參賽目的](#參賽目的)
- [AI簡介](#AI簡介)
- [AI課程推薦](#I課程推薦)
- [數理基礎概略](#數理基礎概略)
- [機器學習概略](#機器學習概略)



## 參賽目的
![示意圖](./pic01.png)

參加這次為期30天的部落格挑戰，主要目的是建立一個個人的LLM和深度學習（DL）知識庫，以及完成數個小型的side project。這將有助於我快速復習已掌握的知識，同時也能夠快速分類、吸收和內化新知識，並將其付諸實踐。此外，這個知識筆記還能方便他人了解目前LLM相關技術的進展，並使用筆記中的流程和程式碼生成自己的應用。
目前，這個知識庫的內容將圍繞深度學習、大模型、Agent等相關主題。這次挑戰的目的是深入探索LLM的各個方面，並通過實際專案應用這些知識，提升我的技術能力並分享我的學習成果。
每一天的內容將展示我實作和添加的過程。
目前暫定的內容

1. AI工程師大補帖
1-1 LLM課程擴展與添加 主要內容將基於https://github.com/mlabonne/llm-course 進行延伸和添加，涵蓋從基礎概念到進階應用的全面介紹。我將深入解析該課程中的重點，並補充更多實際案例和應用技巧。
1-2 deeplearning.ai 上的短課程以及轉化的實作 參考 deeplearning.ai 上所有的短課程內容，這些課程將涵蓋最新的AI技術和工具，包括但不限於以下主題：
    • 語言模型的預訓練和微調
    • 高效提示工程
    • 多模態檢索和生成應用
    • LLM在邊緣設備上的部署
    • 強化學習和人類反饋
1-3 論文閱讀 在挑戰期間，我將閱讀並分析兩篇重要的LLM相關論文，分享其中的關鍵發現和應用。
1-4 Stable Diffusion相關內容 探索和應用Stable Diffusion技術，深入理解其在圖像生成和處理方面的應用。
1-5 其他生成式AI相關的內容
2. 數個小 side project
我將設計並實作以下數個與LLM相關的實際專案：
    • 醫療聊天機器人：利用LLM技術開發一個能夠回答醫療相關問題的聊天機器人，目的是提供精確且即時的醫療資訊輔助。
    • 程式碼LLM程式插件：開發一個能夠輔助編寫和調試程式碼的程式插件，提升開發效率和程式碼品質。
    • RPA + LLM：結合機器人流程自動化（RPA）和LLM技術，設計一個自動化工作流程系統，提升企業的運營效率。
期待成果
通過這次挑戰，我希望能夠更加深入掌握LLM以及生成式AI的技術及其應用，並完成三個具有實際價值的專案。此外，我將通過閱讀並理解關鍵論文，掌握從論文閱讀、程式碼理解到吸收內化和產生新題目的整個過程。


## AI簡介
![示意圖](./pic03.png)

什麼是人工智慧（AI）？

人工智慧（Artificial Intelligence, AI）廣義上指的是讓機器展現類似人類智能的技術。這些技術讓機器能感知環境、學習、推理、決策、甚至創作。AI並非單一技術，而是涵蓋多種技術與方法的廣泛領域。

AI的目標

AI的終極目標是創造出具備通用智能的機器，能像人類一樣思考、學習和解決各種問題。然而，目前AI發展仍處於弱人工智慧（Narrow AI）階段，即只能在特定任務上展現出卓越能力，例如：

圖像識別：分辨照片中的物體、人臉辨識
語音辨識：將語音轉換為文字、語音助理（如Siri、Alexa）
自然語言處理：機器翻譯、聊天機器人、文字生成
推薦系統：電商平台的商品推薦、影音平台的內容推薦
棋類遊戲：AlphaGo在圍棋上的勝利
醫療診斷：輔助醫師判讀醫學影像、預測疾病風險
AI的分類

AI的分類方式有很多種，以下介紹幾種常見的分類：

1. 基於能力的分類

弱人工智慧（Narrow AI/Weak AI）：
專注於特定任務，無法處理超出設計範圍的問題。
目前大多數AI應用都屬於此類，如圖像識別、語音辨識等。
強人工智慧（Strong AI/General AI）：
具備與人類相當或超越人類的智能，能像人類一樣思考、學習和解決各種問題。
目前仍處於理論和研究階段，尚未實現。
不過目前隨折chatgpt的崛起，我覺得可能在這幾年就可能有很多接近強甚至是超人工智能的模型出現。
超人工智慧（Super AI）：
智能超越人類，能解決人類無法解決的問題。
屬於科幻小說和電影中的概念，目前沒有科學依據。

2. 基於技術的分類

機器學習（Machine Learning）：
從資料中學習，並根據學習到的知識做出預測或決策。
常見的機器學習方法包括監督式學習、非監督式學習和強化學習。
深度學習（Deep Learning）：
機器學習的一個分支，使用多層神經網路來學習資料的複雜表示。
在圖像識別、語音辨識和自然語言處理等領域取得了重大突破。
自然語言處理（Natural Language Processing, NLP）：
讓機器理解、處理和生成人類語言。
包括機器翻譯、文字摘要、情感分析、聊天機器人等。
計算機視覺（Computer Vision）：
讓機器理解和解釋視覺資訊，如圖像和影片。
包括圖像分類、物體檢測、圖像分割、人臉辨識等。
專家系統（Expert Systems）：
模擬人類專家的知識和經驗來解決特定領域的問題。
通常使用規則和推論引擎來表示知識和進行決策。
機器人學（Robotics）：
結合AI、機械工程和計算機科學，設計和製造機器人。
機器人可以執行各種任務，如工業製造、醫療手術、家庭服務等。
### 人工智慧（AI）的發展歷史

當然，很樂意為您提供更詳細的AI發展歷史：

**人工智慧發展詳史**

人工智慧的發展史宛如一幅波瀾壯闊的畫卷，交織著科學家的夢想、技術的突破，以及對未來世界的無限想像。讓我們一同深入探索這段充滿挑戰與驚喜的旅程。

**1950s-1960s：AI的誕生與樂觀探索**

- **思想萌芽**：
    - 1943年，沃倫·麥卡洛克（Warren McCulloch）和沃爾特·皮茨（Walter Pitts）提出「人工神經元」模型，奠定了神經網路的基礎。
    - 艾倫·圖靈的圖靈測試，為評估機器智能提供了一個重要標準。
- **達特茅斯會議與AI的誕生**：
    - 1956年，約翰·麥卡錫（John McCarthy）、馬文·閔斯基（Marvin Minsky）等人在達特茅斯學院舉辦研討會，首次提出「人工智慧」一詞，並確立了AI研究的目標。
- **早期樂觀與探索**：
    - 研究者們對AI充滿信心，認為在幾十年內就能實現通用人工智慧（AGI）。
    - 邏輯理論家（Logic Theorist）程式成功證明了數學原理，振奮了研究界的士氣。
    - 然而，早期的AI系統多為基於規則的系統，缺乏學習和適應能力。

**1970s-1980s：知識為本的專家系統與第一次寒冬**

- **知識表示與推理**：
    - 研究者們致力於將人類知識表示成計算機可理解的形式，並開發推論引擎來模擬專家決策。
    - 語義網路（Semantic Network）、框架（Frame）等知識表示方法應運而生。
- **專家系統的輝煌與衰落**：
    - MYCIN在診斷血液感染方面表現出色，DENDRAL則成功應用於化學分析。
    - 然而，專家系統的局限性逐漸顯現：知識獲取困難、系統僵化、無法處理不確定性。
- **第一次AI寒冬**：
    - 過高的期望與現實的落差，導致研究經費削減，AI進入第一次寒冬。

**1990s：機器學習的崛起與第二次寒冬**

- **機器學習的新浪潮**：
    - 統計學習方法（如貝葉斯網路、隱馬爾可夫模型）逐漸取代基於規則的系統。
    - 支援向量機（SVM）在分類問題上展現出強大能力。
- **神經網路的低潮**：
    - 雖然反向傳播演算法在80年代末被提出，但受限於計算能力和資料量，神經網路的發展相對緩慢。
- **第二次AI寒冬**：
    - 1990年代中期，AI再次因未能達到預期而陷入低谷。

**2000s：大資料與深度學習的曙光**

- **大資料時代來臨**：
    - 互聯網的發展帶來海量資料，為機器學習提供了充足的訓練資料。
    - 雲端計算和GPU加速技術也為AI發展提供了強大的計算能力。
- **深度學習的復興**：
    - 2006年，傑弗裡·辛頓（Geoffrey Hinton）等人提出深度置信網路（DBN），開啟了深度學習的新時代。
    - 2012年，AlexNet在ImageNet圖像識別比賽中取得突破性成果，深度學習開始廣受關注。

**2010s至今：AI的黃金時代與未來展望**

- **深度學習的全面開花**：
    - 深度學習在各領域取得重大突破，如AlphaGo戰勝人類圍棋冠軍、GPT-3展現驚人的語言生成能力。
    - 生成對抗網路（GAN）在圖像生成、風格轉換等方面展現出強大潛力。
- **AI的廣泛應用**：
    - AI已深入各行各業，如醫療診斷、金融風控、自動駕駛、智慧城市等。
- **挑戰與未來**：
    - AI的可解釋性、公平性、倫理問題仍待解決。
    - 未來，AI將持續發展，並在更多領域帶來創新與變革。

人工智慧的發展史是一段充滿起伏的旅程，從早期的樂觀探索到兩次寒冬，再到如今的蓬勃發展，AI不斷超越自我，為人類社會帶來無限可能。未來，我們期待AI在更多領域發揮其潛力，同時也需要關注其發展帶來的各種挑戰，共同創造一個更美好的AI未來。

## AI課程推薦:

由於這次的挑戰是專注在LLM上，較不相關以及相隔較遠的東西我大概就只會一筆帶過。

這邊我推薦大致可以通過這個課程入門AI:
[Harvard CS50’s Artificial Intelligence with Python – Full University Course](https://youtu.be/5NgNicANyqM?si=yTbD-6wCbPYzsCVL)

下面的課程在學完之後也可以參考下，不過基本上面那個應該就足夠了。

[General Intro | Stanford CS221: Artificial Intelligence: Principles and Techniques (Autumn 2021)](https://youtu.be/ZiwogMtbjr4?si=1KUL6JkiQE7qyiju)

[MIT 6.034 Artificial Intelligence, Fall 2010](https://youtu.be/TjZBTDzGeGg?si=9qV18PmDo9i63Qxsu)

可以從上面的內容發現，隨著時間的演變，這些基礎學科的內容著重的部分其實也有很多改變，所以要學的話大概也就學自己需要的就可以了。

## 數理基礎概略


在掌握機器學習之前，了解支撐了這些演算法的基本數學概念非常重要。不過其實大概看這三個影片課程大概就可以了，這一系列的影片教學有教學跟實作，其他的就有興趣再看。

1.線性代數:[Linear Algebra for Machine Learning](https://www.youtube.com/playlist?list=PLRDl2inPrWQW1QSWhBU0ki-jq_uElkh2a)

這對於理解許多演算法至關重要，尤其是深度學習中使用的演算法。關鍵概念包括向量、矩陣、行列式、特徵值和特徵向量、向量空間和線性變換。
  
2.微積分:[Calculus for Machine Learning ](https://www.youtube.com/playlist?list=PLRDl2inPrWQVu2OvnTvtkRpJ-wz-URMJx)

許多機器學習演算法涉及連續函數的最佳化，這需要了解導數、積分、極限和級數。另外多變量微積分和梯度的概念也很重要。
 
3.機率與統計:[Probability for Machine Learning ](https://www.youtube.com/playlist?list=PLRDl2inPrWQWwJ1mh4tCUxlLfZ76C1zge)

這些對於理解模型如何從資料中學習並做出預測至關重要。 關鍵概念包括機率論、隨機變數、機率分佈、期望、變異數、協方差、相關性、假設檢定、信賴區間、最大似然估計和貝葉斯推理。


可延伸的閱讀：

電子書：

- [深度學習中的數學](https://github.com/jash-git/Jash-good-idea-20200304-001/blob/master/CN%20AI%20book/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E6%95%B0%E5%AD%A6.pdf)

- [深度學習基礎以及數學原理](https://github.com/HaoMood/File/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80%E5%8F%8A%E6%95%B0%E5%AD%A6%E5%8E%9F%E7%90%86.pdf)

其他相關連結：

- [動手深度學習-中的預備知識那章](https://zh-v2.d2l.ai/d2l-zh-pytorch.pdf)
- [動手深度學習這本書也能讓一般人大致上了解深度學習的運作](http://zh.gluon.ai/chapter_introduction/deep-learning-intro.html)
- [Blog- 深度學習(Deep Learning)-數學整理](https://dysonma.github.io/2021/01/27/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-Deep-Learning-%E6%95%B8%E5%AD%B8%E6%95%B4%E7%90%86/)
- [Blog- 機器/深度學習-基礎數學篇(一)(內容跟上面雷同)](https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E6%95%B8%E5%AD%B8%E7%AF%87-%E4%B8%80-1c8337179ad6)
- [我是如何學習深度學習中的數學的？(可參考方法)](https://yanwei-liu.medium.com/%E6%88%91%E6%98%AF%E5%A6%82%E4%BD%95%E5%AD%B8%E7%BF%92%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E4%B8%AD%E7%9A%84%E6%95%B8%E5%AD%B8%E7%9A%84-a26eee623638)

- [深度學習經典(花書)](https://github.com/ytin16/awesome-machine-learning-1/blob/master/Deep-Learning%E8%8A%B1%E4%B9%A6-%E3%80%8A%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E3%80%8B-%E4%B8%AD%E6%96%87%E7%89%88.pdf)

- [深度學習經典(花書)中的數學推導](https://github.com/MingchaoZhu/DeepLearning)

- [繁中的深度學習中的數學相關資料](https://hackmd.io/@changken/rkukooSGS#%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E5%8F%8A%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E7%9A%84%E8%B3%87%E6%96%99)
- [用 Python 學微積分](https://ryancheunggit.gitbooks.io/calculus-with-python/content/01Functions.html)
- [機器學習的數學基礎：矩陣篇](http://www.hahack.com/math/math-matrix/)
- [機器學習的數學基礎：向量篇](http://www.hahack.com/math/math-vector/)
- [機器學習的數學基礎：線性代數進階篇](http://www.hahack.com/math/math-linear-algebra-graded/)
- [Python for Probability, Statistics, and Machine Learning](https://github.com/unpingco/Python-for-Probability-Statistics-and-Machine-Learning)
- [Think Bayes](https://greenteapress.com/wp/think-bayes/)
- [統計分佈 方開泰教授 王元教授](http://item.jd.com/12019664.html)
- [概率論與數理統計 陳希孺教授](https://www.amazon.cn/dp/B073LBYPZ4/ref=sr_1_1?ie=UTF8&qid=1546071311&sr=8-1&keywords=陈希儒)
- [Linear Regression in Python with Scikit-Learn](https://stackabuse.com/linear-regression-in-python-with-scikit-learn/)
- [線性代數自學課程，國內外學習資源](https://selflearningsuccess.com/linear-algebra-courses/#Mathematics_for_Machine_Learning_Linear_Algebra): 本文彙整國內外線性代數自學課程，提供給規劃學習線性代數的朋友們參考。
- [3Blue1Brown - 線性代數的本質](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab): 此系列的影片介紹幾何相關的概念
- [StatQuest with Josh Starmer - 統計基礎知識](https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9): 為許多統計概念提供簡單明了的解釋。
- [Aerin女士的AP統計直觀理解](https://automata88.medium.com/list/cacc224d5e7d): 提供每個機率分佈背後的Medium文章清單。
- [沉浸式線性代數](https://immersivemath.com/ila/learnmore.html): 線性代數的另一種圖像化詮釋.
- [Khan Academy - 線性代數](https://www.khanacademy.org/math/linear-algebra): 非常適合初學者，因為它以非常直觀的方式解釋了概念。
- [Khan Academy - 微積分](https://www.khanacademy.org/math/calculus-1): 一門涵蓋微積分所有基礎知識的互動課程。
- [Khan Academy - 機率與統計](https://www.khanacademy.org/math/statistics-probability): 以易於理解的格式提供材料。


## 機器學習概略

- **Pyothn 基礎**:其實去 freecode camp 練下大概就可以了，程式語言只要會C ,C++的話，其他的語言就不會相差太多，除了彙編語言或verilog這類的語言之外。

    -  推薦閱讀,應用-[Into Python in one video](https://www.giraffeacademy.com/programming-languages/python/in-one-video/)
- **資料科學函式庫**: 包括熟悉用於數值運算的 NumPy、用於資料操作和分析的 Pandas、用於資料視覺化的 Matplotlib 和 Seaborn。
    -  推薦閱讀,應用-[Visualization Curriculum](https://uwdata.github.io/visualization-curriculum/intro.html)
    -  推薦閱讀,應用-[Python資料科學分享－3.資料視覺化(1)](https://asyncfor.com/jupyter/python/data%20science/2020/05/22/data-viz-1.html)
    -  推薦閱讀,應用-[Python資料科學分享－3.資料視覺化(2)](https://asyncfor.com/jupyter/python/data%20science/2020/05/29/data-viz-2.html)
- **資料預處理**: 這涉及特徵縮放和標準化、處理缺失資料、異常值檢測、分類資料編碼以及將資料拆分為訓練集、驗證集和測試集。
    -  推薦閱讀,應用- 概覽[[資料分析&機器學習] 第2.4講：資料前處理(Missing data, One-hot encoding, Feature Scaling)](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC2-4%E8%AC%9B-%E8%B3%87%E6%96%99%E5%89%8D%E8%99%95%E7%90%86-missing-data-one-hot-encoding-feature-scaling-3b70a7839b4a)
    -  推薦閱讀,應用-[Python資料科學分享—2.資料處理](https://asyncfor.com/jupyter/python/data%20science/2020/05/15/data-etl.html)
    -  推薦閱讀,應用-[[機器學習筆記]資料預處理](https://doremi31618.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98-%E6%95%B8%E6%93%9A%E9%A0%90%E8%99%95%E7%90%8601-ae90853978da)
- **機器學習課程**:[Machine Learning in 2024 – Beginner's Course](https://youtu.be/bmmQA8A-yUA?si=_2Ga-3WKdar_80fj)
- **機器學習函式庫**: 熟練使用 Scikit-learn（一個提供多種監督和非監督學習演算法的函式庫）至關重要。了解如何實現線性迴歸、邏輯迴歸、決策樹、隨機森林、k 最近鄰 (K-NN) 和 K 均值聚類等演算法非常重要。PCA 和 t-SNE 等降維技術也有助於視覺化高維度資料。
    -  推薦閱讀,應用(裡面有ipynb)-[Introducing Scikit-Learn](https://doremi31618.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98-%E6%95%B8%E6%93%9A%E9%A0%90%E8%99%95%E7%90%8601-ae90853978da)
    - 推薦閱讀,應用(裡面有ipynb)-[Python 資料科學手冊(裡面有ipynb ,colab)](https://jakevdp.github.io/PythonDataScienceHandbook/):
      免費的數位書籍，是學習 pandas、NumPy、Matplotlib 和 Seaborn 的絕佳資源。    



