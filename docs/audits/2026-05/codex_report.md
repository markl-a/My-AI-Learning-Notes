===== FINAL AUDIT REPORT START =====

**總評**
這個 repo 比較像「教學內容倉庫」，不是可穩定安裝、測試、發版的 Python 專案。最大問題不是 Markdown 多，而是工程邊界沒有收斂：30 份 `requirements*.txt`、浮動版本、CI 大量 `continue-on-error`、notebook 不鎖版、子專案測試不進 CI。工作樹乾淨；本次只做唯讀靜態檢查，沒有執行會產生 `.coverage`、`htmlcov/` 或 notebook 變更的測試。

**1. Python Script 品質**
共掃到 202 個 `.py`，AST 檢查有 1 個語法錯誤：[12_ai_assistant.py](<D:/Projects/My-AI-Learning-Notes/1.從AI到LLM基礎/4.DL/00.DL_Path/6_卷積神經網路/12_ai_assistant.py:22>) 的 `class CNN Concept Explainer:` 不是合法 Python class name。這會讓全 repo `ruff`、`black`、`pytest` 類工具掃描時直接爆。

指定範圍裡，SFT 腳本可讀性尚可，但可執行性脆弱：[3_train_model.py](<D:/Projects/My-AI-Learning-Notes/2.深入LLM模型工程與LLM運維/5.監督微調 (SFT)/hands_on_project/scripts/3_train_model.py:108>) 使用 `trust_remote_code=True`，教學可接受但應預設關閉或明確白名單；同檔 [216-224](<D:/Projects/My-AI-Learning-Notes/2.深入LLM模型工程與LLM運維/5.監督微調 (SFT)/hands_on_project/scripts/3_train_model.py:216>) 用舊式 `SFTTrainer(... dataset_text_field, max_seq_length, tokenizer ...)`。依 Hugging Face TRL 目前文件，最新版偏向 `SFTConfig(max_length=...)` / `processing_class`，所以在 `trl>=0.7.0` 這種浮動依賴下很容易壞。

`3.LLM應用工程/9.實戰` 有更嚴重的安全味道：[agent_tools.py](<D:/Projects/My-AI-Learning-Notes/3.LLM應用工程/9.實戰/9.1-RAG-Agent端到端實戰/src/agent_tools.py:136>) 用 `eval`，[同檔 285](<D:/Projects/My-AI-Learning-Notes/3.LLM應用工程/9.實戰/9.1-RAG-Agent端到端實戰/src/agent_tools.py:285>) 用 `exec`。目前只是用 regex 與受限 globals 擋，這不是可靠 sandbox。GaLore 更直接：[setup.py](<D:/Projects/My-AI-Learning-Notes/2.深入LLM模型工程與LLM運維/GaLore_Demo/GaLore-master/setup.py:3>) 讀 `requirements.txt`，但該目錄沒有這個檔，`pip install -e` 會失敗。

**2. Notebook 可重現性**
5 個 Colab notebook 都沒有執行輸出殘留、code cell 靜態語法可 parse，這是好的。但沒有 Colab badge；安裝 cell 全用 `pip install -U` 搭配 lower bound，例如 LoRA notebook [line 84](<D:/Projects/My-AI-Learning-Notes/2.深入LLM模型工程與LLM運維/5.監督微調 (SFT)/hands_on_project/notebooks/Colab_LoRA_SFT_Mini_Demo.ipynb:84>)、DPO [line 69](<D:/Projects/My-AI-Learning-Notes/2.深入LLM模型工程與LLM運維/6.偏好對齊 (Alignment) 技術/notebooks/Colab_DPO_Alignment_Mini_Demo.ipynb:69>)、vLLM [line 77](<D:/Projects/My-AI-Learning-Notes/2.深入LLM模型工程與LLM運維/8.模型部署與運維/notebooks/Colab_vLLM_Deploy_PrefixCache_Demo.ipynb:77>)。這不是可重現，只是「今天可能能跑」。

vLLM notebook 明確要求 GPU：[line 58](<D:/Projects/My-AI-Learning-Notes/2.深入LLM模型工程與LLM運維/8.模型部署與運維/notebooks/Colab_vLLM_Deploy_PrefixCache_Demo.ipynb:58>) 直接 assert CUDA；Agent / GraphRAG notebook 需要 API key `getpass`，所以不是真正一鍵跑。CI 的 notebook job 更差：[ci.yml](<D:/Projects/My-AI-Learning-Notes/.github/workflows/ci.yml:139>) 用 `nbconvert --execute --inplace ... || true`，會忽略失敗，還在 CI 中就地改 notebook。

**3. 依賴管理**
目前有 30 份 requirements，無 lockfile、無 constraints。靜態統計有 529 條非精確 pin。版本漂移很明顯：`openai` 在根 LLM requirements 是 `>=1.50.0`，但部署範例仍有 `openai==1.3.0`；`fastapi` 有 `>=0.115.0` 與 `==0.104.1`；`pydantic` 有 `>=2.9.0` 與 `==2.5.0`。`pytest` 也分裂：[requirements.txt](<D:/Projects/My-AI-Learning-Notes/requirements.txt:35>) 是 `pytest>=7.4.0`，[requirements-dev.txt](<D:/Projects/My-AI-Learning-Notes/requirements-dev.txt:10>) 是 `pytest>=8.0.0`。更不乾淨的是 [pyproject.toml](<D:/Projects/My-AI-Learning-Notes/pyproject.toml:56>) 把 `pytest` 放進 runtime dependencies。

**4. CI/CD**
`ci.yml` 目前不是品質閘門。Markdown / docs 變更被整個忽略：[ci.yml](<D:/Projects/My-AI-Learning-Notes/.github/workflows/ci.yml:17>)、[24](<D:/Projects/My-AI-Learning-Notes/.github/workflows/ci.yml:24>)，但 repo 約 80% 是 Markdown。Ruff、Black、MyPy 都 `continue-on-error`：[57-68](<D:/Projects/My-AI-Learning-Notes/.github/workflows/ci.yml:57>)；Bandit、Safety、pip-audit 也都不擋：[163-178](<D:/Projects/My-AI-Learning-Notes/.github/workflows/ci.yml:163>)。更糟的是 security job 只安裝掃描工具，沒有安裝 project requirements，所以 `pip-audit --desc --strict` 多半審的是掃描環境，不是這個 repo 的 30 份 requirements。

Benchmark workflow 也不可靠。LLM / Agent benchmark 寫到 `results/`：[benchmark_llm.py](<D:/Projects/My-AI-Learning-Notes/benchmarks/benchmark_llm.py:358>)、[benchmark_agent.py](<D:/Projects/My-AI-Learning-Notes/benchmarks/benchmark_agent.py:396>)，但 workflow 上傳 `benchmarks/results/*`：[benchmark.yml](<D:/Projects/My-AI-Learning-Notes/.github/workflows/benchmark.yml:62>)、[129](<D:/Projects/My-AI-Learning-Notes/.github/workflows/benchmark.yml:129>)。所以即使跑完，也可能沒有 artifact。

GitHub 顯示 75 個 Dependabot vulnerabilities 的處理方式：先不要逐個手修。這個 repo 沒有 `.github/dependabot.yml`，也沒有鎖版，應先建立 dependency inventory 與 constraints，否則修完還會漂回來。

**5. Repo 健康度**
`.gitignore` 有 `.env`、`.pytest_cache/`、`htmlcov/`、`.coverage`、`.DS_Store`，方向正確。但 git 仍追蹤 32 個 `.DS_Store`；[CHANGELOG.md](<D:/Projects/My-AI-Learning-Notes/CHANGELOG.md:27>) 還寫「清理被誤追蹤的 .DS_Store」，實際不符。secret 掃描沒有看到明顯真 key，但 `.env.example` 使用 `sk-your-openai-api-key-here` 這類會觸發掃描器的假 key：[.env.example](<D:/Projects/My-AI-Learning-Notes/.env.example:15>)。`mkdocs.yml` 也未收斂：[line 153](<D:/Projects/My-AI-Learning-Notes/mkdocs.yml:153>) 指向不存在的 `index.md`，[line 205](<D:/Projects/My-AI-Learning-Notes/mkdocs.yml:205>) 指向不存在的 `tags.md`。

**5 個最該優先修的工程實踐問題**
1. 讓 CI 真的會失敗：移除 [ci.yml](<D:/Projects/My-AI-Learning-Notes/.github/workflows/ci.yml:58>)、[63](<D:/Projects/My-AI-Learning-Notes/.github/workflows/ci.yml:63>)、[68](<D:/Projects/My-AI-Learning-Notes/.github/workflows/ci.yml:68>)、[165](<D:/Projects/My-AI-Learning-Notes/.github/workflows/ci.yml:165>)、[172](<D:/Projects/My-AI-Learning-Notes/.github/workflows/ci.yml:172>)、[178](<D:/Projects/My-AI-Learning-Notes/.github/workflows/ci.yml:178>) 的 `continue-on-error`；把 notebook job 改成不 `--inplace` 且不 `|| true`。驗證命令：`python -m yaml` 不適合，改跑 `python -c "import yaml, pathlib; [yaml.safe_load(p.read_text(encoding='utf-8')) for p in pathlib.Path('.github/workflows').glob('*.yml')]"`。
2. 建立依賴治理：新增 `constraints.txt` 或用 `pip-tools` 產生鎖定檔；把 `requirements-full.txt` 當入口，統一 `openai/fastapi/pydantic/langchain/trl` 版本。先跑：`pip-compile requirements-full.txt -o requirements.lock.txt`。
3. 補 Dependabot：新增 `.github/dependabot.yml`，至少掃 `/` 與每個子專案 requirements；同時在 CI 加 `pip-audit -r requirements.txt -r requirements-dev.txt`，再逐步擴到 30 份 requirements。
4. 修可執行性硬錯：改 [12_ai_assistant.py](<D:/Projects/My-AI-Learning-Notes/1.從AI到LLM基礎/4.DL/00.DL_Path/6_卷積神經網路/12_ai_assistant.py:22>) class name；補或移除 GaLore [setup.py](<D:/Projects/My-AI-Learning-Notes/2.深入LLM模型工程與LLM運維/GaLore_Demo/GaLore-master/setup.py:3>) 依賴的 `requirements.txt`。
5. 清 repo 污染與 docs build：用 `git rm --cached -r -- '**/.DS_Store'` 清 32 個已追蹤 `.DS_Store`；補 `docs/index.md` 或把 [mkdocs.yml](<D:/Projects/My-AI-Learning-Notes/mkdocs.yml:153>) 改指 `README.md`，並補 `tags.md` 或移除 [line 205](<D:/Projects/My-AI-Learning-Notes/mkdocs.yml:205>)。

外部 API 比對參考：Hugging Face TRL SFTTrainer 文件、LangChain ChatOpenAI / messages 文件、vLLM prefix caching 文件。

===== FINAL AUDIT REPORT END =====
