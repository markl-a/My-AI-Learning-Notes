
# Kaggle_Tips_and_Tricks.md

## 簡介
本筆記基於參賽經驗、討論區見解、核實過的內核方法、以及眾多解決方案論文彙整而成，旨在協助你提升 Kaggle 競賽表現。內容涵蓋從資料理解 (EDA) 到特徵工程、模型優化與泛化策略等各個層面。透過下列方法與技巧，你可更有效率地針對不同問題特性做出相應調整，並在 Leaderboard 上取得更佳的成績。

## I. 資料探索與理解 (EDA) / 資料預處理與探索性分析

### 目的
EDA 的主要目的是深入了解資料結構、分佈、模式、異常值和缺失值問題，為後續特徵工程和模型選擇提供明確方向。

### 常用方法

1. **基礎資料分析 / 基本統計**：  
   - 使用 `df.info()` 和 `df.describe()` 快速了解資料維度、欄位型態及缺失值分布。  
   - 計算平均值、中位數、標準差、頻率等統計量，了解資料基本分佈情況。

2. **視覺化**：  
   - 直方圖 (Histograms)：查看數值特徵分佈。  
   - 箱型圖 (Box Plots)：檢視中位數、四分位數及異常值。  
   - 散佈圖 (Scatter Plots)：觀察兩特徵間的關係。  
   - 熱力圖 (Heatmaps)：顯示特徵間的相關係數矩陣，發現多重共線性問題。  
   - 條形圖 (Bar Charts)：比較類別特徵的頻率或比例。  
   - (若為時間序列) 折線圖 (Line Charts)：觀察資料隨時間變化趨勢。

   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt

   # 例：查看 SalePrice 分佈直方圖
   sns.histplot(data=train, x='SalePrice', kde=True)
   plt.show()

   # 查看特徵間相關性
   corr = train.corr()
   sns.heatmap(corr, cmap='coolwarm', square=True)
   plt.show()
   ```

3. **缺失值分析 (Missing Values)**：  
   - 識別缺失值比例和模式，決定填補方式（均值/中位數/眾數/插補模型）。  
   - 對於類別缺失值，可考慮以 `'None'`、`'Unknown'` 或該欄位眾數填補。  
   - 若資料量充足，可視情況刪除缺失過多的樣本或特徵。

   ```python
   # 以 Neighborhood 為群組中位數填補 LotFrontage
   train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'] \
                              .transform(lambda x: x.fillna(x.median()))
   ```

4. **異常值與離群值處理 (Outliers)**：  
   - 使用 IQR、Z-score 或箱型圖觀察異常值，考量是否對其進行修正、刪除或使用對數轉換降低偏態。  
   - 結合領域知識，判斷離群值的合理性。

5. **重複資料檢查**：  
   - 確保不存在重複行資料，避免影響模型訓練。

6. **資料分布與假設**：  
   - 若目標值呈高度偏態，可考慮對數轉換 (Log transform) 使分佈更接近常態，以便模型更好擬合。

7. **特定領域資料探索**：  
   - 依競賽主題，結合領域知識，針對特徵做深入探索（如房價預測中，分析房屋品質、地區特性等）。

### 資料清理技巧範例
```python
# 缺失值填補範例
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')
train['GarageType'] = train['GarageType'].fillna('None')

# 使用模型法(KNN或回歸)補缺值 (需額外實作，此為示意)
# 可利用特定特徵構建模型來預測缺失值
```

---

## II. 特徵工程 (Feature Engineering)

### 目的
特徵工程是 Kaggle 比賽中提升模型效果的關鍵，透過創造更具代表性的特徵或轉換既有特徵來更好捕捉資料本質。

### 常用方法

#### A. 特徵建立 (Feature Creation)
- **組合特徵**：  
  利用加減乘除將特徵組合。例如：  
  `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`  
  建立特徵交互項，如：  
  `OverallGrade = OverallQual * OverallCond`
  
- **多項式特徵 (Polynomial Features)**：  
  對存在非線性關係的特徵，嘗試二次、三次方特徵。

- **分組聚合 (Group Aggregation)**：  
  按分類/群組特徵計算平均值、標準差、最大值、最小值、頻次等統計特徵。

- **基於領域知識的特徵**：  
  依領域邏輯創造特徵，使模型更具解釋性。

- **目標編碼 (Target Encoding)**、**頻率編碼 (Frequency Encoding)**：  
  利用目標值或出現頻率對類別特徵編碼。

- **時間特徵、空間特徵**：  
  從日期時間中提取年月日、小時、是否週末/假日等。  
  若有經緯度資料，計算距離和方位。

#### B. 特徵轉換 (Feature Transformation)
- **數值特徵處理**：  
  標準化 (Standardization)、歸一化 (Normalization)、對數轉換 (Log Transform)、分箱 (Binning)、Box-Cox 轉換等。

- **類別特徵處理**：  
  獨熱編碼 (One-Hot Encoding)、標籤編碼 (Label Encoding)、目標編碼 (Target Encoding)、頻率編碼 (Frequency Encoding)。

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Label Encoding 順序性類別特徵
for col in ['ExterQual','ExterCond','HeatingQC','KitchenQual']:
    lbl = LabelEncoder()
    train[col] = lbl.fit_transform(train[col].astype(str))

# One-Hot Encoding 名義型類別
train = pd.get_dummies(train, drop_first=True)

# 數值特徵標準化
scaler = StandardScaler()
num_cols = train.select_dtypes(include=[float, int]).columns
train[num_cols] = scaler.fit_transform(train[num_cols])
```

#### C. 特徵選擇 (Feature Selection)
- **過濾法 (Filter Methods)**：根據相關性、統計檢定去除冗餘特徵。  
- **包裹法 (Wrapper Methods)**：使用 RFE 等方法結合模型性能迭代選擇特徵。  
- **嵌入法 (Embedded Methods)**：使用 LASSO、Ridge、樹模型特徵重要度。

```python
from sklearn.linear_model import LassoCV
import numpy as np

X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

lasso = LassoCV(cv=5, random_state=42).fit(X, y)
importance = pd.Series(np.abs(lasso.coef_), index=X.columns)
selected_features = importance[importance > 0].index
X = X[selected_features]
```

---

## III. 模型選擇與訓練 (Model Selection and Training)

### 模型選擇
- **根據任務類型**：  
  分類：LogisticRegression、RandomForest、XGBoost、LightGBM、NN  
  回歸：LinearRegression、Ridge、Lasso、RandomForest、GBDT
- **基於資料特性**：  
  高維資料：考慮正則化或降維  
  非線性：基於樹的模型或核方法  
  大規模資料：LightGBM、XGBoost

### 訓練策略
- **交叉驗證 (Cross-Validation)**：  
  K-Fold、Stratified K-Fold、Time Series CV、Group K-Fold 提升穩定性。
  
- **超參數調整 (Hyperparameter Tuning)**：  
  使用 Grid Search、Random Search、Optuna、Bayesian Optimization 自動尋找最佳參數組合。

- **早停 (Early Stopping)**、**學習率調整 (Learning Rate Scheduling)**：  
  防止過擬合與加快模型收斂。

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = Ridge(alpha=10)
scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=kf)
print("Mean CV RMSE:", -scores.mean())
```

---

## IV. 提升模型泛化能力 (Enhancing Model Generalization)

### 防止過擬合
- **特徵選擇與降維**：去冗餘特徵，降低模型複雜度。  
- **正則化 (Ridge、Lasso、ElasticNet)**：限制模型參數大小，避免過度擬合。  
- **Dropout (NN)**：降低神經網絡過擬合。  
- **資料增強 (Data Augmentation)**：對圖像、語音資料生成變形樣本增加多樣性。

### 交叉驗證策略
- **Stratified K-Fold**：保持類別比例一致。  
- **Time Series CV**：確保訓練集先於驗證集在時間上。  
- **Group K-Fold**：分組確保同群組不跨訓練、驗證集。

### 模型融合與集成學習 (Ensemble)
- **Bagging**：RandomForest  
- **Boosting**：XGBoost、LightGBM  
- **Stacking**：使用二階模型整合多模型預測。  
- **Blending、Weighted Average、Rank Averaging**：多模型結果組合提升穩定性。

---

## V. 其他實戰技巧
- **模型校準 (Calibration)**：使預測分佈更接近實際分佈。  
- **偽標籤 (Pseudo-Labeling)**：使用模型預測未標註資料的高信度預測值作為訓練補充。  
- **領域知識 (Domain Knowledge)**：加值於特徵工程與模型選擇。  
- **結果後處理 (Post-processing)**：依業務規則對最終預測結果微調。  
- **實驗與時間管理**：  
  設定里程碑、使用版本控制 (Git)、詳細記錄實驗參數、結果方便回溯。
- **學習他人經驗**：研究論壇、Kernel、Top 解決方案，不斷精進。

---

## VI. 常用工具與套件
- 資料處理：`pandas`, `numpy`  
- 可視化：`matplotlib`, `seaborn`, `plotly`  
- 機器學習：`scikit-learn`, `lightgbm`, `xgboost`  
- 深度學習：`tensorflow`, `pytorch`  
- 自動調參：`optuna`, `bayesian-optimization`  
- 實驗管理：`mlflow`, `wandb`

---

## VII. 注意事項
- **資料分佈差異**：檢查訓練/測試集分佈，避免資料洩漏。  
- **可重現性**：固定隨機種子、使用 Docker/虛擬環境。  
- **遵守比賽規則**：仔細閱讀官方要求。  
- **資源管理**：優化程式碼、合理分配運算資源。

---

## 總結
本筆記整合了從資料理解、特徵工程、模型選擇、泛化策略到集成方法和實務建議的完整流程。透過多次實驗迭代和嘗試，以及對領域知識與工具的充分運用，你將能在 Kaggle 競賽中不斷進步，並爭取更佳的排名表現。
