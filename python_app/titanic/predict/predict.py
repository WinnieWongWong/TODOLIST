import pandas as pd

from sklearn.pipeline import Pipeline            # 用以包裝成pipeline
from sklearn.compose import ColumnTransformer    # 用以整合欄位進行轉換
from sklearn.impute import SimpleImputer         # 用以進行補值前處理
from sklearn.preprocessing import StandardScaler # 用以進行標準化前處理
from sklearn.preprocessing import OneHotEncoder  # 用以進行編碼前處理

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')


# 針對數值型特徵，採用補值及標準化轉換器
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# 針對類別型特徵，採用補值及編碼轉換器
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]) 
    # 若要使Pipeline中的Transformer以pandas格式輸出，則sparse_output需為False

# ColumnTransformer(transformers=[('欄位前贅字名稱', 轉換器 Transformer, [欲選取的欄位名稱])])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['Age', 'Fare']),
        ('cat', categorical_transformer, ['Embarked', 'Sex', 'Pclass'])
    ])

# 透過 ColumnTransformer 轉換的內容會以 numpy 呈現，
# 可使用 set_output 將 Transformer 的結果設定成 pandas 輸出
preprocessor.set_output(transform='pandas')

# 過濾掉暫時不處理的特徵和建立目標特徵
X_train = train_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y_train = train_data['Survived']
X_test = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# 將先前處理的 Transformer 與模型 Estimator 同樣透過一序列的 tuple 建置成 Pipeline
logreg = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', LogisticRegression())])

# 訓練模型
logreg.fit(X_train, y_train)

# 進行交叉驗證
## logreg 為將前處理 Transformer 及模型 Estimator 串接起來的 Pipeline，
## 給予資料則會經過前處理及模型預測
scores = cross_val_score(logreg, X_train, y_train, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Average cross-validation score: {scores.mean():.2f}")

# 使用網格搜尋法尋找最佳參數
param_grid = {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(logreg, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best cross-validation score: {grid.best_score_:.2f}")


# 預測測試集
y_pred = logreg.predict(X_test)


# 將預測結果輸出到CSV檔案中
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})

# Define the path where you want to save the CSV
file_path = '../data/submission.csv'

output.to_csv(file_path, index=False)



#print(train_data.head());

#print(train_data.info());
