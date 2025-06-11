import pandas as pd;

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv("data/malicious_dataset.csv");
test_data= pd.read_csv("data/test.csv");

# Check column names
print(train_data.columns)
print(test_data.columns)

#Wy to handle data

#When data miss in "Num value" , it will be filled with middle value
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

#When data miss in "Category" , it will be filled with frequent value
#handle_unknown='ignore' parameter ensures that if there are any unknown categories during transformation, they are ignored rather than causing an error. The sparse_output=False option means the output will be a dense array rather than a sparse matrix.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]) 

#put each column into transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, ['Method', 'URL', 'Headers', 'Body', 'Payload_type', 'Malicious'])
])


preprocessor.set_output(transform='pandas')

#Filter away column (Feature) that will not handled.
X_train = train_data.drop(['ID', 'Notes', 'Desciption'], axis=1)
y_train = train_data['Malicious']
X_test = test_data.drop(['ID', 'Notes', 'Desciption'], axis=1)

logreg = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', LogisticRegression())])


# 訓練模型
logreg.fit(X_train, y_train)

#Perform cross validation using logistic regression model -> logreg
#plitting the training data into 5 subsets (as specified by cv=5). It trains the model on 4 of these subsets and tests it on the remaining one, repeating this process for all subsets
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
output = pd.DataFrame({'ID': test_data.ID, 'Malicious': y_pred})

# Define the path where you want to save the CSV
file_path = 'data/malicious_submission.csv'

output.to_csv(file_path, index=False)


