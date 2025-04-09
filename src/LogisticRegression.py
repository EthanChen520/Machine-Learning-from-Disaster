import shap
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(current_dir, "../data"))
train_path = os.path.join(data_path, "train.csv")
test_x_path = os.path.join(data_path, "test.csv")
test_y_path = os.path.join(data_path, "gender_submission.csv")

# 读取数据
train_df = pd.read_csv(train_path)
test_x_df = pd.read_csv(test_x_path)
test_y_df = pd.read_csv(test_y_path)
test_df = pd.merge(test_x_df, test_y_df, on='PassengerId', how='inner')


def preprocess_data(train_df, test_df, scale=False):
    # 复制数据避免修改原始DataFrame
    train = train_df.copy()
    test = test_df.copy()

    # 定义预处理参数（基于训练数据）
    age_median = train['Age'].median()
    fare_median = train['Fare'].median()
    embarked_mode = train['Embarked'].mode()[0]

    # 填充缺失值（统一用训练集的参数）
    for df in [train, test]:
        df['Age'].fillna(age_median, inplace=True)
        df['Fare'].fillna(fare_median, inplace=True)
        df['Embarked'].fillna(embarked_mode, inplace=True)
        df['Cabin'] = df['Cabin'].fillna('Unknown').str[0]

    # 年龄分箱（统一用训练集的bins）
    bins = [0, 12, 18, 35, 60, float('inf')]
    labels = ['Child', 'Teenager', 'Young Adult', 'Middle-aged', 'Senior']
    train['Age_group'] = pd.cut(train['Age'], bins=bins, labels=labels, right=False)
    test['Age_group'] = pd.cut(test['Age'], bins=bins, labels=labels, right=False)

    # 为每个类别特征创建独立的LabelEncoder
    encoders = {
        'Age_group': LabelEncoder(),
        'Sex': LabelEncoder(),
        'Embarked': LabelEncoder()
    }

    # 编码训练集和测试集
    train['Age_group_encoded'] = encoders['Age_group'].fit_transform(train['Age_group'])
    test['Age_group_encoded'] = encoders['Age_group'].transform(test['Age_group'])

    train['Sex_encoded'] = encoders['Sex'].fit_transform(train['Sex'])
    test['Sex_encoded'] = encoders['Sex'].transform(test['Sex'])

    train['Embarked_encoded'] = encoders['Embarked'].fit_transform(train['Embarked'])
    test['Embarked_encoded'] = encoders['Embarked'].transform(test['Embarked'])

    # 特征选择
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
               'Age_group_encoded', 'Sex_encoded', 'Embarked_encoded']

    X_train = train[features]
    y_train = train['Survived']
    X_test = test[features]
    y_test = test['Survived']

    # 根据参数决定是否标准化
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.values
        X_test = X_test.values

    return X_train, y_train, X_test, y_test

# 数据预处理
X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df, scale=False)


# 训练模型
best_lr_model = LogisticRegression(
    C=0.1, solver='liblinear', random_state=42, max_iter=200
)
best_lr_model.fit(X_train, y_train)

# 评估
y_pred = best_lr_model.predict(X_test)
print("准确率:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))