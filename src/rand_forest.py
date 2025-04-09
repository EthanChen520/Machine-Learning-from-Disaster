import shap
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from grid_search import grid_search_model

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(current_dir, "../data"))
train_path = os.path.abspath(os.path.join(data_path, "train.csv"))
test_x_path = os.path.abspath(os.path.join(data_path, "test.csv"))
test_y_path = os.path.abspath(os.path.join(data_path, "gender_submission.csv"))

# 读取数据
train_df = pd.read_csv(train_path)
test_x_df = pd.read_csv(test_x_path)
test_y_df = pd.read_csv(test_y_path)
test_df = pd.merge(test_x_df, test_y_df, on='PassengerId', how='inner')

def clean_data(df):
    #处理缺失值

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Cabin'] = df['Cabin'].fillna('Unknown').str[0]

    # 使用 pd.cut 进行分组
    bins = [0, 12, 18, 35, 60, float('inf')]  # 定义分组区间
    labels = ['Child', 'Teenager', 'Young Adult', 'Middle-aged', 'Senior']  # 定义每个组的标签

    df['Age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    # 使用 LabelEncoder 进行编码
    le = LabelEncoder()
    df['Age_group_encoded'] = le.fit_transform(df['Age_group'])
    df['Sex_encoded'] = le.fit_transform(df['Sex'])
    df['Embarked_encoded'] = le.fit_transform(df['Embarked'])

    # 固定特征列
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
                'Age_group_encoded', 'Sex_encoded', 'Embarked_encoded']
    x = df[features]
    y = df["Survived"]
    return x, y


x_train, y_train = clean_data(train_df)
x_test, y_test = clean_data(test_df)

# 初始化 RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# # 网络搜索最佳参数
# grid_search_model(rf_model,x_train,y_train,x_test, y_test)

def best_parameter():
    # 使用最佳参数初始化 RandomForestClassifier
    best_rf_model = RandomForestClassifier(
        max_depth=10,
        max_features=None,
        min_samples_leaf=1,
        min_samples_split=10,
        n_estimators=100,
        random_state=42
    )

    # 训练模型
    best_rf_model.fit(x_train, y_train)

    # 使用训练好的模型进行预测
    y_pred = best_rf_model.predict(x_test)

    # 输出预测准确率和分类报告
    print("准确率:", accuracy_score(y_test, y_pred))
    print("\n分类报告:\n", classification_report(y_test, y_pred))

    return best_rf_model

# 获取训练好的最佳模型
best_rf_model = best_parameter()

# SHAP 分析（新版 API）
explainer = shap.Explainer(best_rf_model)
shap_values = explainer(x_test)
shap.plots.beeswarm(shap_values[:, :, 1])  # 或 summary_plot