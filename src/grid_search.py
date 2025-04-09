from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

def grid_search_model(rf_model,x_train,y_train,x_test, y_test):
    # 定义网格搜索的参数网格
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    # 网格搜索
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(x_train, y_train)

    # 输出最佳参数和交叉验证得分
    print("最佳参数：", grid_search.best_params_)
    print("最佳交叉验证得分：", grid_search.best_score_)

    # 使用最佳参数的模型进行预测
    best_rf_model = grid_search.best_estimator_
    y_pred = best_rf_model.predict(x_test)

    # 输出预测准确率和分类报告
    print("准确率:", accuracy_score(y_test, y_pred))
    print("\n分类报告:\n", classification_report(y_test, y_pred))