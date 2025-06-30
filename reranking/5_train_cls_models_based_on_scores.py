import json

sample = {
    "semantic": 0.7,
    "fluency": 0.8,
    "phono_sim": 0.8,
    "glyph_sim": 0.8,
}

import numpy as np
import pandas as pd

# --------------------------- 训练数据构造 ---------------------------
# 生成示例数据


# 转换为DataFrame
df = pd.read_csv('scores.csv', encoding='utf-8')
nan_rows = df[df.isnull().any(axis=1)]
print(nan_rows.to_dict())
print(df)
text_df = df['text']
df.drop('text', axis=1, inplace=True)
df.drop('text_1', axis=1, inplace=True)
X = df.drop("label", axis=1)
print(X)

y = df["label"]
print(y)
# --------------------------- 模型训练 ---------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
import os



features = X[['semantic','fluency','phono_sim', 'glyph_sim', 'pinyin_sim', 'component_sim', 'stroke_sim',
              'semantic_1', 'fluency_1', 'phono_sim_1', 'glyph_sim_1', 'pinyin_sim_1', 'component_sim_1',
                            'stroke_sim_1', ]]

# 划分训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.8, random_state=42)

# 定义模型
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500),
    "Random Forest": RandomForestClassifier(),
}
save_directory = "saved_models"
os.makedirs(save_directory, exist_ok=True)  # 如果目录不存在，创建它

# 训练和评估模型
for name, model in models.items():
    # params = param_grids.get(name, {})
    # if not params:
    #     continue
    # grid_search = GridSearchCV(model, params, cv=5, scoring="accuracy", n_jobs=-1)
    # grid_search.fit(X_train, y_train)
    # grid_search.fit(X, y)

    # best_model = grid_search.best_estimator_

    # model.fit(X, y)
    model.fit(X_train, y_train)

    # 替换文件名中的特殊字符，例如空格和括号
    filename = name.replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
    save_path = os.path.join(save_directory, filename)

    # 保存模型
    # joblib.dump(best_model, save_path)
    joblib.dump(model, save_path)
    print(f"Model '{name}' saved to {save_path}")

    # y_pred = best_model.predict(X_test)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")

# --------------------------- 预测 ---------------------------
# 新样本
# new_sample = {
#     "semantic": 0.7,
#     "fluency": 0.8,
#     "phono_sim": 0.8,
#     "glyph_sim": 0.8,
# }

# # 转换为DataFrame
# new_df = pd.DataFrame([new_sample])
# features = new_df[["phono_sim", "glyph_sim",'pinyin_sim','component_sim','freq_sim','stroke_sim', "semantic", "fluency"]]



models_name_set = {
    "Logistic Regression",
    "SVM (RBF Kernel)",
    "Neural Network",
    "Random Forest",
}
# # 使用训练好的Logistic Regression模型预测
# # 你可以选择其他模型
for model_name in models_name_set:
    model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    print(model_name)
    model = joblib.load(os.path.join(save_directory, f"{model_name}.joblib"))
    out = {}
    with open(f'scores_{model_name}.json', 'w', encoding='utf-8') as f:
        predictions = model.predict(features)

        # if hasattr(model, "decision_function"):
        #     probabilities = model.decision_function(features)
        # elif hasattr(model, "predict_proba"):
        #     probabilities = model.predict_proba(features)
        probabilities = model.predict_proba(features)
        #
        # print(f"Prediction: {probability} (Label {prediction})")
        # print(f"Probability of Label 1: {probability:.4f}")
        for prediction, probability, text in zip(predictions, probabilities, text_df.tolist()):
            # print(prediction, probability[1], text)
            out[text] = probability[1]
        f.write(json.dumps(out, ensure_ascii=False, indent=4))

