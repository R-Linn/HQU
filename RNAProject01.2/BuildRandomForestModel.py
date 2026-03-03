import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# 绘图配置 - 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def loadSelectedFeatures():
    """
    加载筛选后的特征数据
    返回:
        featureMatrixTraining: 训练集特征矩阵
        labelTraining: 训练集标签
        featureMatrixValidation: 验证集特征矩阵
        labelValidation: 验证集标签
        featureMatrixTest: 测试集特征矩阵
        labelTest: 测试集标签
        selectedFeatureNames: 选中的特征名列表
    异常:
        FileNotFoundError: 特征文件未找到时抛出异常
    """
    try:
        trainingSet = pd.read_csv("TrainingSetSelectedFeatures.csv")
        validationSet = pd.read_csv("ValidationSetSelectedFeatures.csv")
        testSet = pd.read_csv("TestSetSelectedFeatures.csv")
        selectedFeatureNames = joblib.load("SelectedFeatureNames.pkl")
        
        # 分离特征列和标签列
        featureMatrixTraining = trainingSet[selectedFeatureNames]
        labelTraining = trainingSet["sampleLabel"]  # 适配前序重构的标签列名
        featureMatrixValidation = validationSet[selectedFeatureNames]
        labelValidation = validationSet["sampleLabel"]
        featureMatrixTest = testSet[selectedFeatureNames]
        labelTest = testSet["sampleLabel"]
        
        return featureMatrixTraining, labelTraining, featureMatrixValidation, labelValidation, featureMatrixTest, labelTest, selectedFeatureNames
    
    except FileNotFoundError as error:
        print(f"`文件未找到：{error.filename}，请先运行FeatureSelectionAndEvaluation.py生成筛选后的特征文件")
        raise

def trainRandomForestWithGridSearch(featureMatrixTraining, labelTraining):
    """
    网格搜索优化随机森林模型（论文要求）
    使用5折交叉验证，以AUC为评分指标优化模型参数
    参数:
        featureMatrixTraining: 训练集特征矩阵
        labelTraining: 训练集标签
    返回:
        网格搜索得到的最优随机森林模型
    """
    print("=== 开始随机森林模型优化（网格搜索）===")
    
    # 定义参数网格（可根据需求调整搜索范围）
    parameterGrid = {
        "n_estimators": [50, 100, 200],          # 决策树数量
        "max_depth": [8, 10, 12],                # 决策树最大深度
        "min_samples_split": [2, 5, 10],         # 内部节点划分所需最小样本数
        "min_samples_leaf": [1, 2, 4]            # 叶节点所需最小样本数
    }
    
    # 初始化随机森林模型（设置随机种子保证结果可复现）
    randomForest = RandomForestClassifier(random_state=42, n_jobs=1)
    
    # 网格搜索（5折交叉验证）
    gridSearch = GridSearchCV(
        estimator=randomForest,
        param_grid=parameterGrid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1, # 调整为-1，全CPU调度运行
        verbose=1
    )
    
    gridSearch.fit(featureMatrixTraining, labelTraining)
    
    # 获取最优模型
    bestRandomForestModel = gridSearch.best_estimator_
    
    # 保存最优参数
    bestParameters = pd.DataFrame(gridSearch.best_params_, index=[0])
    bestParameters.to_csv("RFBestParameters.csv", index=False)
    
    print(f"\n网格搜索完成：")
    print(f"最优参数：{gridSearch.best_params_}")
    print(f"交叉验证最优AUC：{gridSearch.best_score_:.4f}")
    
    return bestRandomForestModel

def evaluateModel(model, featureMatrixValidation, labelValidation, featureMatrixTest, labelTest):
    """
    多指标评估模型性能（验证集+测试集）
    参数:
        model: 训练好的机器学习模型
        featureMatrixValidation: 验证集特征矩阵
        labelValidation: 验证集标签
        featureMatrixTest: 测试集特征矩阵
        labelTest: 测试集标签
    返回:
        validationProbability: 验证集预测概率
        testProbability: 测试集预测概率
    """
    # 验证集评估
    validationPrediction = model.predict(featureMatrixValidation)
    validationProbability = model.predict_proba(featureMatrixValidation)[:, 1]
    validationAccuracy = accuracy_score(labelValidation, validationPrediction)
    validationPrecision = precision_score(labelValidation, validationPrediction)
    validationRecall = recall_score(labelValidation, validationPrediction)
    validationF1Score = f1_score(labelValidation, validationPrediction)
    validationAucValue = roc_auc_score(labelValidation, validationProbability)
    
    # 测试集评估
    testPrediction = model.predict(featureMatrixTest)
    testProbability = model.predict_proba(featureMatrixTest)[:, 1]
    testAccuracy = accuracy_score(labelTest, testPrediction)
    testPrecision = precision_score(labelTest, testPrediction)
    testRecall = recall_score(labelTest, testPrediction)
    testF1Score = f1_score(labelTest, testPrediction)
    testAucValue = roc_auc_score(labelTest, testProbability)
    
    # 保存性能指标
    performanceMetrics = pd.DataFrame({
        "metricName": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
        "validationSet": [validationAccuracy, validationPrecision, validationRecall, validationF1Score, validationAucValue],
        "testSet": [testAccuracy, testPrecision, testRecall, testF1Score, testAucValue]
    })
    performanceMetrics = performanceMetrics.round(4)
    performanceMetrics.to_csv("RFPerformanceMetrics.csv", index=False)
    
    # 输出评估结果
    print("\n=== 随机森林模型性能评估 ===")
    print("验证集：")
    print(f"  准确率(Accuracy)：{validationAccuracy:.4f}")
    print(f"  精确率(Precision)：{validationPrecision:.4f}")
    print(f"  召回率(Recall)：{validationRecall:.4f}")
    print(f"  F1值(F1-Score)：{validationF1Score:.4f}")
    print(f"  AUC值：{validationAucValue:.4f}")
    
    print("测试集：")
    print(f"  准确率(Accuracy)：{testAccuracy:.4f}")
    print(f"  精确率(Precision)：{testPrecision:.4f}")
    print(f"  召回率(Recall)：{testRecall:.4f}")
    print(f"  F1值(F1-Score)：{testF1Score:.4f}")
    print(f"  AUC值：{testAucValue:.4f}")
    
    return validationProbability, testProbability

def plotFeatureImportance(model, featureNames):
    """
    绘制特征重要性图（Top10），优化样式适配论文要求
    参数:
        model: 训练好的随机森林模型
        featureNames: 特征名列表
    """
    # 获取特征重要性并排序
    featureImportanceValues = model.feature_importances_
    sortedIndices = np.argsort(featureImportanceValues)[::-1]
    
    # 整理特征重要性数据
    featureImportanceDataFrame = pd.DataFrame({
        "featureName": [featureNames[index] for index in sortedIndices],
        "importanceValue": [featureImportanceValues[index] for index in sortedIndices]
    })
    
    # 保存完整的特征重要性数据
    featureImportanceDataFrame.to_csv("RFFeatureImportance.csv", index=False)
    
    # 绘制Top10特征重要性水平柱状图
    plt.figure(figsize=(12, 8))
    topTenFeatureDataFrame = featureImportanceDataFrame.head(10)
    bars = plt.barh(topTenFeatureDataFrame["featureName"], topTenFeatureDataFrame["importanceValue"], color="#1f77b4")
    
    # 为每个柱子添加数值标注
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.4f}", ha="left", va="center")
    
    # 图表样式优化
    plt.xlabel("Feature Importance Value", fontsize=12, fontfamily='Arial')
    plt.ylabel("Feature Name", fontsize=12, fontfamily='Arial')
    plt.title("Top 10 Feature Importance of Random Forest Model", fontsize=14, fontweight="bold", fontfamily='Arial')
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    # 保存图片（bbox_inches="tight"防止标签被截断）
    plt.savefig("LongNonCodingRnaFeatureImportance.png", dpi=150, bbox_inches="tight")
    
    print("\n特征重要性图已保存：LongNonCodingRnaFeatureImportance.png")

def main():
    # 1. 加载筛选后的特征数据
    featureMatrixTraining, labelTraining, featureMatrixValidation, labelValidation, featureMatrixTest, labelTest, selectedFeatureNames = loadSelectedFeatures()
    
    # 2. 训练优化后的随机森林模型
    bestRandomForestModel = trainRandomForestWithGridSearch(featureMatrixTraining, labelTraining)
    
    # 3. 评估模型性能（验证集+测试集）
    validationProbability, testProbability = evaluateModel(bestRandomForestModel, featureMatrixValidation, labelValidation, featureMatrixTest, labelTest)
    
    # 4. 绘制特征重要性图
    plotFeatureImportance(bestRandomForestModel, selectedFeatureNames)
    
    # 5. 保存模型和预测结果
    joblib.dump(bestRandomForestModel, "OptimizedRandomForestModel.pkl")
    np.save("RFValidationProbabilities.npy", validationProbability)
    np.save("RFTestProbabilities.npy", testProbability)
    
    print("\n随机森林模型训练完成，保存文件：")
    print("   - OptimizedRandomForestModel.pkl")
    print("   - RFBestParameters.csv")
    print("   - RFPerformanceMetrics.csv")
    print("   - RFFeatureImportance.csv")
    print("   - LongNonCodingRnaFeatureImportance.png")

if __name__ == "__main__":
    main()