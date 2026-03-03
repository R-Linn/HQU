import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# 绘图配置 - 设置中文字体和负号正常显示（全局配置字体，避免重复设置）
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 优先Arial，其次黑体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'  # 全局设置字体为Arial

def loadData():
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
        
        # 分离特征列和标签列（适配前序重构的标签列名）
        featureMatrixTraining = trainingSet[selectedFeatureNames]
        labelTraining = trainingSet["sampleLabel"]
        featureMatrixValidation = validationSet[selectedFeatureNames]
        labelValidation = validationSet["sampleLabel"]
        featureMatrixTest = testSet[selectedFeatureNames]
        labelTest = testSet["sampleLabel"]
        
        return featureMatrixTraining, labelTraining, featureMatrixValidation, labelValidation, featureMatrixTest, labelTest, selectedFeatureNames
    
    except FileNotFoundError as error:
        print(f"❌ 文件未找到：{error.filename}，请先运行FeatureSelectionAndEvaluation.py生成筛选后的特征文件")
        raise

def trainBaseModels(featureMatrixTraining, labelTraining, featureMatrixValidation, labelValidation, featureMatrixTest, labelTest):
    """
    训练基础模型（逻辑回归LR/支持向量机SVM/随机森林RF，论文要求）
    参数:
        featureMatrixTraining: 训练集特征矩阵
        labelTraining: 训练集标签
        featureMatrixValidation: 验证集特征矩阵
        labelValidation: 验证集标签
        featureMatrixTest: 测试集特征矩阵
        labelTest: 测试集标签
    返回:
        trainedModels: 训练好的基础模型字典
        performanceDataFrame: 基础模型性能DataFrame
        modelProbabilityDictionary: 基础模型测试集预测概率字典
    """
    print("=== 开始训练基础模型 ===")
    
    # 定义基础模型（固定参数，论文要求配置）
    machineLearningModels = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000, C=0.01, penalty='l2', solver='liblinear'),
        "SupportVectorMachine": SVC(kernel="rbf", probability=True, random_state=42, C=0.1, gamma="scale"),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    }
    
    # 训练并评估每个基础模型
    modelPerformanceList = []
    modelProbabilityDictionary = {}
    trainedModels = {}
    
    for modelName, model in machineLearningModels.items():
        print(f"\n训练 {modelName}...")
        
        # 模型训练
        model.fit(featureMatrixTraining, labelTraining)
        trainedModels[modelName] = model
        
        # 模型预测（验证集+测试集）
        validationPrediction = model.predict(featureMatrixValidation)
        testPrediction = model.predict(featureMatrixTest)
        validationProbability = model.predict_proba(featureMatrixValidation)[:, 1]
        testProbability = model.predict_proba(featureMatrixTest)[:, 1]
        modelProbabilityDictionary[modelName] = testProbability
        
        # 计算评估指标
        validationAccuracy = accuracy_score(labelValidation, validationPrediction)
        validationPrecision = precision_score(labelValidation, validationPrediction)
        validationRecall = recall_score(labelValidation, validationPrediction)
        validationF1Score = f1_score(labelValidation, validationPrediction)
        validationAucValue = roc_auc_score(labelValidation, validationProbability)
        
        testAccuracy = accuracy_score(labelTest, testPrediction)
        testPrecision = precision_score(labelTest, testPrediction)
        testRecall = recall_score(labelTest, testPrediction)
        testF1Score = f1_score(labelTest, testPrediction)
        testAucValue = roc_auc_score(labelTest, testProbability)
        
        # 保存单模型性能数据
        modelPerformanceList.append({
            "modelName": modelName,
            "validationAccuracy": validationAccuracy,
            "validationPrecision": validationPrecision,
            "validationRecall": validationRecall,
            "validationF1Score": validationF1Score,
            "validationAucValue": validationAucValue,
            "testSetAccuracy": testAccuracy,
            "testSetPrecision": testPrecision,
            "testSetRecall": testRecall,
            "testSetF1Score": testF1Score,
            "testSetAreaUnderCurve": testAucValue
        })
    
    # 转换为DataFrame并保存
    performanceDataFrame = pd.DataFrame(modelPerformanceList)
    performanceDataFrame = performanceDataFrame.round(4)
    performanceDataFrame.to_csv("BaseModelsPerformance.csv", index=False)
    
    # 绘制模型性能对比图
    plotModelPerformanceComparison(performanceDataFrame)
    
    return trainedModels, performanceDataFrame, modelProbabilityDictionary

def trainStackingModel(featureMatrixTraining, labelTraining, featureMatrixValidation, labelValidation, featureMatrixTest, labelTest, baseModels):
    """
    训练Stacking集成模型（以逻辑回归为最终分类器）
    参数:
        featureMatrixTraining: 训练集特征矩阵
        labelTraining: 训练集标签
        featureMatrixValidation: 验证集特征矩阵
        labelValidation: 验证集标签
        featureMatrixTest: 测试集特征矩阵
        labelTest: 测试集标签
        baseModels: 训练好的基础模型字典
    返回:
        stackingModel: 训练好的Stacking集成模型
        stackingTestProbability: Stacking模型测试集预测概率
    """
    print("\n=== 开始训练Stacking集成模型 ===")
    
    # 定义Stacking模型的基础估计器
    stackingEstimators = [(modelName, model) for modelName, model in baseModels.items()]
    stackingModel = StackingClassifier(
        estimators=stackingEstimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=5  # 5折交叉验证
    )
    
    # 训练Stacking模型
    stackingModel.fit(featureMatrixTraining, labelTraining)
    
    # 模型预测与评估
    validationPrediction = stackingModel.predict(featureMatrixValidation)
    testPrediction = stackingModel.predict(featureMatrixTest)
    validationProbability = stackingModel.predict_proba(featureMatrixValidation)[:, 1]
    testProbability = stackingModel.predict_proba(featureMatrixTest)[:, 1]
    
    validationAccuracy = accuracy_score(labelValidation, validationPrediction)
    validationPrecision = precision_score(labelValidation, validationPrediction)
    validationRecall = recall_score(labelValidation, validationPrediction)
    validationF1Score = f1_score(labelValidation, validationPrediction)
    validationAucValue = roc_auc_score(labelValidation, validationProbability)
    
    testAccuracy = accuracy_score(labelTest, testPrediction)
    testPrecision = precision_score(labelTest, testPrediction)
    testRecall = recall_score(labelTest, testPrediction)
    testF1Score = f1_score(labelTest, testPrediction)
    testAucValue = roc_auc_score(labelTest, testProbability)
    
    # 保存Stacking模型性能
    stackingPerformance = pd.DataFrame({
        "modelName": ["StackingEnsemble"],
        "validationAccuracy": [validationAccuracy],
        "validationPrecision": [validationPrecision],
        "validationRecall": [validationRecall],
        "validationF1Score": [validationF1Score],
        "validationAucValue": [validationAucValue],
        "testSetAccuracy": [testAccuracy],
        "testSetPrecision": [testPrecision],
        "testSetRecall": [testRecall],
        "testSetF1Score": [testF1Score],
        "testSetAreaUnderCurve": [testAucValue]
    }).round(4)
    
    # 合并基础模型和Stacking模型性能数据
    basePerformance = pd.read_csv("BaseModelsPerformance.csv")
    allPerformance = pd.concat([basePerformance, stackingPerformance], ignore_index=True)
    allPerformance.to_csv("AllModelsPerformance.csv", index=False)
    
    # 输出Stacking模型核心性能指标
    print("\n=== Stacking集成模型性能 ===")
    print(f"验证集AUC：{validationAucValue:.4f} | 测试集AUC：{testAucValue:.4f}")
    print(f"验证集F1：{validationF1Score:.4f} | 测试集F1：{testF1Score:.4f}")
    
    return stackingModel, testProbability

def plotModelPerformanceComparison(performanceDataFrame):
    """
    绘制模型性能对比柱状图（AUC+Accuracy，论文要求）
    参数:
        performanceDataFrame: 模型性能DataFrame
    """
    plt.figure(figsize=(12, 6))
    
    # 提取绘图数据
    modelNames = performanceDataFrame["modelName"].tolist()
    aucScores = performanceDataFrame["testSetAreaUnderCurve"].tolist()
    accuracyScores = performanceDataFrame["testSetAccuracy"].tolist()
    
    # 绘制双柱状图
    xAxis = np.arange(len(modelNames))
    barWidth = 0.35
    
    plt.bar(xAxis - barWidth/2, aucScores, barWidth, label="AUC", color="#66c2a5")
    plt.bar(xAxis + barWidth/2, accuracyScores, barWidth, label="Accuracy", color="#fc8d62")
    
    # 图表样式优化（移除fontfamily参数，使用全局配置）
    plt.xlabel("Model Name", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Model Performance Comparison (Test Set)", fontsize=14, fontweight='bold')
    plt.xticks(xAxis, modelNames, rotation=15, ha="right")
    plt.legend(fontsize=10)  # 仅设置字体大小，字体由全局配置控制
    
    # 为每个柱子添加数值标注
    for index, (aucValue, accuracyValue) in enumerate(zip(aucScores, accuracyScores)):
        plt.text(index - barWidth/2, aucValue+0.02, f"{aucValue:.4f}", ha="center", fontsize=10)
        plt.text(index + barWidth/2, accuracyValue+0.02, f"{accuracyValue:.4f}", ha="center", fontsize=10)
    
    plt.tight_layout()
    plt.savefig("ModelPerformanceComparison.png", dpi=150)

def plotFinalRocCurve(modelProbabilityDictionary, testLabel):
    """
    绘制最终ROC曲线（所有模型对比，论文要求优化样式）
    参数:
        modelProbabilityDictionary: 所有模型的测试集预测概率字典
        testLabel: 测试集真实标签
    """
    plt.figure(figsize=(10, 8))
    colorList = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    # 绘制每个模型的ROC曲线
    for index, (modelName, probability) in enumerate(modelProbabilityDictionary.items()):
        falsePositiveRate, truePositiveRate, _ = roc_curve(testLabel, probability)
        aucValue = roc_auc_score(testLabel, probability)
        plt.plot(falsePositiveRate, truePositiveRate, label=f"{modelName} (AUC={aucValue:.4f})", linewidth=1.2, color=colorList[index])
    
    # 绘制随机猜测基线
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess", linewidth=1.0)
    
    # 图表样式优化（移除fontfamily参数，使用全局配置）
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title("ROC Curves of Different Machine Learning Models (Test Set)", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)  # 仅设置位置和大小
    plt.tight_layout()
    
    # 保存图片（防止标签截断）
    plt.savefig("FinalROCCurve.png", dpi=150, bbox_inches="tight")
    
    print("\n✅ ROC曲线已保存：FinalROCCurve.png")

def main():
    # 1. 加载筛选后的特征数据
    featureMatrixTraining, labelTraining, featureMatrixValidation, labelValidation, featureMatrixTest, labelTest, selectedFeatureNames = loadData()
    
    # 2. 训练基础模型（LR/SVM/RF）
    baseModels, basePerformance, modelProbabilityDictionary = trainBaseModels(
        featureMatrixTraining, labelTraining, featureMatrixValidation, labelValidation, featureMatrixTest, labelTest
    )
    
    # 3. 训练Stacking集成模型
    stackingModel, stackingTestProbability = trainStackingModel(
        featureMatrixTraining, labelTraining, featureMatrixValidation, labelValidation, featureMatrixTest, labelTest, baseModels
    )
    
    # 4. 更新概率字典，添加Stacking模型的预测概率
    modelProbabilityDictionary["StackingEnsemble"] = stackingTestProbability
    
    # 5. 绘制最终ROC曲线（所有模型对比）
    plotFinalRocCurve(modelProbabilityDictionary, labelTest)
    
    # 6. 保存所有训练好的模型
    joblib.dump(baseModels, "BaseModels.pkl")
    joblib.dump(stackingModel, "StackingEnsembleModel.pkl")
    
    # 输出保存文件列表
    print("\n✅ 多模型训练完成，保存文件：")
    print("   - BaseModels.pkl")
    print("   - StackingEnsembleModel.pkl")
    print("   - BaseModelsPerformance.csv")
    print("   - AllModelsPerformance.csv")
    print("   - ModelPerformanceComparison.png")
    print("   - FinalROCCurve.png")

if __name__ == "__main__":
    main()