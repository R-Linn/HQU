import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# 绘图配置 - 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def loadFeatureDataSets():
    """
    加载特征数据集（训练集/验证集/测试集）
    返回:
        XTrain: 训练集特征矩阵
        yTrain: 训练集标签
        XValidation: 验证集特征矩阵
        yValidation: 验证集标签
        XTest: 测试集特征矩阵
        yTest: 测试集标签
        featureColumns: 特征列名列表
    异常:
        FileNotFoundError: 特征文件未找到时抛出异常
    """
    try:
        trainingSet = pd.read_csv("TrainingSetWithFeatures.csv")
        validationSet = pd.read_csv("ValidationSetWithFeatures.csv")
        testSet = pd.read_csv("TestSetWithFeatures.csv")
        
        # 分离特征列和标签列（排除ID和标签列）
        featureColumns = [column for column in trainingSet.columns if column not in ["longNonCodingRnaIdentifier", "sampleLabel"]]
        
        featureMatrixTraining = trainingSet[featureColumns]
        labelTraining = trainingSet["sampleLabel"]
        featureMatrixValidation = validationSet[featureColumns]
        labelValidation = validationSet["sampleLabel"]
        featureMatrixTest = testSet[featureColumns]
        labelTest = testSet["sampleLabel"]
        
        return featureMatrixTraining, labelTraining, featureMatrixValidation, labelValidation, featureMatrixTest, labelTest, featureColumns
    
    except FileNotFoundError as error:
        print(f"文件未找到：{error.filename}，请先运行ExtractSequenceFeatures.py生成特征文件")
        raise

def evaluateFeaturePerformance(featureMatrixTraining, labelTraining, featureMatrixTest, labelTest):
    """
    使用逻辑回归模型评估特征性能（多指标）
    参数:
        featureMatrixTraining: 训练集特征矩阵
        labelTraining: 训练集标签
        featureMatrixTest: 测试集特征矩阵
        labelTest: 测试集标签
    返回:
        准确率(accuracy)、精确率(precision)、召回率(recall)、F1分数(f1)、AUC值(auc)（均保留4位小数）
    """
    # 初始化逻辑回归模型（设置随机种子保证结果可复现）
    logisticRegressionModel = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    logisticRegressionModel.fit(featureMatrixTraining, labelTraining)
    
    # 模型预测
    testSetPrediction = logisticRegressionModel.predict(featureMatrixTest)
    testSetProbability = logisticRegressionModel.predict_proba(featureMatrixTest)[:, 1]
    
    # 计算评估指标
    accuracy = accuracy_score(labelTest, testSetPrediction)
    precision = precision_score(labelTest, testSetPrediction)
    recall = recall_score(labelTest, testSetPrediction)
    f1Score = f1_score(labelTest, testSetPrediction)
    aucValue = roc_auc_score(labelTest, testSetProbability)
    
    return round(accuracy,4), round(precision,4), round(recall,4), round(f1Score,4), round(aucValue,4)

def featureSelection(featureMatrixTraining, labelTraining, featureMatrixValidation, featureMatrixTest, featureColumns):
    """
    特征筛选（方差分析+互信息法，论文要求）
    步骤:
        1. 方差分析：剔除恒定特征（方差<0.01）
        2. 互信息法：保留互信息值Top30的特征
    参数:
        featureMatrixTraining: 训练集特征矩阵
        labelTraining: 训练集标签
        featureMatrixValidation: 验证集特征矩阵
        featureMatrixTest: 测试集特征矩阵
        featureColumns: 原始特征列名列表
    返回:
        筛选后的训练/验证/测试集特征矩阵、选中的特征名列表
    """
    print("=== 开始特征筛选 ===")
    print(f"初始特征维度：{featureMatrixTraining.shape[1]}")
    
    # 1. 方差分析筛选：剔除方差小于0.01的恒定特征
    varianceThreshold = VarianceThreshold(threshold=0.01)
    featureMatrixTrainingVarianceFiltered = varianceThreshold.fit_transform(featureMatrixTraining)
    featureMatrixValidationVarianceFiltered = varianceThreshold.transform(featureMatrixValidation)
    featureMatrixTestVarianceFiltered = varianceThreshold.transform(featureMatrixTest)
    
    # 保留方差筛选后的特征名
    featureVarianceFiltered = [featureColumns[index] for index in varianceThreshold.get_support(indices=True)]
    print(f"方差筛选后特征维度：{len(featureVarianceFiltered)}")
    
    # 2. 互信息法筛选：计算特征与标签的互信息值
    mutualInformation = mutual_info_classif(featureMatrixTrainingVarianceFiltered, labelTraining, random_state=42)
    mutualInformationDataFrame = pd.DataFrame({
        "featureName": featureVarianceFiltered, 
        "mutualInformationValue": mutualInformation
    }).sort_values("mutualInformationValue", ascending=False)
    
    # 保存所有特征的互信息值
    mutualInformationDataFrame.to_csv("FeatureMutualInformation.csv", index=False)
    
    # 保留互信息值前30的特征（可根据需求调整top_n值）
    topFeatureNumber = 30
    topMutualInformationFeatures = mutualInformationDataFrame.head(topFeatureNumber)["featureName"].tolist()
    topMutualInformationIndex = [featureVarianceFiltered.index(feature) for feature in topMutualInformationFeatures]
    
    # 筛选最终特征矩阵
    featureMatrixTrainingFinal = featureMatrixTrainingVarianceFiltered[:, topMutualInformationIndex]
    featureMatrixValidationFinal = featureMatrixValidationVarianceFiltered[:, topMutualInformationIndex]
    featureMatrixTestFinal = featureMatrixTestVarianceFiltered[:, topMutualInformationIndex]
    
    print(f"互信息筛选后特征维度：{len(topMutualInformationFeatures)}")
    
    # 绘制Top10互信息值柱状图
    plt.figure(figsize=(12, 6))
    sns.barplot(x="mutualInformationValue", y="featureName", data=mutualInformationDataFrame.head(10))
    plt.title("Top 10 Features by Mutual Information", fontfamily='Arial', fontsize=14, fontweight='bold')
    plt.xlabel("Mutual Information Value", fontfamily='Arial')
    plt.ylabel("Feature Name", fontfamily='Arial')
    plt.xticks(fontfamily = "Arial")
    plt.yticks(fontfamily = "Arial")
    plt.tight_layout()
    plt.savefig("FeatureMutualInformationTop10.png", dpi=150)
    
    # 绘制特征维度变化图
    plt.figure(figsize=(8, 5))
    processingStages = ["初始特征", "方差筛选后", "互信息筛选后"]
    featureDimensions = [len(featureColumns), len(featureVarianceFiltered), len(topMutualInformationFeatures)]
    plt.bar(processingStages, featureDimensions, color=["#66c2a5", "#fc8d62", "#8da0cb"])
    plt.title("Feature Dimension Change After Filtering", fontfamily='Arial', fontsize=14, fontweight='bold')
    plt.ylabel("Feature Dimension", fontfamily='Arial')
    
    # 在柱状图上标注数值
    for index, value in enumerate(featureDimensions):
        plt.text(index, value+2, str(value), ha="center", fontfamily='Arial')
    
    plt.tight_layout()
    plt.savefig("FeatureDimensionChange.png", dpi=150)
    
    return featureMatrixTrainingFinal, featureMatrixValidationFinal, featureMatrixTestFinal, topMutualInformationFeatures

def main():
    # ===== 新增：调试标签分布 =====
    print("=== 检查数据集标签分布 ===")
    featureMatrixTraining, labelTraining, featureMatrixValidation, labelValidation, featureMatrixTest, labelTest, featureColumns = loadFeatureDataSets()
    
    # 统计各类别数量
    trainingLabelCounts = labelTraining.value_counts()
    validationLabelCounts = labelValidation.value_counts()
    testLabelCounts = labelTest.value_counts()
    
    print(f"训练集标签分布：{trainingLabelCounts.to_dict()}")
    print(f"验证集标签分布：{validationLabelCounts.to_dict()}")
    print(f"测试集标签分布：{testLabelCounts.to_dict()}")
    
    # 检查是否有至少2个类别
    if len(trainingLabelCounts) < 2:
        print("❌ 训练集只有单一类别，无法训练二分类模型！")
        return  # 终止脚本，避免报错
    # ===== 调试代码结束 =====

    # 1. 加载特征数据集
    featureMatrixTraining, labelTraining, featureMatrixValidation, labelValidation, featureMatrixTest, labelTest, featureColumns = loadFeatureDataSets()
    
    # 2. 评估原始特征性能
    originalAccuracy, originalPrecision, originalRecall, originalF1Score, originalAucValue = evaluateFeaturePerformance(
        featureMatrixTraining, labelTraining, featureMatrixTest, labelTest
    )
    print("\n=== 原始特征性能 ===")
    print(f"准确率: {originalAccuracy} | 精确率: {originalPrecision} | 召回率: {originalRecall} | F1分数: {originalF1Score} | AUC值: {originalAucValue}")
    
    # 3. 特征筛选
    featureMatrixTrainingFinal, featureMatrixValidationFinal, featureMatrixTestFinal, selectedFeatureNames = featureSelection(
        featureMatrixTraining, labelTraining, featureMatrixValidation, featureMatrixTest, featureColumns
    )
    
    # 4. 评估筛选后特征性能
    selectedAccuracy, selectedPrecision, selectedRecall, selectedF1Score, selectedAucValue = evaluateFeaturePerformance(
        featureMatrixTrainingFinal, labelTraining, featureMatrixTestFinal, labelTest
    )
    print("\n=== 筛选后特征性能 ===")
    print(f"准确率: {selectedAccuracy} | 精确率: {selectedPrecision} | 召回率: {selectedRecall} | F1分数: {selectedF1Score} | AUC值: {selectedAucValue}")
    
    # 5. 保存筛选后的特征数据（保留ID和标签）
    trainingSetOriginal = pd.read_csv("TrainingSetWithFeatures.csv")
    validationSetOriginal = pd.read_csv("ValidationSetWithFeatures.csv")
    testSetOriginal = pd.read_csv("TestSetWithFeatures.csv")
    
    # 构建筛选后的完整数据集
    trainingSetSelectedFeatures = trainingSetOriginal[["longNonCodingRnaIdentifier", "sampleLabel"] + selectedFeatureNames]
    validationSetSelectedFeatures = validationSetOriginal[["longNonCodingRnaIdentifier", "sampleLabel"] + selectedFeatureNames]
    testSetSelectedFeatures = testSetOriginal[["longNonCodingRnaIdentifier", "sampleLabel"] + selectedFeatureNames]
    
    # 保存文件
    trainingSetSelectedFeatures.to_csv("TrainingSetSelectedFeatures.csv", index=False)
    validationSetSelectedFeatures.to_csv("ValidationSetSelectedFeatures.csv", index=False)
    testSetSelectedFeatures.to_csv("TestSetSelectedFeatures.csv", index=False)
    
    # 保存选中的特征名（供后续模型训练使用）
    joblib.dump(selectedFeatureNames, "SelectedFeatureNames.pkl")
    
    print("\n特征筛选完成，保存文件：")
    print("   - TrainingSetSelectedFeatures.csv")
    print("   - ValidationSetSelectedFeatures.csv")
    print("   - TestSetSelectedFeatures.csv")
    print("   - SelectedFeatureNames.pkl")
    print("   - FeatureMutualInformation.csv")
    print("   - FeatureMutualInformationTop10.png")
    print("   - FeatureDimensionChange.png")

if __name__ == "__main__":
    main()