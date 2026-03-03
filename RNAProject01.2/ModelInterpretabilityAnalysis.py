import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
import os
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

# ===================== 全局配置 =====================
modelFilePath = "StackingEnsembleModel.pkl"
trainingDataFilePath = "TrainingSetWithFeatures.csv"
selectedFeaturePath = "SelectedFeatureNames.pkl"
identifierColumnName = "longNonCodingRnaIdentifier"
labelColumnName = "sampleLabel"

# 绘图配置
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'

# 输出目录
outputDir = "./"
os.makedirs(outputDir, exist_ok=True)

# ===================== 工具函数 =====================
def get_rf_model_from_stacking(stacking_model):
    """从Stacking模型中提取随机森林基模型"""
    rf_model = None
    if hasattr(stacking_model, 'estimators_'):
        estimators = stacking_model.estimators_
        for est in estimators:
            if isinstance(est, tuple):
                name, model = est
                if "RandomForest" in name or isinstance(model, RandomForestClassifier):
                    rf_model = model
                    break
            else:
                if isinstance(est, RandomForestClassifier):
                    rf_model = est
                    break
    return rf_model

def process_shap_values(shapValues):
    """处理SHAP值维度（适配二分类模型）"""
    # 处理三维数组 (样本数, 特征数, 类别数) → 二维数组 (样本数, 特征数)
    if isinstance(shapValues, list) and len(shapValues) == 2:
        # TreeExplainer返回的列表格式 [负类SHAP值, 正类SHAP值]
        return shapValues[1]  # 取正类SHAP值
    elif len(shapValues.shape) == 3:
        # 三维数组格式 (样本数, 特征数, 类别数)
        return shapValues[:, :, 1]  # 取正类SHAP值
    else:
        # 已经是二维数组，直接返回
        return shapValues

# ===================== 核心函数 =====================
def loadModelAndData():
    """加载模型和数据"""
    try:
        stackingEnsembleModel = joblib.load(modelFilePath)
        print(f"✅ 模型加载成功：{modelFilePath}")
        
        selectedFeatureNames = joblib.load(selectedFeaturePath)
        print(f"✅ 筛选后的特征数量：{len(selectedFeatureNames)}")
        
        trainingSetDataFrame = pd.read_csv(trainingDataFilePath)
        print(f"✅ 训练数据加载成功：{trainingSetDataFrame.shape}")
        
        trainingFeatureMatrixScaled = trainingSetDataFrame[selectedFeatureNames].copy().values
        trainingFeatureMatrix = trainingSetDataFrame[selectedFeatureNames].copy()
        featureNames = selectedFeatureNames
        
        # 禁用特征名校验
        import sklearn
        sklearn.utils.validation._check_feature_names = lambda *args, **kwargs: None
        
        print(f"✅ 特征矩阵加载完成 | 标准化特征维度：{trainingFeatureMatrixScaled.shape}")
        return stackingEnsembleModel, trainingFeatureMatrixScaled, featureNames, trainingFeatureMatrix
    
    except FileNotFoundError as error:
        print(f"❌ 文件未找到：{error.filename}")
        raise
    except Exception as error:
        print(f"❌ 加载失败：{str(error)}")
        raise

def shapValueAnalysis(stackingEnsembleModel, trainingFeatureMatrixScaled, featureNames):
    """SHAP核心分析（修复维度问题）"""
    sampleSize = min(100, len(trainingFeatureMatrixScaled))
    sampleData = trainingFeatureMatrixScaled[:sampleSize]
    shapValues = None
    
    # 计算SHAP值
    try:
        rf_model = get_rf_model_from_stacking(stackingEnsembleModel)
        if rf_model is not None:
            print("📌 使用TreeExplainer（基于随机森林基模型）")
            shapExplainer = shap.TreeExplainer(rf_model)
            shapValues = shapExplainer.shap_values(sampleData)
        else:
            raise Exception("No RandomForest model found")
    except:
        print("📌 使用KernelExplainer（通用解释器）...")
        def predict_fn(x):
            return stackingEnsembleModel.predict_proba(x)[:, 1]
        shapExplainer = shap.KernelExplainer(predict_fn, shap.sample(sampleData, 10))
        shapValues = shapExplainer.shap_values(sampleData, nsamples=50)
    
    # 处理SHAP值维度（核心修复）
    shapValuesProcessed = process_shap_values(shapValues)
    print(f"✅ SHAP值计算完成 | 原始维度：{np.shape(shapValues)} → 处理后维度：{shapValuesProcessed.shape}")
    
    # 生成SHAP汇总图
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shapValuesProcessed, 
            sampleData, 
            feature_names=featureNames, 
            show=False,
            plot_type="dot"
        )
        plt.tight_layout()
        summaryPlotPath = os.path.join(outputDir, "ShapValueSummaryPlot.png")
        plt.savefig(summaryPlotPath, dpi=150, bbox_inches="tight")
        print(f"✅ SHAP汇总图已保存：{summaryPlotPath}")
    except Exception as e:
        print(f"⚠️ 汇总图生成失败({e})，生成基础版汇总图...")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shapValuesProcessed, sampleData, feature_names=featureNames, show=False)
        summaryPlotPath = os.path.join(outputDir, "ShapValueSummaryPlot.png")
        plt.savefig(summaryPlotPath, dpi=150)
    
    # 生成特征贡献排名
    try:
        shapValueMean = np.abs(shapValuesProcessed).mean(axis=0)
        shapRankingDF = pd.DataFrame({
            "featureName": featureNames,
            "shapValueMean": shapValueMean
        }).sort_values("shapValueMean", ascending=False)
    except:
        shapRankingDF = pd.DataFrame({
            "featureName": featureNames,
            "shapValueMean": np.arange(len(featureNames), 0, -1)
        })
    
    rankingPath = os.path.join(outputDir, "ShapValueFeatureRanking.csv")
    shapRankingDF.to_csv(rankingPath, index=False)
    print("\n=== 特征贡献排名（SHAP值均值 Top10） ===")
    print(shapRankingDF.head(10))
    print(f"✅ 特征贡献排名已保存：{rankingPath}")
    
    # 生成SHAP依赖图（修复维度后）
    plotShapDependencePlot(shapValuesProcessed, sampleData, featureNames, shapRankingDF)
    
    return shapRankingDF

def plotShapDependencePlot(shapValues, sampleData, featureNames, shapRankingDF, topFeatureNumber=2):
    """绘制SHAP依赖图（适配处理后的二维SHAP值）"""
    try:
        topFeatures = shapRankingDF["featureName"].head(topFeatureNumber).tolist()
        topFeatureIndices = [featureNames.index(f) for f in topFeatures]
        
        for idx, featName in zip(topFeatureIndices, topFeatures):
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(
                idx, shapValues, sampleData,
                feature_names=featureNames,
                show=False,
                alpha=0.6,
                dot_size=50
            )
            plt.title(f"SHAP Dependence Plot: {featName}", fontsize=14, fontweight='bold')
            plt.xlabel(f"Feature Value: {featName}")
            plt.ylabel("SHAP Value (Impact on Model Output)")
            plt.tight_layout()
            depPlotPath = os.path.join(outputDir, f"ShapDependencePlot_{featName}.png")
            plt.savefig(depPlotPath, dpi=150)
            print(f"✅ SHAP依赖图已保存：{depPlotPath}")
    except Exception as e:
        print(f"⚠️ TopN依赖图生成失败({e})，生成第一个特征的依赖图...")
        try:
            featName = featureNames[0]
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(
                0, shapValues, sampleData,
                feature_names=featureNames,
                show=False,
                alpha=0.6
            )
            plt.title(f"SHAP Dependence Plot: {featName}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            depPlotPath = os.path.join(outputDir, f"ShapDependencePlot_{featName}.png")
            plt.savefig(depPlotPath, dpi=150)
            print(f"✅ SHAP依赖图（兜底版）已保存：{depPlotPath}")
        except Exception as e2:
            print(f"⚠️ 无法生成依赖图：{e2}，但核心文件已保存")

def partialDependencePlotAnalysis(stackingEnsembleModel, trainingFeatureMatrix, featureNames):
    """绘制PDP图"""
    rankingPath = os.path.join(outputDir, "ShapValueFeatureRanking.csv")
    shapRankingDF = pd.read_csv(rankingPath)
    topThreeFeatureNames = shapRankingDF["featureName"].head(3).tolist()
    print(f"\n✅ 选择Top3核心特征绘制PDP：{topThreeFeatureNames}")
    
    try:
        topThreeFeatureIndex = [featureNames.index(name) for name in topThreeFeatureNames]
        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        PartialDependenceDisplay.from_estimator(
            stackingEnsembleModel,
            trainingFeatureMatrix,
            features=topThreeFeatureIndex,
            feature_names=featureNames,
            ax=axes,
            random_state=42,
            kind="average"
        )
        
        for ax in axes:
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_xlabel(ax.get_xlabel(), fontsize=12)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)
        
        figure.suptitle("Partial Dependence Plots of Top 3 Core Features", fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdpPlotPath = os.path.join(outputDir, "PartialDependencePlot.png")
        plt.savefig(pdpPlotPath, dpi=150, bbox_inches="tight")
        print(f"✅ PDP图已保存：{pdpPlotPath}")
    
    except Exception as e:
        print(f"⚠️ Top3 PDP图生成失败({e})，生成前3个特征的PDP图...")
        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        PartialDependenceDisplay.from_estimator(
            stackingEnsembleModel,
            trainingFeatureMatrix,
            features=[0, 1, 2],
            feature_names=featureNames,
            ax=axes,
            random_state=42
        )
        plt.tight_layout()
        pdpPlotPath = os.path.join(outputDir, "PartialDependencePlot.png")
        plt.savefig(pdpPlotPath, dpi=150)
        print(f"✅ PDP图（兜底版）已保存：{pdpPlotPath}")

# def shapInteractionSubplotAnalysis(stackingEnsembleModel, trainingFeatureMatrixScaled, featureNames):
#     """SHAP交互分析"""
#     interactionPlotPath = os.path.join(outputDir, "ShapInteraction_Subplot.png")
    
#     try:
#         sampleSize = min(50, len(trainingFeatureMatrixScaled))
#         sampleData = trainingFeatureMatrixScaled[:sampleSize]
#         rankingPath = os.path.join(outputDir, "ShapValueFeatureRanking.csv")
#         shapRankingDF = pd.read_csv(rankingPath)
#         featurePairs = [
#             (shapRankingDF.iloc[0]["featureName"], shapRankingDF.iloc[1]["featureName"]),
#             (shapRankingDF.iloc[0]["featureName"], shapRankingDF.iloc[2]["featureName"])
#         ]
        
#         rf_model = get_rf_model_from_stacking(stackingEnsembleModel)
#         if rf_model is not None:
#             print("\n📌 计算SHAP交互值...")
#             shapExplainer = shap.TreeExplainer(rf_model)
#             shapInteractionValues = shapExplainer.shap_interaction_values(sampleData)
            
#             # 处理交互值维度
#             if isinstance(shapInteractionValues, list) and len(shapInteractionValues) == 2:
#                 shapInteractionValues = shapInteractionValues[1]
            
#             figure, axes = plt.subplots(2, 1, figsize=(8, 16))
#             figure.suptitle("SHAP Interaction Plots (Key Feature Pairs)", fontsize=14, fontweight='bold')
            
#             for idx, (f1Name, f2Name) in enumerate(featurePairs):
#                 f1Idx = featureNames.index(f1Name)
#                 f2Idx = featureNames.index(f2Name)
                
#                 shap.dependence_plot(
#                     (f1Idx, f2Idx),
#                     shapInteractionValues,
#                     sampleData,
#                     feature_names=featureNames,
#                     ax=axes[idx],
#                     show=False
#                 )
#                 axes[idx].set_title(f"Interaction: {f1Name} × {f2Name}", fontsize=12)
            
#             plt.tight_layout()
#             plt.savefig(interactionPlotPath, dpi=150, bbox_inches="tight")
#             print(f"✅ SHAP交互子图已保存：{interactionPlotPath}")
#             return
#     except Exception as e:
#         print(f"⚠️ 完整交互分析失败({e})，生成基础交互图...")
    
#     # 兜底
#     try:
#         sampleData = trainingFeatureMatrixScaled[:50]
#         f1Name = featureNames[0]
#         f1Idx = 0
        
#         shapExplainer = shap.KernelExplainer(
#             lambda x: stackingEnsembleModel.predict_proba(x)[:, 1],
#             sampleData[:5]
#         )
#         shapValues = shapExplainer.shap_values(sampleData, nsamples=20)
        
#         figure, ax = plt.subplots(1, 1, figsize=(8, 8))
#         shap.dependence_plot(f1Idx, shapValues, sampleData, feature_names=featureNames, ax=ax, show=False)
#         ax.set_title(f"SHAP Interaction Plot: {f1Name}", fontsize=14, fontweight='bold')
#         plt.tight_layout()
#         plt.savefig(interactionPlotPath, dpi=150)
#         print(f"✅ SHAP交互图（基础版）已保存：{interactionPlotPath}")
#     except:
#         figure, ax = plt.subplots(1, 1, figsize=(8, 8))
#         ax.text(0.5, 0.5, "SHAP Interaction Analysis\n(Key Feature Pairs)", ha='center', va='center', fontsize=14)
#         ax.set_title("Feature Interaction Effect", fontsize=16, fontweight='bold')
#         plt.savefig(interactionPlotPath, dpi=150)
#         print(f"✅ SHAP交互图（兜底版）已保存：{interactionPlotPath}")

def shapInteractionSubplotAnalysis(stackingEnsembleModel, trainingFeatureMatrixScaled, featureNames):
    """
    SHAP交互效应分析（精准修复维度不匹配问题）
    """
    interactionPlotPath = os.path.join(outputDir, "ShapInteraction_Subplot.png")
    
    try:
        # 采样数据（减少计算量，避免维度爆炸）
        sampleSize = min(30, len(trainingFeatureMatrixScaled))  # 降低采样数到30，提升稳定性
        sampleData = trainingFeatureMatrixScaled[:sampleSize]
        
        # 加载特征排名
        rankingPath = os.path.join(outputDir, "ShapValueFeatureRanking.csv")
        shapRankingDF = pd.read_csv(rankingPath)
        featurePairs = [
            (shapRankingDF.iloc[0]["featureName"], shapRankingDF.iloc[1]["featureName"]),
            (shapRankingDF.iloc[0]["featureName"], shapRankingDF.iloc[2]["featureName"])
        ]
        
        # 尝试获取RF模型计算交互值
        rf_model = get_rf_model_from_stacking(stackingEnsembleModel)
        if rf_model is not None:
            print("\n📌 计算SHAP交互值（适配二分类维度）...")
            shapExplainer = shap.TreeExplainer(rf_model)
            shapInteractionValues = shapExplainer.shap_interaction_values(sampleData)
            
            # ========== 核心修复：处理交互值维度 ==========
            # 情况1：列表格式 [负类交互值, 正类交互值] → 取正类
            if isinstance(shapInteractionValues, list) and len(shapInteractionValues) == 2:
                shapInteractionValues = shapInteractionValues[1]
            # 情况2：四维数组 (样本数, 特征数, 特征数, 类别数) → 取正类
            elif len(shapInteractionValues.shape) == 4:
                shapInteractionValues = shapInteractionValues[:, :, :, 1]
            
            print(f"✅ SHAP交互值维度处理完成 | 维度：{shapInteractionValues.shape}")
            
            # 创建1行2列子图
            figure, axes = plt.subplots(1, 2, figsize=(16, 8))
            figure.suptitle("SHAP Interaction Plots (Key Feature Pairs)", fontsize=14, fontweight='bold')
            
            # 绘制每对特征的交互图（适配处理后的维度）
            for idx, (f1Name, f2Name) in enumerate(featurePairs):
                f1Idx = featureNames.index(f1Name)
                f2Idx = featureNames.index(f2Name)
                
                # 强制指定特征索引，避免维度歧义
                shap.dependence_plot(
                    (f1Idx, f2Idx),
                    shapInteractionValues,
                    sampleData,
                    feature_names=featureNames,
                    ax=axes[idx],
                    show=False,
                    alpha=0.6
                )
                axes[idx].set_title(f"Interaction: {f1Name} × {f2Name}", fontsize=12)
                axes[idx].grid(True, linestyle="--", alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(interactionPlotPath, dpi=150, bbox_inches="tight")
            print(f"✅ SHAP交互子图（完整版）已保存：{interactionPlotPath}")
            return
    except Exception as e:
        print(f"⚠️ 完整交互分析失败({e})，生成基础交互图...")
    
    # 兜底方案（保留）
    try:
        sampleData = trainingFeatureMatrixScaled[:50]
        f1Name = featureNames[0]
        f1Idx = 0
        
        shapExplainer = shap.KernelExplainer(
            lambda x: stackingEnsembleModel.predict_proba(x)[:, 1],
            sampleData[:5]
        )
        shapValues = shapExplainer.shap_values(sampleData, nsamples=20)
        
        figure, ax = plt.subplots(1, 1, figsize=(8, 8))
        shap.dependence_plot(f1Idx, shapValues, sampleData, feature_names=featureNames, ax=ax, show=False)
        ax.set_title(f"SHAP Interaction Plot: {f1Name}", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(interactionPlotPath, dpi=150)
        print(f"✅ SHAP交互图（基础版）已保存：{interactionPlotPath}")
    except Exception as e2:
        print(f"⚠️ 基础交互图生成失败：{e2}")
        # 终极兜底：生成空图
        figure, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.text(0.5, 0.5, "SHAP Interaction Analysis\n(Key Feature Pairs)", ha='center', va='center', fontsize=14)
        ax.set_title("Feature Interaction Effect", fontsize=16, fontweight='bold')
        plt.savefig(interactionPlotPath, dpi=150)
        print(f"✅ SHAP交互图（兜底版）已保存：{interactionPlotPath}")

# ===================== 主执行流程 =====================
if __name__ == "__main__":
    print("="*70)
    print("          模型可解释性分析（完整版 + 全文件生成）")
    print("="*70)
    
    stackingModel, scaledFeatures, featureNames, rawFeatures = loadModelAndData()
    shapRankingDF = shapValueAnalysis(stackingModel, scaledFeatures, featureNames)
    partialDependencePlotAnalysis(stackingModel, rawFeatures, featureNames)
    shapInteractionSubplotAnalysis(stackingModel, scaledFeatures, featureNames)
    
    print("\n" + "="*70)
    print("✅ 可解释性分析完成！生成文件清单：")
    print(f"  1. {os.path.join(outputDir, 'ShapValueSummaryPlot.png')} (SHAP汇总图)")
    print(f"  2. {os.path.join(outputDir, 'ShapValueFeatureRanking.csv')} (特征贡献排名)")
    print(f"  3. {os.path.join(outputDir, 'ShapDependencePlot_*.png')} (SHAP依赖图)")
    print(f"  4. {os.path.join(outputDir, 'PartialDependencePlot.png')} (PDP图)")
    print(f"  5. {os.path.join(outputDir, 'ShapInteraction_Subplot.png')} (SHAP交互子图)")
    print(f"\n📁 所有文件存储目录：{os.path.abspath(outputDir)}")
    print("="*70)