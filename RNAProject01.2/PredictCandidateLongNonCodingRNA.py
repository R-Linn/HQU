import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
# 注意：确保这个模块能正常导入
from ExtractSequenceFeatures import extractSequenceFeaturesFromFastaRecord, calculateLowComplexityRegionRatio
warnings.filterwarnings('ignore')

# 绘图配置（提前设置字体，避免重复指定）
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 提前定义图例字体属性
legend_font = {'family': 'Arial', 'size': 12}

# 全局配置
ModelFilePath = "StackingEnsembleModel.pkl"
ScalerFilePath = "FeatureStandardScaler.pkl"
SelectedFeaturePath = "SelectedFeatureNames.pkl"
ConfidenceThreshold = 0.8  # 高置信度阈值（可调整）

class CandidateFastaRecord:
    """候选lncRNA Fasta记录类"""
    def __init__(self, identifier, sequence):
        self.identifier = identifier
        self.sequence = sequence

def LoadCandidateFasta(fastaFilePath):
    """加载候选lncRNA Fasta文件"""
    candidateRecords = []
    with open(fastaFilePath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        currentId = ""
        currentSeq = ""
        for line in lines:
            line = line.strip()
            if line.startswith(">"):
                if currentId and currentSeq:
                    candidateRecords.append(CandidateFastaRecord(currentId, currentSeq))
                currentId = line[1:]
                currentSeq = ""
            else:
                currentSeq += line.upper()
        # 处理最后一条
        if currentId and currentSeq:
            candidateRecords.append(CandidateFastaRecord(currentId, currentSeq))
    return candidateRecords

def ExtractCandidateFeatures(candidateRecords):
    """提取候选lncRNA特征（返回numpy数组+标识符列表）"""
    # 提取特征
    feature_data = []
    identifiers = []  # 单独存储标识符
    # 先提取第一条记录，查看完整的特征数量和结构
    sample_feat = None
    for idx, record in enumerate(candidateRecords):
        feat = extractSequenceFeaturesFromFastaRecord(record)
        identifiers.append(record.identifier)
        
        # 第一条记录做详细调试
        if idx == 0:
            sample_feat = feat
            print(f"\n🔍 样本特征详细信息：")
            print(f"   - extractSequenceFeaturesFromFastaRecord返回总长度: {len(feat)}")
            print(f"   - 第一个元素（ID）: {feat[0]}")
            print(f"   - 数值特征数量: {len(feat)-1}")
        
        # 只保留数值特征（去掉第一个ID元素），转换为float
        try:
            numeric_feat = [float(x) for x in feat[1:]]
        except ValueError:
            # 处理非数值特征（替换为0）
            numeric_feat = []
            for x in feat[1:]:
                try:
                    numeric_feat.append(float(x))
                except:
                    numeric_feat.append(0.0)
        feature_data.append(numeric_feat)
    
    # 转换为numpy数组（无列名，纯数值）
    feature_array = np.array(feature_data, dtype=np.float32)
    print(f"\n🔍 全量特征数组维度：{feature_array.shape}（样本数 × 86个特征）")
    
    # 加载30个核心特征名，获取其在86个特征中的索引
    selected_feat = joblib.load(SelectedFeaturePath)
    print(f"\n🔍 模型所需核心特征列表（30个）: {selected_feat[:5]}...（显示前5个）")
    
    # 生成86个全量特征名（和你数据的列名完全一致）
    base_features = [
        'adenineRatio', 'thymineRatio', 'cytosineRatio', 'guanineRatio', 
        'guanineCytosineContent', 'lowComplexityRegionRatio'
    ]
    dinucleotides = ['AA','AT','AC','AG','TA','TT','TC','TG','CA','CT','CC','CG','GA','GT','GC','GG']
    dinucleotide_features = [f'dinucleotide{dn}' for dn in dinucleotides]
    trinucleotides = [a+b+c for a in ['A','T','C','G'] for b in ['A','T','C','G'] for c in ['A','T','C','G']]
    trinucleotide_features = [f'trinucleotide{tn}' for tn in trinucleotides]
    all_86_features = base_features + dinucleotide_features + trinucleotide_features
    
    # 验证86个特征名的完整性
    assert len(all_86_features) == 86, f"全量特征名数量错误，应为86个，实际{len(all_86_features)}个"
    
    # 获取30个核心特征在86维数组中的索引
    selected_indices = []
    missing_feat = []
    for feat_name in selected_feat:
        if feat_name in all_86_features:
            selected_indices.append(all_86_features.index(feat_name))
        else:
            missing_feat.append(feat_name)
    
    if missing_feat:
        raise ValueError(f"❌ 核心特征在全量特征中缺失：{missing_feat}")
    print(f"\n🔍 30个核心特征在86维数组中的索引：{selected_indices[:5]}...（显示前5个）")
    
    return identifiers, feature_array, selected_indices

def PredictCandidateLncRNA(feature_array, selected_indices, identifiers):
    """预测候选lncRNA（先标准化86维，再筛选30维）"""
    # 加载模型和标准化器
    try:
        stackingModel = joblib.load(ModelFilePath)
        scaler = joblib.load(ScalerFilePath)
    except FileNotFoundError as e:
        print(f"❌ 模型/标准化器文件未找到: {e}")
        raise
    
    # 步骤1：处理空值/无穷值（86维全量特征）
    feature_array = np.nan_to_num(feature_array, nan=np.nanmean(feature_array))
    feature_array = np.clip(feature_array, -1e6, 1e6)  # 限制极值
    
    # 步骤2：对86个全量特征做标准化（和训练时一致）
    print("\n🔍 开始对86个全量特征做标准化...")
    feature_array_scaled = scaler.transform(feature_array)
    print(f"✅ 标准化完成，标准化后数组维度：{feature_array_scaled.shape}")
    
    # 步骤3：筛选30个核心特征（标准化后筛选）
    core_feature_scaled = feature_array_scaled[:, selected_indices]
    print(f"\n🔍 筛选后核心特征维度：{core_feature_scaled.shape}（样本数 × 30个特征）")
    
    # 步骤4：预测（纯数值数组输入，无特征名问题）
    print("\n🚀 开始预测...（数据量较大，可能需要几分钟，请耐心等待）")
    predictions = stackingModel.predict(core_feature_scaled)
    probabilities = stackingModel.predict_proba(core_feature_scaled)[:, 1]
    print(f"✅ 预测完成！共预测 {len(predictions)} 条lncRNA")
    
    # 转换为DataFrame方便后续处理
    result_df = pd.DataFrame({
        "LongNonCodingRnaIdentifier": identifiers,
        "PredictionLabel": predictions.astype(int),
        "PredictionProbability": probabilities.round(6)
    })
    
    # 筛选高置信度候选
    highConfidenceDF = result_df[result_df["PredictionProbability"] >= ConfidenceThreshold].copy()
    
    return result_df, highConfidenceDF

def PlotPredProbDistribution(result_df):
    """绘制预测概率分布直方图（修复字体参数错误）"""
    plt.figure(figsize=(10, 6))
    # 优化绘图性能（数据量太大，采样绘制）
    sample_size = min(10000, len(result_df))
    sample_df = result_df.sample(sample_size, random_state=42)
    
    sns.histplot(sample_df["PredictionProbability"], bins=20, kde=True, color="#8da0cb")
    plt.axvline(x=ConfidenceThreshold, color="red", linestyle="--", label=f"Confidence Threshold ({ConfidenceThreshold})")
    plt.xlabel("Prediction Probability of Phase Separation", fontfamily='Arial', fontsize=12)
    plt.ylabel("Number of lncRNA (Sampled)", fontfamily='Arial', fontsize=12)
    plt.title("Distribution of Candidate lncRNA Prediction Probability", fontfamily='Arial', fontsize=14, fontweight='bold')
    # 修复：改用prop参数设置图例字体，移除fontfamily
    plt.legend(prop=legend_font)
    plt.tight_layout()
    plt.savefig("PredictionProbabilityDistribution.png", dpi=150)
    print("✅ 预测概率分布图已保存：PredictionProbabilityDistribution.png")

def main():
    # 1. 加载候选Fasta文件
    candidateFastaPath = "candidateLncRNA.fa"
    try:
        candidateRecords = LoadCandidateFasta(candidateFastaPath)
        print(f"✅ 加载候选lncRNA数量：{len(candidateRecords)}")
    except FileNotFoundError:
        print(f"❌ 候选Fasta文件未找到：{candidateFastaPath}")
        return
    except Exception as e:
        print(f"❌ 加载Fasta文件出错：{e}")
        return
    
    # 2. 提取特征（返回numpy数组，无列名）
    try:
        identifiers, feature_array, selected_indices = ExtractCandidateFeatures(candidateRecords)
        print(f"✅ 特征提取完成，全量特征维度：{feature_array.shape}")
    except Exception as e:
        print(f"❌ 特征提取出错：{e}")
        import traceback
        traceback.print_exc()  # 打印详细错误栈
        return
    
    # 3. 预测（先标准化86维，再筛选30维）
    try:
        result_df, highConfidenceDF = PredictCandidateLncRNA(feature_array, selected_indices, identifiers)
    except Exception as e:
        print(f"❌ 预测过程出错：{e}")
        import traceback
        traceback.print_exc()  # 打印详细错误栈
        return
    
    # 4. 统计结果
    print("\n=== 候选lncRNA预测结果统计 ===")
    total = len(result_df)
    positive = len(result_df[result_df['PredictionLabel']==1])
    high_conf = len(highConfidenceDF)
    print(f"总预测数：{total:,}")
    print(f"预测为相分离lncRNA数：{positive:,} ({positive/total:.2%})")
    print(f"高置信度（≥{ConfidenceThreshold}）数：{high_conf:,} ({high_conf/total:.2%})")
    if high_conf > 0:
        print(f"高置信度概率均值：{highConfidenceDF['PredictionProbability'].mean():.4f}")
        print(f"高置信度概率范围：{highConfidenceDF['PredictionProbability'].min():.4f} ~ {highConfidenceDF['PredictionProbability'].max():.4f}")
    else:
        print(f"高置信度概率均值：0.0000")
        print(f"高置信度概率范围：0.0000 ~ 0.0000")
    
    # 5. 绘制概率分布图
    PlotPredProbDistribution(result_df)
    
    # 6. 保存结果
    try:
        # 优化保存性能（分块保存）
        result_df.to_csv("CandidateLncRNAPredictionResult.csv", index=False, chunksize=10000)
        highConfidenceDF.sort_values("PredictionProbability", ascending=False).to_csv("HighConfidenceCandidateDetail.csv", index=False)
        
        print("\n✅ 预测完成，保存文件：")
        print("   - CandidateLncRNAPredictionResult.csv（完整预测结果）")
        print("   - HighConfidenceCandidateDetail.csv（高置信度结果）")
        print("   - PredictionProbabilityDistribution.png（概率分布图）")
    except Exception as e:
        print(f"❌ 保存文件出错：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()