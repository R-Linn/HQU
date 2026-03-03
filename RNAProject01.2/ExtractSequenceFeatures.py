import pandas as pd
import numpy as np
import joblib
from Bio.SeqUtils import gc_fraction as GC
from Bio.Seq import Seq
import warnings
warnings.filterwarnings('ignore')

class FastaSequenceRecord:
    """
    与修改后PreOperation.py中完全一致的类定义（驼峰+全拼）
    用于反序列化DatasetSplitResult.pkl文件
    """
    def __init__(self, identifier, sequence, sampleLabel):
        self.identifier = identifier  # RNA标识符
        self.sequence = sequence  # 核苷酸序列
        self.sampleLabel = sampleLabel  # 样本标签（正/负）
        
# 定义核苷酸组合列表
dinucleotideCombinationList = [
    "AA", "AT", "AC", "AG",
    "TA", "TT", "TC", "TG",
    "CA", "CT", "CC", "CG",
    "GA", "GT", "GC", "GG"
]

trinucleotideCombinationList = [
    "AAA", "AAT", "AAC", "AAG",
    "ATA", "ATT", "ATC", "ATG",
    "ACA", "ACT", "ACC", "ACG",
    "AGA", "AGT", "AGC", "AGG",
    "TAA", "TAT", "TAC", "TAG",
    "TTA", "TTT", "TTC", "TTG",
    "TCA", "TCT", "TCC", "TCG",
    "TGA", "TGT", "TGC", "TGG",
    "CAA", "CAT", "CAC", "CAG",
    "CTA", "CTT", "CTC", "CTG",
    "CCA", "CCT", "CCC", "CCG",
    "CGA", "CGT", "CGC", "CGG",
    "GAA", "GAT", "GAC", "GAG",
    "GTA", "GTT", "GTC", "GTG",
    "GCA", "GCT", "GCC", "GCG",
    "GGA", "GGT", "GGC", "GGG"
]

def calculateLowComplexityRegionRatio(sequenceString):
    """
    计算低复杂度区域占比
    判定规则：单碱基连续出现≥5次的区域视为低复杂度区域
    参数:
        sequenceString: 序列字符串
    返回:
        低复杂度区域长度占总序列长度的比例（保留4位小数）
    """
    sequenceLength = len(sequenceString)
    if sequenceLength == 0:
        return 0.0
    
    lowComplexityLength = 0
    currentBase = sequenceString[0]
    currentLength = 1
    
    for base in sequenceString[1:]:
        if base == currentBase:
            currentLength += 1
        else:
            if currentLength >= 5:
                lowComplexityLength += currentLength
            currentBase = base
            currentLength = 1
    
    # 处理最后一段连续碱基
    if currentLength >= 5:
        lowComplexityLength += currentLength
    
    return round(lowComplexityLength / sequenceLength, 4)

def extractSequenceFeaturesFromFastaRecord(fastaSequenceRecord):
    """
    从FastaSequenceRecord对象提取序列特征（适配新属性名）
    特征包含：单碱基频率、GC含量、低复杂度占比、二核苷酸频率、三核苷酸频率
    参数:
        fastaSequenceRecord: FastaSequenceRecord类实例
    返回:
        包含所有特征的列表
    """
    # 适配新属性名：sequence 替代原 sequence
    sequenceString = fastaSequenceRecord.sequence
    sequenceLength = len(sequenceString)
    
    # 1. 单碱基频率（腺嘌呤A、胸腺嘧啶T、胞嘧啶C、鸟嘌呤G）
    adenineCount = sequenceString.count("A")
    thymineCount = sequenceString.count("T")
    cytosineCount = sequenceString.count("C")
    guanineCount = sequenceString.count("G")
    
    adenineRatio = round(adenineCount / sequenceLength, 4) if sequenceLength > 0 else 0.0
    thymineRatio = round(thymineCount / sequenceLength, 4) if sequenceLength > 0 else 0.0
    cytosineRatio = round(cytosineCount / sequenceLength, 4) if sequenceLength > 0 else 0.0
    guanineRatio = round(guanineCount / sequenceLength, 4) if sequenceLength > 0 else 0.0
    
    # 2. GC含量（鸟嘌呤+胞嘧啶占比）
    guanineCytosineContent = round(GC(sequenceString), 4) if sequenceLength > 0 else 0.0
    
    # 3. 低复杂度区域占比
    lowComplexityRegionRatio = calculateLowComplexityRegionRatio(sequenceString)
    
    # 4. 二核苷酸频率
    dinucleotideFrequencyList = []
    for dinucleotideCombination in dinucleotideCombinationList:
        dinucleotideCount = sequenceString.count(dinucleotideCombination)
        dinucleotideFrequency = round(dinucleotideCount / (sequenceLength - 1) if sequenceLength > 1 else 0.0, 4)
        dinucleotideFrequencyList.append(dinucleotideFrequency)
    
    # 5. 三核苷酸频率（论文要求新增）
    trinucleotideFrequencyList = []
    for trinucleotideCombination in trinucleotideCombinationList:
        trinucleotideCount = sequenceString.count(trinucleotideCombination)
        trinucleotideFrequency = round(trinucleotideCount / (sequenceLength - 2) if sequenceLength > 2 else 0.0, 4)
        trinucleotideFrequencyList.append(trinucleotideFrequency)
    
    # 组合所有特征（序列ID + 基础特征 + 二核苷酸特征 + 三核苷酸特征）
    featureList = [
        fastaSequenceRecord.identifier,
        adenineRatio, thymineRatio, cytosineRatio, guanineRatio,
        guanineCytosineContent,
        lowComplexityRegionRatio
    ] + dinucleotideFrequencyList + trinucleotideFrequencyList
    
    return featureList

def processLongNonCodingRnaDataset(datasetSplitPath):
    """
    处理lncRNA数据集并提取特征，包含特征标准化（适配新键名）
    参数:
        datasetSplitPath: 数据集划分结果文件路径（DatasetSplitResult.pkl）
    返回:
        训练集、验证集、测试集的特征DataFrame
    """
    # 加载划分后的数据集
    allDataSets = joblib.load(datasetSplitPath)
    
    # 定义特征列名
    featureColumnNameList = [
        "longNonCodingRnaIdentifier",
        "adenineRatio", "thymineRatio", "cytosineRatio", "guanineRatio",
        "guanineCytosineContent", "lowComplexityRegionRatio"
    ] + [f"dinucleotide{combination}" for combination in dinucleotideCombinationList] + \
       [f"trinucleotide{combination}" for combination in trinucleotideCombinationList]
    
    # 处理每个子集（适配新键名：trainingPositiveSamples 等）
    processedDataSets = {}
    for dataSetName, records in allDataSets.items():
        if not records:
            continue
        
        # 批量提取特征
        featureData = [extractSequenceFeaturesFromFastaRecord(record) for record in records]
        
        # 转换为DataFrame
        dataFrame = pd.DataFrame(featureData, columns=featureColumnNameList)
        
        # 添加样本标签（适配新键名：Positive=1 正样本，Negative=0 负样本）
        dataFrame["sampleLabel"] = 1 if "Positive" in dataSetName else 0
        processedDataSets[dataSetName] = dataFrame
    
    # 合并正负样本，生成完整的训练/验证/测试集（适配新键名）
    trainingSetDataFrame = pd.concat([processedDataSets["trainingPositiveSamples"], processedDataSets["trainingNegativeSamples"]], ignore_index=True)
    validationSetDataFrame = pd.concat([processedDataSets["validationPositiveSamples"], processedDataSets["validationNegativeSamples"]], ignore_index=True)
    testSetDataFrame = pd.concat([processedDataSets["testPositiveSamples"], processedDataSets["testNegativeSamples"]], ignore_index=True)
    
    # 特征标准化（仅标准化数值特征，保留ID和标签）
    from sklearn.preprocessing import StandardScaler
    featureColumns = [column for column in trainingSetDataFrame.columns if column not in ["longNonCodingRnaIdentifier", "sampleLabel"]]
    
    # 训练标准化器（仅基于训练集，避免数据泄露）
    standardScaler = StandardScaler()
    trainingSetDataFrame[featureColumns] = standardScaler.fit_transform(trainingSetDataFrame[featureColumns])
    
    # 将训练集的标准化规则应用到验证集和测试集
    validationSetDataFrame[featureColumns] = standardScaler.transform(validationSetDataFrame[featureColumns])
    testSetDataFrame[featureColumns] = standardScaler.transform(testSetDataFrame[featureColumns])
    
    # 保存标准化器（供后续模型预测使用）
    joblib.dump(standardScaler, "FeatureStandardScaler.pkl")
    
    # 保存特征数据集
    trainingSetDataFrame.to_csv("TrainingSetWithFeatures.csv", index=False)
    validationSetDataFrame.to_csv("ValidationSetWithFeatures.csv", index=False)
    testSetDataFrame.to_csv("TestSetWithFeatures.csv", index=False)
    
    print(f"特征提取完成：")
    print(f"   训练集：{trainingSetDataFrame.shape} | 验证集：{validationSetDataFrame.shape} | 测试集：{testSetDataFrame.shape}")
    
    # 特征统计（仅输出训练集统计结果）
    print("\n=== 特征提取结果统计 ===")
    print(f"总特征维度：{len(featureColumns)}（单碱基4+GC1+低复杂度1+2-mer16+3-mer64）")
    print(f"训练集特征均值范围：{trainingSetDataFrame[featureColumns].mean().min():.4f} ~ {trainingSetDataFrame[featureColumns].mean().max():.4f}")
    print(f"训练集特征方差范围：{trainingSetDataFrame[featureColumns].var().min():.4f} ~ {trainingSetDataFrame[featureColumns].var().max():.4f}")
    
    # 保存特征统计结果
    featureStatistics = pd.DataFrame({
        "featureName": featureColumns,
        "meanValue": trainingSetDataFrame[featureColumns].mean().values,
        "standardDeviation": trainingSetDataFrame[featureColumns].std().values,
        "minimumValue": trainingSetDataFrame[featureColumns].min().values,
        "maximumValue": trainingSetDataFrame[featureColumns].max().values
    })
    featureStatistics.to_csv("FeatureStatistics.csv", index=False)
    
    return trainingSetDataFrame, validationSetDataFrame, testSetDataFrame

if __name__ == "__main__":
    # 处理数据集（需先运行修改后的PreOperation.py生成DatasetSplitResult.pkl）
    try:
        trainingSet, validationSet, testSet = processLongNonCodingRnaDataset("DatasetSplitResult.pkl")
        print("\n所有特征文件已保存：")
        print("   - TrainingSetWithFeatures.csv")
        print("   - ValidationSetWithFeatures.csv")
        print("   - TestSetWithFeatures.csv")
        print("   - FeatureStandardScaler.pkl")
        print("   - FeatureStatistics.csv")
    except FileNotFoundError as error:
        print(f"文件未找到：{error.filename}，请先运行PreOperation.py生成数据集划分文件")