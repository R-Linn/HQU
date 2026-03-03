import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 全局配置 - 设置随机种子保证结果可复现
random.seed(42)
np.random.seed(42)

class FastaSequenceRecord:
    """Fasta序列记录类（大驼峰命名，单词全拼）"""
    def __init__(self, identifier, sequence, sampleLabel):
        self.identifier = identifier  # 序列唯一标识符（全拼）
        self.sequence = sequence  # 核苷酸序列（全拼）
        self.sampleLabel = sampleLabel                # 样本标签（1=正样本，0=负样本）

def LoadFastaFile(fastaFilePath, sampleLabel):
    """
    加载Fasta文件并转换为FastaSequenceRecord列表（小驼峰命名，单词全拼）
    参数:
        fastaFilePath: Fasta文件路径
        sampleLabel: 该文件中序列的标签（1表示正样本，0表示负样本）
    返回:
        包含FastaSequenceRecord对象的列表
    """
    sequenceRecordList = []
    with open(fastaFilePath, 'r', encoding='utf-8') as file:
        fileLines = file.readlines()
        currentSequenceIdentifier = ""
        currentSequence = ""
        
        for line in fileLines:
            line = line.strip()
            if line.startswith(">"):
                # 如果已有未保存的序列，先保存
                if currentSequenceIdentifier and currentSequence:
                    sequenceRecord = FastaSequenceRecord(
                        identifier=currentSequenceIdentifier,
                        sequence=currentSequence,
                        sampleLabel=sampleLabel
                    )
                    sequenceRecordList.append(sequenceRecord)
                
                currentSequenceIdentifier = line[1:]
                currentSequence = ""
            else:
                # 拼接序列并转换为大写
                currentSequence += line.upper()
        
        # 处理最后一条序列
        if currentSequenceIdentifier and currentSequence:
            sequenceRecord = FastaSequenceRecord(
                identifier=currentSequenceIdentifier,
                sequence=currentSequence,
                sampleLabel=sampleLabel
            )
            sequenceRecordList.append(sequenceRecord)
    
    return sequenceRecordList

def CalculateSequenceLengthStatistics(sequenceRecordList):
    """
    统计序列长度特征（小驼峰命名，单词全拼）
    参数:
        sequenceRecordList: FastaSequenceRecord对象列表
    返回:
        均值、最小值、最大值、标准差
    """
    sequenceLengthList = [len(record.sequence) for record in sequenceRecordList]
    meanLength = np.mean(sequenceLengthList)
    minLength = np.min(sequenceLengthList)
    maxLength = np.max(sequenceLengthList)
    stdLength = np.std(sequenceLengthList)
    
    return meanLength, minLength, maxLength, stdLength

def SplitSequenceRecordDataset(sequenceRecordList, testSizeFirst=0.2, testSizeSecond=0.5, randomState=42):
    """
    拆分序列记录数据集（小驼峰命名，单词全拼）
    逻辑：先拆分正负样本，再合并，保证每类样本都分配到训练/验证/测试集
    参数:
        sequenceRecordList: 待拆分的序列记录列表（混合正负样本）
        testSizeFirst: 第一次拆分的测试集比例
        testSizeSecond: 第二次拆分的测试集比例
        randomState: 随机种子
    返回:
        训练集、验证集、测试集
    """
    # 分离正负样本
    positiveSequenceRecords = [record for record in sequenceRecordList if record.sampleLabel == 1]
    negativeSequenceRecords = [record for record in sequenceRecordList if record.sampleLabel == 0]
    
    # 拆分正样本
    positiveTrainingSet, positiveTemporarySet = train_test_split(
        positiveSequenceRecords,
        test_size=testSizeFirst,
        random_state=randomState
    )
    positiveValidationSet, positiveTestSet = train_test_split(
        positiveTemporarySet,
        test_size=testSizeSecond,
        random_state=randomState
    )
    
    # 拆分负样本
    negativeTrainingSet, negativeTemporarySet = train_test_split(
        negativeSequenceRecords,
        test_size=testSizeFirst,
        random_state=randomState
    )
    negativeValidationSet, negativeTestSet = train_test_split(
        negativeTemporarySet,
        test_size=testSizeSecond,
        random_state=randomState
    )
    
    # 合并训练/验证/测试集
    trainingSet = positiveTrainingSet + negativeTrainingSet
    validationSet = positiveValidationSet + negativeValidationSet
    testSet = positiveTestSet + negativeTestSet
    
    # 打乱顺序（避免同类样本扎堆）
    random.shuffle(trainingSet)
    random.shuffle(validationSet)
    random.shuffle(testSet)
    
    return trainingSet, validationSet, testSet

def SplitPositiveAndNegativeSamples(sequenceRecordList):
    """
    拆分正负样本（小驼峰命名，单词全拼）
    参数:
        sequenceRecordList: 序列记录列表
    返回:
        正样本列表、负样本列表
    """
    positiveSamples = [record for record in sequenceRecordList if record.sampleLabel == 1]
    negativeSamples = [record for record in sequenceRecordList if record.sampleLabel == 0]
    return positiveSamples, negativeSamples

def Main():
    """主函数（大驼峰命名）"""
    # 1. 定义文件路径（全拼+清晰命名）
    positiveSampleFastaFilePath = "gencode.v49.lncRNA_transcripts_copy.fa"  # 正样本lncRNA文件路径
    negativeSampleFastaFilePath = "positiveRawData.fna"                      # 负样本对照序列文件路径
    
    # 2. 加载正负样本Fasta文件
    positiveSampleRecordList = LoadFastaFile(positiveSampleFastaFilePath, sampleLabel=1)
    negativeSampleRecordList = LoadFastaFile(negativeSampleFastaFilePath, sampleLabel=0)
    
    print(f"数据加载完成 - 正样本数量：{len(positiveSampleRecordList)} | 负样本数量：{len(negativeSampleRecordList)}")
    
    # 3. 混合正负样本
    allSequenceRecordList = positiveSampleRecordList + negativeSampleRecordList
    
    # 4. 拆分数据集
    trainingDataset, validationDataset, testDataset = SplitSequenceRecordDataset(allSequenceRecordList)
    
    # 5. 拆分训练/验证/测试集的正负样本
    positiveTrainingSamples, negativeTrainingSamples = SplitPositiveAndNegativeSamples(trainingDataset)
    positiveValidationSamples, negativeValidationSamples = SplitPositiveAndNegativeSamples(validationDataset)
    positiveTestSamples, negativeTestSamples = SplitPositiveAndNegativeSamples(testDataset)
    
    # 打印拆分后各类样本数量（调试用）
    print(f"\n=== 数据集拆分后样本数量统计 ===")
    print(f"训练集：正样本{len(positiveTrainingSamples)} | 负样本{len(negativeTrainingSamples)}")
    print(f"验证集：正样本{len(positiveValidationSamples)} | 负样本{len(negativeValidationSamples)}")
    print(f"测试集：正样本{len(positiveTestSamples)} | 负样本{len(negativeTestSamples)}")
    
    # 6. 统计序列长度特征
    positiveTrainingMean, positiveTrainingMin, positiveTrainingMax, positiveTrainingStd = CalculateSequenceLengthStatistics(positiveTrainingSamples)
    negativeTrainingMean, negativeTrainingMin, negativeTrainingMax, negativeTrainingStd = CalculateSequenceLengthStatistics(negativeTrainingSamples)
    positiveValidationMean, positiveValidationMin, positiveValidationMax, positiveValidationStd = CalculateSequenceLengthStatistics(positiveValidationSamples)
    negativeValidationMean, negativeValidationMin, negativeValidationMax, negativeValidationStd = CalculateSequenceLengthStatistics(negativeValidationSamples)
    positiveTestMean, positiveTestMin, positiveTestMax, positiveTestStd = CalculateSequenceLengthStatistics(positiveTestSamples)
    negativeTestMean, negativeTestMin, negativeTestMax, negativeTestStd = CalculateSequenceLengthStatistics(negativeTestSamples)
    
    # 输出长度统计结果
    print("\n=== 正/负样本序列长度统计（核苷酸）===")
    print(f"训练集正样本：均值{positiveTrainingMean:.2f} ± {positiveTrainingStd:.2f}，范围[{positiveTrainingMin}, {positiveTrainingMax}]")
    print(f"训练集负样本：均值{negativeTrainingMean:.2f} ± {negativeTrainingStd:.2f}，范围[{negativeTrainingMin}, {negativeTrainingMax}]")
    print(f"验证集正样本：均值{positiveValidationMean:.2f} ± {positiveValidationStd:.2f}，范围[{positiveValidationMin}, {positiveValidationMax}]")
    print(f"验证集负样本：均值{negativeValidationMean:.2f} ± {negativeValidationStd:.2f}，范围[{negativeValidationMin}, {negativeValidationMax}]")
    print(f"测试集正样本：均值{positiveTestMean:.2f} ± {positiveTestStd:.2f}，范围[{positiveTestMin}, {positiveTestMax}]")
    print(f"测试集负样本：均值{negativeTestMean:.2f} ± {negativeTestStd:.2f}，范围[{negativeTestMin}, {negativeTestMax}]")
    
    # 7. 保存长度统计结果
    sequenceLengthStatisticsDataFrame = pd.DataFrame({
        "数据集类型": ["训练集正样本", "训练集负样本", "验证集正样本", "验证集负样本", "测试集正样本", "测试集负样本"],
        "样本数量": [len(positiveTrainingSamples), len(negativeTrainingSamples), 
                  len(positiveValidationSamples), len(negativeValidationSamples),
                  len(positiveTestSamples), len(negativeTestSamples)],
        "平均长度": [positiveTrainingMean, negativeTrainingMean, positiveValidationMean, negativeValidationMean, positiveTestMean, negativeTestMean],
        "长度标准差": [positiveTrainingStd, negativeTrainingStd, positiveValidationStd, negativeValidationStd, positiveTestStd, negativeTestStd],
        "最小长度": [positiveTrainingMin, negativeTrainingMin, positiveValidationMin, negativeValidationMin, positiveTestMin, negativeTestMin],
        "最大长度": [positiveTrainingMax, negativeTrainingMax, positiveValidationMax, negativeValidationMax, positiveTestMax, negativeTestMax]
    })
    sequenceLengthStatisticsDataFrame.to_csv("SequenceLengthStatistics.csv", index=False)
    
    # 8. 正/负样本核心特征初步对比
    try:
        from ExtractSequenceFeatures import extractSequenceFeaturesFromFastaRecord
        # 随机抽取50条样本
        positiveSample = random.sample(positiveSampleRecordList, 50) if len(positiveSampleRecordList)>=50 else positiveSampleRecordList
        negativeSample = random.sample(negativeSampleRecordList, 50) if len(negativeSampleRecordList)>=50 else negativeSampleRecordList
        
        # 提取特征
        positiveFeatureList = [extractSequenceFeaturesFromFastaRecord(record) for record in positiveSample]
        negativeFeatureList = [extractSequenceFeaturesFromFastaRecord(record) for record in negativeSample]
        
        # 提取GC含量、低复杂度占比
        positiveGcContentList = [feature[5] for feature in positiveFeatureList]
        negativeGcContentList = [feature[5] for feature in negativeFeatureList]
        positiveLowComplexityRatioList = [feature[6] for feature in positiveFeatureList]
        negativeLowComplexityRatioList = [feature[6] for feature in negativeFeatureList]
        
        # 保存对比结果
        featureComparisonDataFrame = pd.DataFrame({
            "样本类型": ["正样本"]*len(positiveGcContentList) + ["负样本"]*len(negativeGcContentList),
            "GC含量": positiveGcContentList + negativeGcContentList,
            "低复杂度区域占比": positiveLowComplexityRatioList + negativeLowComplexityRatioList
        })
        featureComparisonDataFrame.to_csv("PositiveNegativeFeatureComparison.csv", index=False)
        print("\n正/负样本核心特征对比数据已保存：PositiveNegativeFeatureComparison.csv")
        
    except (ImportError, AttributeError) as error:
        print(f"\n警告：特征对比统计失败 - {error}")
    
    # 9. 保存数据集划分结果（供后续特征提取使用）
    allDatasetDictionary = {
        "trainingPositiveSamples": positiveTrainingSamples, 
        "trainingNegativeSamples": negativeTrainingSamples,
        "validationPositiveSamples": positiveValidationSamples, 
        "validationNegativeSamples": negativeValidationSamples,
        "testPositiveSamples": positiveTestSamples, 
        "testNegativeSamples": negativeTestSamples
    }
    
    # 保存为pickle文件
    import joblib
    joblib.dump(allDatasetDictionary, "DatasetSplitResult.pkl")
    print("\n数据集划分结果已保存：DatasetSplitResult.pkl")

if __name__ == "__main__":
    Main()