#!/usr/bin/env python3
"""
清理TBD文件夹 - 删除已经整理到10_AtomicKnowledge的笔记
"""

import os
from pathlib import Path

# TBD路径
TBD_PATH = Path("/Users/xinby/Library/CloudStorage/OneDrive-Personal/Statistics/ByAca/TBD")
ATOMIC_PATH = Path("/Users/xinby/Library/CloudStorage/OneDrive-Personal/Statistics/ByAca/10_AtomicKnowledge")

# 已整理的文件映射 (TBD原文件名 -> 10_AtomicKnowledge路径)
organized_files = {
    # 统计推断
    "Fisher Information.md": "math.ST_Statistics/02_Inference/Fisher_Information.md",
    "UMVUE.md": "math.ST_Statistics/02_Inference/UMVUE.md",
    "Sufficiency.md": "math.ST_Statistics/02_Inference/Sufficiency.md",
    "Cramér-Rao Lower Bound.md": "math.ST_Statistics/02_Inference/Cramer-Rao_Lower_Bound.md",
    "Rao-Blackwell Procedure.md": "math.ST_Statistics/02_Inference/Rao-Blackwell_Procedure.md",
    "Lehmann-Scheffé Theorem.md": "math.ST_Statistics/02_Inference/Lehmann-Scheffe_Theorem.md",
    "Neyman Factorization Theorem.md": "math.ST_Statistics/02_Inference/Neyman_Factorization_Theorem.md",
    "Delta Method.md": "math.ST_Statistics/02_Inference/Delta_Method.md",
    "Law of Large Numbers.md": "math.ST_Statistics/02_Inference/Law_of_Large_Numbers.md",
    "Fisher's Exact Test.md": "math.ST_Statistics/02_Inference/Fishers_Exact_Test.md",
    "Kruskal-Wallis Test.md": "math.ST_Statistics/02_Inference/Kruskal-Wallis_Test.md",
    "Mann-Whitney U Test.md": "math.ST_Statistics/02_Inference/Mann-Whitney_U_Test.md",
    "Nonparametric Statistics Tests.md": "math.ST_Statistics/02_Inference/Nonparametric_Statistics_Tests.md",
    "Sign Test.md": "math.ST_Statistics/02_Inference/Sign_Test.md",
    "Wilcoxon Rank Sum Test.md": "math.ST_Statistics/02_Inference/Wilcoxon_Rank_Sum_Test.md",
    "Wilcoxon Signed Rank Test.md": "math.ST_Statistics/02_Inference/Wilcoxon_Signed_Rank_Test.md",
    
    # 概率论
    "Exponential Family.md": "math.PR_Probability/Exponential_Family.md",
    "Central Limit Theorem.md": "math.PR_Probability/Central_Limit_Theorem.md",
    "Convergens in Statistics.md": "math.PR_Probability/Convergence_in_Statistics.md",
    "Box-Muller Transformation - Generating Normal Random Variable by Polar Method.md": "math.PR_Probability/Random_Variable_Generation/Box-Muller_Transformation_Generating_Normal_Random_Variable_by_Polar_Method.md",
    "Chebyshev Inequality.md": "math.PR_Probability/Chebyshev_Inequality.md",
    "Generating Binomial Random Variable from U(0,1).md": "math.PR_Probability/Random_Variable_Generation/Generating_Binomial_Random_Variable_from_U_0_1.md",
    "Generating Continuous Random Variables.md": "math.PR_Probability/Random_Variable_Generation/Generating_Continuous_Random_Variables.md",
    "Generating Discrete Random Variables.md": "math.PR_Probability/Random_Variable_Generation/Generating_Discrete_Random_Variables.md",
    "Generating Exponential Random Variable from U(0,1).md": "math.PR_Probability/Random_Variable_Generation/Generating_Exponential_Random_Variable_from_U_0_1.md",
    "Generating Geometric Random Variable from U(0,1).md": "math.PR_Probability/Random_Variable_Generation/Generating_Geometric_Random_Variable_from_U_0_1.md",
    "Generating Poisson Random Variable from U(0,1).md": "math.PR_Probability/Random_Variable_Generation/Generating_Poisson_Random_Variable_from_U_0_1.md",
    
    # GLM
    "GLM.md": "stat.ME_Methodology/Generalized_Linear_Models.md",
    
    # NLP/LLM
    "BERT.md": "cs.CL_NLP_LLM/BERT.md",
    "Deep Reasoning for LLMs.md": "cs.CL_NLP_LLM/Deep_Reasoning_for_LLMs.md",
    "LanguageModels.md": "cs.CL_NLP_LLM/LanguageModels.md",
    "Mitigating LLM Hallucinations via Conformal Abstention.md": "cs.CL_NLP_LLM/Mitigating_LLM_Hallucinations_via_Conformal_Abstention.md",
    "Pretrain and Alignment for LLMs.md": "cs.CL_NLP_LLM/Pretrain_and_Alignment_for_LLMs.md",
    "WordEmbeddings.md": "cs.CL_NLP_LLM/WordEmbeddings.md",
    "WordEmbeddings2.md": "cs.CL_NLP_LLM/WordEmbeddings2.md",
    
    # 深度学习
    "Transformer and its Friends.md": "ml.TH_MachineLearning/03_DeepLearning/Transformer_and_Variants.md",
    "Attention,Transformer.md": "ml.TH_MachineLearning/03_DeepLearning/Attention_Transformer.md",
    "CNN.md": "ml.TH_MachineLearning/03_DeepLearning/CNN.md",
    "CNN Architecture.md": "ml.TH_MachineLearning/03_DeepLearning/CNN_Architecture.md",
    "Computer Vision.md": "ml.TH_MachineLearning/03_DeepLearning/Computer_Vision.md",
    "FCNN.md": "ml.TH_MachineLearning/03_DeepLearning/FCNN.md",
    "Modern CNNs.md": "ml.TH_MachineLearning/03_DeepLearning/Modern_CNNs.md",
    "Modern RNNs.md": "ml.TH_MachineLearning/03_DeepLearning/Modern_RNNs.md",
    "RNN Architecture.md": "ml.TH_MachineLearning/03_DeepLearning/RNN_Architecture.md",
    
    # 实分析
    "Menu for Real Analysis.md": "math.RA_RealAnalysis/Menu_for_Real_Analysis.md",
    "REALANA10_Open Sets and Closed Sets.md": "math.RA_RealAnalysis/REALANA10_Open_Sets_and_Closed_Sets.md",
    "REALANA11_Measurement Theory Introduction.md": "math.RA_RealAnalysis/REALANA11_Measurement_Theory_Introduction.md",
    "REALANA11_Measurement Theory Introduction (2).md": "math.RA_RealAnalysis/REALANA11_Measurement_Theory_Introduction_(2).md",
    "REALANA12_Outer Measure.md": "math.RA_RealAnalysis/REALANA12_Outer_Measure.md",
    "REALANA1_Introduction to Real Analysis.md": "math.RA_RealAnalysis/REALANA1_Introduction_to_Real_Analysis.md",
    "REALANA2_Representations of Sets and its Operations.md": "math.RA_RealAnalysis/REALANA2_Representations_of_Sets_and_its_Operations.md",
    "REALANA3_Describe a Statement by Sets-ByLeon.md": "math.RA_RealAnalysis/REALANA3_Describe_a_Statement_by_Sets-ByLeon.md",
    "REALANA3_Describe a Statement by Sets.md": "math.RA_RealAnalysis/REALANA3_Describe_a_Statement_by_Sets.md",
    "REALANA4_Upper Limit and Lower Limit of  a Sequence.md": "math.RA_RealAnalysis/REALANA4_Upper_Limit_and_Lower_Limit_of__a_Sequence.md",
    "REALANA5_Equivalent and Cardinality.md": "math.RA_RealAnalysis/REALANA5_Equivalent_and_Cardinality.md",
    "REALANA6_Countable Sets.md": "math.RA_RealAnalysis/REALANA6_Countable_Sets.md",
    "REALANA7_Uncountable Sets.md": "math.RA_RealAnalysis/REALANA7_Uncountable_Sets.md",
    "REALANA8_Metric Space & n_d Euclidean Space.md": "math.RA_RealAnalysis/REALANA8_Metric_Space_&_n_d_Euclidean_Space.md",
    "REALANA9_Cluster Points, Interior Points, Boundary Points.md": "math.RA_RealAnalysis/REALANA9_Cluster_Points_Interior_Points_Boundary_Points.md",
    "Uniformly Continuous.md": "math.RA_RealAnalysis/Uniformly_Continuous.md",
    
    # 时间序列
    "Exponential Smoothing.md": "math.ST_Statistics/03_TimeSeries/Exponential_Smoothing.md",
    "Introduction to Forecasting.md": "math.ST_Statistics/03_TimeSeries/Introduction_to_Forecasting.md",
    "Stationary Stochastic Process.md": "math.ST_Statistics/03_TimeSeries/Stationary_Stochastic_Process.md",
    "Statistical Models For Time Series.md": "math.ST_Statistics/03_TimeSeries/Statistical_Models_For_Time_Series.md",
    "Auto-Correlation Function (Stochastic Process).md": "math.ST_Statistics/03_TimeSeries/Auto-Correlation_Function_(Stochastic_Process).md",
    "Linear Correlation (Stochastic Process).md": "math.ST_Statistics/03_TimeSeries/Linear_Correlation_(Stochastic_Process).md",
    "A Very Simple Note on Spectrual Analysis.md": "math.ST_Statistics/03_TimeSeries/A_Very_Simple_Note_on_Spectrual_Analysis.md",
    "School of Frequency.md": "math.ST_Statistics/03_TimeSeries/School_of_Frequency.md",
    "Autoregressive Model.md": "math.ST_Statistics/03_TimeSeries/Autoregressive_Model.md",
    "Autoregressive Model (II).md": "math.ST_Statistics/03_TimeSeries/Autoregressive_Model__II_.md",
    
    # 机器学习方法
    "AdaBoost.md": "ml.TH_MachineLearning/02_Methods/AdaBoost.md",
    "Bagging and Pasting.md": "ml.TH_MachineLearning/02_Methods/Bagging_and_Pasting.md",
    "Boosting.md": "ml.TH_MachineLearning/02_Methods/Boosting.md",
    "Ensemble Learning.md": "ml.TH_MachineLearning/02_Methods/Ensemble_Learning.md",
    "Gradient Boosting.md": "ml.TH_MachineLearning/02_Methods/Gradient_Boosting.md",
    "Introduction to Meta Learning.md": "ml.TH_MachineLearning/02_Methods/Introduction_to_Meta_Learning.md",
    "Introduction to Reinforcement Learning.md": "ml.TH_MachineLearning/02_Methods/Introduction_to_Reinforcement_Learning.md",
    "LifeLong Learning.md": "ml.TH_MachineLearning/02_Methods/LifeLong_Learning.md",
    "Optimization Algorithms.md": "ml.TH_MachineLearning/02_Methods/Optimization_Algorithms.md",
    "Regularization.md": "ml.TH_MachineLearning/02_Methods/Regularization.md",
    "Support Vector Machine (Ver1_张颢).md": "ml.TH_MachineLearning/02_Methods/Support_Vector_Machine__Ver1_张颢_.md",
    "Transfer Learning.md": "ml.TH_MachineLearning/02_Methods/Transfer_Learning.md",
    "Automatically Determining Hyperparameters.md": "ml.TH_MachineLearning/02_Methods/Automatically_Determining_Hyperparameters.md",
    "Gauss Mixture Model.md": "ml.TH_MachineLearning/01_Theory/Gauss_Mixture_Model.md",
    "Markov Decision Process.md": "ml.TH_MachineLearning/01_Theory/Markov_Decision_Process.md",
    
    # 线性代数
    "Column Space of Matrix.md": "math.LA_LinearAlgebra/Column_Space_of_Matrix.md",
    "Norm.md": "math.LA_LinearAlgebra/Norm.md",
    
    # 其他
    "Complete Statistics.md": "math.ST_Statistics/02_Inference/Complete_Statistics.md",
}

def main():
    print("🧹 开始清理TBD文件夹...\n")
    
    deleted_count = 0
    not_found_count = 0
    verified_count = 0
    
    for tbd_file, atomic_file in organized_files.items():
        tbd_path = TBD_PATH / tbd_file
        atomic_path = ATOMIC_PATH / atomic_file
        
        # 检查文件是否存在
        if not tbd_path.exists():
            print(f"  ⚠️  TBD中未找到: {tbd_file}")
            not_found_count += 1
            continue
        
        # 验证目标文件已存在
        if atomic_path.exists():
            # 删除TBD中的文件
            try:
                tbd_path.unlink()
                print(f"  ✅ 已删除: {tbd_file}")
                deleted_count += 1
            except Exception as e:
                print(f"  ❌ 删除失败 {tbd_file}: {e}")
        else:
            print(f"  ⚠️  目标文件不存在，保留源文件: {tbd_file}")
            verified_count += 1
    
    # 删除空文件和临时文件
    empty_files = ["Untitled.md", "Untitled 1.md", "Support Vector Machine (Ver2 CS229).md"]
    for empty_file in empty_files:
        empty_path = TBD_PATH / empty_file
        if empty_path.exists():
            try:
                empty_path.unlink()
                print(f"  🗑️  已删除空文件: {empty_file}")
                deleted_count += 1
            except Exception as e:
                print(f"  ❌ 删除失败 {empty_file}: {e}")
    
    print(f"\n📊 清理完成!")
    print(f"  ✅ 成功删除: {deleted_count} 个文件")
    print(f"  ⚠️  未找到: {not_found_count} 个文件")
    print(f"  🔍 需验证: {verified_count} 个文件")
    
    # 统计剩余文件
    remaining = list(TBD_PATH.glob("*.md"))
    print(f"\n📁 TBD中剩余 {len(remaining)} 个markdown文件")

if __name__ == "__main__":
    main()



