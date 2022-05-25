# barra-model

A framework to do factors return decomposition

数据组织方式说明

1.Test_Data/
Under directory "T/"
(1)daily_return.csv : 股票在T期的日收益率
(2)daily_weight.csv : 在T期时，为T+1期的投资构造的日投资组合权重
(3)factors.csv : T期期初的因子暴露值，也是T-1期期末的因子暴露度
(4)industries.csv : T期期初的行业分类，也是T-1期期末的行业分类


2.Dump_Data/
Under directory "T/"
(1)style_factor_return.csv : 每日的风格因子收益率
(2)industry_factor_return.csv : 每日的行业收益率
(3)residual_return.csv : 每只股票的残差收益率
(4)return_decomposition.csv : T期实际收益分解的结果，第1期未给出投资权重，不计算

详细计算方法及实验报告请参看“Barra模型模拟计算报告.pdf”文件
