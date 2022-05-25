# 以下量作为常量，可在模拟之前改变，不影响算法的可行性

N_days = 20          # 交易日数量
N_stocks = 1000      # 股票数量
N_factors = 10       # 因子数量
N_industries = 20    # 行业数量

Seed_r = 12345       # 生成股票收益率时的随机数种子
Seed_f = 34567       # 生成因子值时的随机数种子
Seed_i = 56789       # 生成行业分类时的随机数种子
Seed_w = 78901       # 生成日投资组合时的随机数种子

L_days = len(str(N_days))
L_stocks = len(str(N_stocks))
L_factors = len(str(N_factors))
L_industries = len(str(N_industries))