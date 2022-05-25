import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


class BarraModel():
    def __init__(self, days, stocks, factors, industries):
        self.N_days = days                   # 交易日数量
        self.N_stocks = stocks               # 股票数量
        self.N_factors = factors             # 风格因子数量
        self.N_industries = industries       # 行业数量
        
        
    def generate_data(self, Seed_r, Seed_f, Seed_i, Seed_w):
        L_days = str(len(str(self.N_days)))                     # 交易日总数的字符串长度
        L_stocks = str(len(str(self.N_stocks)))                 # 股票代码的字符串长度
        L_factors = str(len(str(self.N_factors)))               # 风格因子数量的字符串长度
        L_industries = str(len(str(self.N_industries)))         # 行业数量的字符串长度
        
        os.system('rm -rf Test_Data')                           # 先删除之前的数据
        os.system('mkdir Test_Data')                            # 制作测试数据目录
        for t in range(1, self.N_days + 1):
            s = '{:0>n}'.replace('n', L_days).format(t)
            if s not in os.listdir('Test_Data/'):
                os.system('mkdir Test_Data/' + s)
                
        rng = np.random.RandomState(Seed_r)                     # set the seed to make the example deterministic
        stocks = np.array(['S' + '{:0>n}'.replace('n', L_stocks).format(i) for i in range(1, self.N_stocks + 1)])
        for x in os.listdir('Test_Data/'):
            ret = 200 * (rng.rand(self.N_stocks,) - 0.5)        # Uniform distribution between -100 and 100
            df_ret = pd.DataFrame({'Share_Code': stocks, 'Returns': ret})
            df_ret.to_csv(f'Test_Data/{x}/daily_return.csv', index = 0)
        
        rng = np.random.RandomState(Seed_f)                     # set the seed to make the example deterministic
        zscore = lambda x: (x - np.nanmean(x)) / np.nanstd(x)   # 将数据严格地标准化为 N(0, 1)
        for x in os.listdir('Test_Data/'):
            d = {}
            for factor in range(1, self.N_factors + 1):
                a = rng.randn(self.N_stocks,)                                            # 生成标准正态分布的一组因子
                num = int(a.size * 0.01 * np.random.random())
                indices = rng.choice(np.arange(a.size), replace = False, size = num)     # 随机选择一些索引
                a[indices] = np.nan                                                      # 随机将少量因子值置为NaN 
                d['F' + '{:0>n}'.replace('n', L_factors).format(factor)] = zscore(a)     # 重新标准化
            df_factors = pd.DataFrame(d)
            df_factors[np.abs(df_factors) > 3] = np.sign(df_factors) * 3                 # 处理三倍标准差以外的数据
            df_factors.insert(loc = 0, column = 'Share_Code', value = stocks)            # 添加股票名称列
            df_factors.to_csv(f'Test_Data/{x}/factors.csv', index = 0)                   # 将因子值写入csv文件
        
        rng = np.random.RandomState(Seed_i)                      # set the seed to make the example deterministic
        industries = rng.randint(self.N_industries, size = self.N_stocks)                # 行业分类一维表
        d = {'Share_Code': stocks}
        for industry in range(1, self.N_industries + 1):
            a = np.zeros(self.N_stocks)                                                  # 全是0的一列
            a[(industries == industry - 1).nonzero()[0]] = 1                             # 属于该行业的股票相应值置为1
            d['I' + '{:0>n}'.replace('n', L_industries).format(industry)] = a    
        df_industries = pd.DataFrame(d)
        for x in os.listdir('Test_Data/'):
            df_industries.to_csv(f'Test_Data/{x}/industries.csv', index = 0)
            
        rng = np.random.RandomState(Seed_w)                      # set the seed to make the example deterministic     
        def scale(w):
            w -= np.mean(w)                                      # 平移，使所有权重均值为0
            w *= 2 / sum(abs(w))                                 # rescale，使 long = short = 1
            return w
        for x in os.listdir('Test_Data/'):
            w = scale(rng.rand(self.N_stocks,))                  # Generate U[0, 1], then do scale
            df_weight = pd.DataFrame({'Share_Code': stocks, 'Weight': w})
            df_weight.to_csv(f'Test_Data/{x}/daily_weight.csv', index = 0)
            
    
    def calculate(self, dump_name = 'Dump_Data'):                # Calculate and Dump
        os.system(f'rm -rf {dump_name}')
        os.system(f'mkdir {dump_name}')
        ret_fac = [np.zeros(self.N_factors)]                     # return of style factors 
        ret_ind = [np.zeros(self.N_industries)]                  # return of industry factors
        for day in sorted(os.listdir('Test_Data/')):
            os.system(f'mkdir {dump_name}/{day}')
            df_ret = pd.read_csv(f'Test_Data/{day}/daily_return.csv').set_index('Share_Code')
            df_fac = pd.read_csv(f'Test_Data/{day}/factors.csv').set_index('Share_Code').fillna(0)   # fill NAN
            df_ind = pd.read_csv(f'Test_Data/{day}/industries.csv').set_index('Share_Code')
            df = pd.concat([df_ret, df_fac, df_ind], axis = 1)
            model = sm.OLS(df.iloc[:,0], df.iloc[:,1:])          # 普通最小二乘模型
            results = model.fit()                                # 回归
            rp = results.params                                  # 每日的风格因子收益 F_ret[j] 和 行业收益率 I_ret[j]
            df_fr = pd.DataFrame(rp, columns=['Factor_Ret'])     # factor_ret
            df_fr.index.names = ['Factor_Name']
            df_fr.iloc[:self.N_factors,].to_csv(f'{dump_name}/{day}/style_factor_return.csv')
            df_fr.iloc[self.N_factors:,].to_csv(f'{dump_name}/{day}/industry_factor_return.csv')
            residual = results.resid                             # 股票日残差收益率 S_ret[i]
            df_resid = pd.DataFrame(residual, index = df_ret.index, columns = ['Residual_Returns'])
            df_resid.to_csv(f'{dump_name}/{day}/residual_return.csv')
            
            
    def decomposition(self, dump_name = 'Dump_Data'):            # 对投资组合做收益分解
        date_list = sorted(os.listdir('Test_Data'))
        for t in range(1, len(date_list)):
            today = date_list[t]
            yesterday = date_list[t - 1]
            M_sfExpo = np.array(pd.read_csv(f'Test_Data/{today}/factors.csv').iloc[:,1:].fillna(0)) # 风格因子暴露
            M_ifExpo = np.array(pd.read_csv(f'Test_Data/{today}/industries.csv').iloc[:,1:])        # 行业暴露
            df = pd.read_csv(f'{dump_name}/{today}/style_factor_return.csv')                        # 风格因子收益
            sf_name = df['Factor_Name']
            V_sfRet = np.array(df['Factor_Ret'])   
            df = pd.read_csv(f'{dump_name}/{today}/industry_factor_return.csv')                     # 行业因子收益
            if_name = df['Factor_Name']
            V_ifRet = np.array(df['Factor_Ret'])  
            V_resi = np.array(pd.read_csv(f'{dump_name}/{today}/residual_return.csv')['Residual_Returns']) # 残差收益
            W = np.array(pd.read_csv(f'Test_Data/{yesterday}/daily_weight.csv')['Weight'])          # 投资权重
            rsf = np.matmul(W.T, M_sfExpo) * V_sfRet             # return of style factor
            rif = np.matmul(W.T, M_ifExpo) * V_ifRet             # return of industry factor
            u = np.dot(W.T, V_resi)                              # redidual term
            df_d = pd.DataFrame(np.append(rsf, rif), index = pd.concat([sf_name, if_name]), columns=['Return'])
            df_d.loc['Residual'] = u
            df_d.to_csv(f'{dump_name}/{today}/return_decomposition.csv')
        
    
    def show_factor_ret(self, tp, split = 1, path_name = 'Dump_Data'):    
         # tp: 'style' or 'industry' or 'porfolio'; split: number of group; path_name: path of dumped data
        if tp == 'portfolio':
            ret_fac = [np.zeros(self.N_factors + self.N_industries)]
            for day in sorted(os.listdir(path_name))[1:]:
                df_fr = pd.read_csv(f'{path_name}/{day}/return_decomposition.csv').set_index('Factor_Name')
                ret_fac.append(df_fr['Return'][:-1])
        else:
            N = self.N_factors if tp == 'style' else self.N_industries
            ret_fac = [np.zeros(N)]                          # return of factors 
            for day in sorted(os.listdir(path_name)):
                df_fr = pd.read_csv(f'{path_name}/{day}/{tp}_factor_return.csv').set_index('Factor_Name')
                ret_fac.append(df_fr['Factor_Ret'])
        data = np.array(ret_fac).T                           # factors
        group_size = (len(data) + 1) // split                # 每一组的因子个数
        s_title = ('{:^15}'.format('Factor') + '{:^15}'.format('Days') + '{:^15}'.format('Mean') 
                   + '{:^15}'.format('Mean / Std') + '{:^15}'.format('AutoTsCorr') + '\n')
        s_info = ''
        for i in range(len(data)):
            plt.plot(data[i].cumsum(), label = ret_fac[1].index[i])
            s_info += (f'{ret_fac[1].index[i] :^15}{len(data[i]) - 1 :^15}{np.mean(data[i]) :^15.5f}' 
                       + f'{np.mean(data[i]) / np.std(data[i]) :^15.5f}' 
                       + f'{np.corrcoef(range(len(data[i])), data[i].cumsum())[0][1] :^15.5f}\n')
            if (i + 1) % group_size == 0 or i + 1 == len(data):
                plt.xlabel('Days')
                plt.ylabel('Cum ret')
                if tp == 'portfolio':
                    plt.title(f'Cumulative decomposed portfolio return')
                else:
                    plt.title(f'Cumulative return of {tp} factors')
                plt.legend(bbox_to_anchor=(1, 1))
                plt.show()                                   # 画图 
                print(s_title + s_info)                      # 打印信息
                s_info = ''
                
                
    def run_model(self, r, f, i, w):                         # 使用默认参数运行该模型
        self.generate_data(r, f, i, w)
        self.calculate()
        self.decomposition()
        self.show_factor_ret('style')
        self.show_factor_ret('industry', 2)
        self.show_factor_ret('portfolio', 3)
        
        
if __name__ == '__main__':
    barra1 = BarraModel(20, 1000, 10, 20)
    barra1.run_model(12345, 34567, 56789, 78901)
    
    