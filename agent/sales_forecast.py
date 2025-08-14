import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from datetime import datetime
import os
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

class ProphetSalesForecast:
    """基于Prophet的销量预测系统（专为少量数据优化）"""
    
    def __init__(self):
        """初始化系统"""
        self.time_series = None
        self.asin_time_series = None
        self.prophet_model = None
        self.prophet_forecast = None
        self.report = {
            'data_exploration': {},
            'prophet': {},
            'best_model': {'name': 'Prophet', 'reason': '专为少量数据设计'}
        }
        
    def load_data(self, file_path, date_column, value_column, date_format=None):
        """
        加载销售数据
        
        参数:
            file_path: 文件路径
            date_column: 日期列名称
            value_column: 销量列名称
            date_format: 日期格式（可选）
        """
        try:
            # 读取数据
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("不支持的文件格式，请使用CSV或Excel文件")
                
            # 转换日期列
            if date_format:
                self.data[date_column] = pd.to_datetime(self.data[date_column], format=date_format)
            else:
                self.data[date_column] = pd.to_datetime(self.data[date_column])
                
            # 按日期排序
            self.data.sort_values(by=date_column, inplace=True)
            
            # 创建时间序列
            self.time_series = self.data.set_index(date_column)[value_column]
            
            print(f"数据加载成功，共 {len(self.time_series)} 个数据点")
            print(f"日期范围: {self.time_series.index.min()} 至 {self.time_series.index.max()}")
            
            return self.time_series
            
        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise
    
    def load_data_from_wide_auto(self, file_path):
        """
        自动加载宽表数据并转换为长表
        
        参数:
            file_path: 文件路径
        """
        try:
            # 读取宽表数据
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(file_path, sheet_name=1)  # 第二个工作表
            else:
                raise ValueError("不支持的文件格式，请使用CSV或Excel文件")
            
            # 自动检测日期列
            date_columns = data.columns[3:].tolist()
            value_column = '销量'
            
            print(f"自动识别日期列: {', '.join(date_columns[:5])}... (共{len(date_columns)}列)")
            
            # 宽表转长表
            long_data = data.melt(
                id_vars=['国家', 'ASIN', 'ParentASIN'],
                value_vars=date_columns,
                var_name='日期',
                value_name=value_column
            )
            
            # 转换日期列
            long_data['日期'] = pd.to_datetime(long_data['日期'], errors='coerce')
            
            # 清理无效数据
            long_data.dropna(subset=['日期', value_column], inplace=True)
            long_data.sort_values(by='日期', inplace=True)
            
            # 创建整体时间序列
            self.time_series = long_data.groupby('日期')[value_column].sum()
            
            # 按 ASIN 分组创建时间序列
            self.asin_time_series = {}
            for asin, group in long_data.groupby('ASIN'):
                ts = group.set_index('日期')[value_column].sort_index()
                if len(ts) >= 6:  # 至少需要6个数据点
                    self.asin_time_series[asin] = ts
            
            print(f"宽表转换成功，共 {len(self.time_series)} 个时间点，{len(self.asin_time_series)} 个有效ASIN")
            print(f"日期范围: {self.time_series.index.min()} 至 {self.time_series.index.max()}")
            
            return self.time_series, self.asin_time_series
            
        except Exception as e:
            print(f"宽表数据加载错误: {str(e)}")
            raise
    
    def quick_data_exploration(self, plot=True, asin=None):
        """
        快速数据探索与分析
        
        参数:
            plot: 是否绘制数据图表
            asin: 指定ASIN进行分析（可选）
        """
        if asin and self.asin_time_series:
            time_series = self.asin_time_series[asin]
            print(f"\n正在进行ASIN {asin} 的数据探索...")
        elif self.time_series is None:
            raise ValueError("请先加载数据")
        else:
            time_series = self.time_series
            print("\n正在进行快速数据探索...")
            
        # 基本统计信息
        stats = time_series.describe()
        print("\n数据基本统计信息:")
        print(stats)
        
        # 缺失值检查
        missing_values = time_series.isnull().sum()
        print(f"\n缺失值数量: {missing_values}")
        
        # 保存数据探索结果
        self.report['data_exploration'] = {
            'stats': stats.to_dict(),
            'missing_values': int(missing_values),
            'data_points': len(time_series)
        }
        
        # 绘制数据图表
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(time_series)
            plt.title(f'ASIN {asin} The sales trend of the product' if asin else 'The sales trend of the product')
            plt.xlabel('Date')
            plt.ylabel('sales volume')
            plt.grid(True)
            plt.tight_layout()
            
            filename = f'data_trend{"_" + asin if asin else ""}.png'
            plt.savefig(filename)
            print(f"数据趋势图已保存为 '{filename}'")
            plt.close()
        
        return self.report['data_exploration']
    
    def preprocess_data(self, transform_method='log', asin=None):
        """
        数据预处理（转换）
        
        参数:
            transform_method: 转换方法，可选 'log' 或 'sqrt'
            asin: 指定ASIN进行预处理（可选）
        """
        if asin and self.asin_time_series:
            time_series = self.asin_time_series[asin]
            print(f"\n正在进行ASIN {asin} 的数据预处理...")
        elif self.time_series is None:
            raise ValueError("请先加载数据")
        else:
            time_series = self.time_series
            print("\n正在进行数据预处理...")
            
        # 数据转换（Prophet对转换不敏感，但可以尝试）
        if transform_method == 'log':
            # 对数转换（添加小常数避免log(0)）
            transformed_data = np.log1p(time_series)
            print(f"使用对数转换处理数据（log(1+x)）")
        elif transform_method == 'sqrt':
            # 平方根转换
            transformed_data = np.sqrt(time_series)
            print(f"使用平方根转换处理数据")
        else:
            # 不进行转换
            transformed_data = time_series
            print(f"不进行数据转换")
        
        return transformed_data
    
    def run_prophet_model(self, forecast_steps=12, asin=None, growth='linear', seasonality_mode='additive'):
        """
        运行 Prophet 模型进行预测（专为少量数据优化）
        
        参数:
            forecast_steps: 预测未来的步数
            asin: 指定ASIN进行预测（可选）
            growth: 增长模型 ('linear', 'logistic', 'flat')
            seasonality_mode: 季节性模式 ('additive', 'multiplicative')
        """
        if asin and self.asin_time_series:
            if asin not in self.asin_time_series:
                raise ValueError(f"ASIN {asin} 不存在或数据点不足")
            time_series = self.asin_time_series[asin]
            print(f"\n正在为ASIN {asin} 运行Prophet模型...")
        elif self.time_series is None:
            raise ValueError("请先加载数据")
        else:
            time_series = self.time_series
            
        try:
            print("准备 Prophet 模型数据...")
            
            # 准备 Prophet 格式的数据
            df_prophet = pd.DataFrame({
                'ds': time_series.index,
                'y': time_series.values
            })
            
            # 创建并训练模型（专为少量数据优化）
            print("训练 Prophet 模型中...")
            self.prophet_model = Prophet(
                growth=growth,
                seasonality_mode=seasonality_mode,
                yearly_seasonality='auto',
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,  # 减少突变点影响
                seasonality_prior_scale=10.0,  # 增加季节性强度
                holidays_prior_scale=10.0,
                mcmc_samples=0,  # 不使用MCMC（小数据更快）
                interval_width=0.95  # 95%置信区间
            )
            
            # 添加自定义季节性（如果有足够数据）
            if len(time_series) > 12:
                self.prophet_model.add_seasonality(
                    name='monthly',
                    period=30.5,
                    fourier_order=3  # 减少参数数量
                )
            
            self.prophet_model.fit(df_prophet)
            
            # 创建预测数据框
            freq = pd.infer_freq(time_series.index) or 'MS'
            future = self.prophet_model.make_future_dataframe(
                periods=forecast_steps, 
                freq=freq,
                include_history=True
            )
            
            # 预测
            self.prophet_forecast = self.prophet_model.predict(future)
            
            # 模型诊断
            print("\nProphet 模型组件分解:")
            fig = self.prophet_model.plot_components(self.prophet_forecast)
            plt.tight_layout()
            plt.savefig(f'prophet_components{"_" + asin if asin else ""}.png')
            plt.close()
            
            # 计算评估指标（使用历史数据）
            historical = self.prophet_forecast[self.prophet_forecast['ds'] <= time_series.index.max()]
            mae = mean_absolute_error(time_series, historical['yhat'])
            rmse = np.sqrt(mean_squared_error(time_series, historical['yhat']))
            
            # 保存模型结果
            model_key = f'prophet_{asin}' if asin else 'prophet'
            self.report[model_key] = {
                'forecast_steps': int(forecast_steps),
                'forecast_mean': self.prophet_forecast['yhat'][-forecast_steps:].tolist(),
                'lower_95': self.prophet_forecast['yhat_lower'][-forecast_steps:].tolist(),
                'upper_95': self.prophet_forecast['yhat_upper'][-forecast_steps:].tolist(),
                'mae': float(mae),
                'rmse': float(rmse),
                'growth_model': growth,
                'seasonality_mode': seasonality_mode
            }
            
            print(f"Prophet 模型预测完成，预测未来 {forecast_steps} 个周期")
            print(f"历史数据MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            
            return self.prophet_forecast
            
        except Exception as e:
            print(f"Prophet 模型错误: {str(e)}")
            raise
    
    def plot_forecast(self, asin=None, title=None, figsize=(12, 6)):
        """
        绘制预测结果
        
        参数:
            asin: 指定ASIN进行绘图（可选）
            title: 图表标题
            figsize: 图表大小
        """
        if asin and self.asin_time_series:
            if asin not in self.asin_time_series:
                raise ValueError(f"ASIN {asin} 不存在或数据点不足")
            time_series = self.asin_time_series[asin]
            print(f"\n正在为ASIN {asin} 绘制预测图表...")
        elif self.time_series is None:
            raise ValueError("请先加载数据")
        else:
            time_series = self.time_series
            
        try:
            plt.figure(figsize=figsize)
            
            # 绘制历史数据
            plt.plot(time_series.index, time_series, 'b-', label='Historical Sales')
            
            # 绘制预测数据
            forecast_dates = self.prophet_forecast['ds']
            forecast_values = self.prophet_forecast['yhat']
            
            # 分离历史和预测
            history_mask = forecast_dates <= time_series.index.max()
            future_mask = forecast_dates > time_series.index.max()
            
            # 绘制历史拟合
            plt.plot(forecast_dates[history_mask], forecast_values[history_mask], 'g--', label='Model Fit')
            
            # 绘制预测
            plt.plot(forecast_dates[future_mask], forecast_values[future_mask], 'r--', label='Forecasted Sales')
            
            # 绘制置信区间
            plt.fill_between(
                forecast_dates, 
                self.prophet_forecast['yhat_lower'], 
                self.prophet_forecast['yhat_upper'], 
                color='gray', alpha=0.2, label='95% Confidence Interval'
            )
            
            # 设置标题和标签
            title = title or f'ASIN {asin} Sales Forecast' if asin else 'Sales Forecast'
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Sales Volume')
            plt.legend()
            plt.grid(True)
            
            # 保存图表
            filename = f'prophet_forecast{"_" + asin if asin else ""}.png'
            plt.savefig(filename)
            print(f"预测图表已保存为 '{filename}'")
            plt.close()
            
        except Exception as e:
            print(f"绘图错误: {str(e)}")
            raise
    
    def generate_report(self, output_file=None, asin=None):
        """
        生成预测报告
        
        参数:
            output_file: 输出文件路径（可选）
            asin: 指定ASIN生成报告（可选）
        """
        if asin and self.asin_time_series:
            if asin not in self.asin_time_series:
                raise ValueError(f"ASIN {asin} 不存在或数据点不足")
            time_series = self.asin_time_series[asin]
            print(f"\n正在为ASIN {asin} 生成预测报告...")
        elif self.time_series is None:
            raise ValueError("请先加载数据")
        else:
            time_series = self.time_series
            
        try:
            # 生成报告
            report = self.report.copy()
            
            # 添加时间戳
            report['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 添加数据摘要
            report['summary'] = {
                'data_points': len(time_series),
                'date_range': {
                    'start': str(time_series.index.min()),
                    'end': str(time_series.index.max())
                },
                'best_model': self.report['best_model']
            }
            
            # 确保所有值为JSON可序列化
            def convert_types(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return str(obj)
                elif isinstance(obj, bool):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                return obj
            
            report = convert_types(report)
            
            # 保存报告
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=4)
                print(f"预测报告已保存为 '{output_file}'")
            
            return report
            
        except Exception as e:
            print(f"生成报告错误: {str(e)}")
            raise
    
    def forecast_for_top_asins(self, n=5, forecast_steps=12):
        """
        为销量最高的前N个ASIN分别进行预测
        
        参数:
            n: 选择前N个ASIN
            forecast_steps: 预测未来的步数
        """
        if self.asin_time_series is None or len(self.asin_time_series) == 0:
            raise ValueError("没有按ASIN分组的时间序列数据")
            
        try:
            # 按总销量排序，选择前N个ASIN
            asin_sales = {asin: ts.sum() for asin, ts in self.asin_time_series.items()}
            top_asins = sorted(asin_sales, key=asin_sales.get, reverse=True)[:n]
            
            print(f"\n选择销量最高的前 {n} 个ASIN进行预测:")
            for i, asin in enumerate(top_asins, 1):
                print(f"{i}. ASIN: {asin}, 总销量: {asin_sales[asin]:.2f}")
            
            # 为每个ASIN进行预测
            for asin in top_asins:
                print(f"\n\n===== 预测 ASIN {asin} =====")
                try:
                    # 数据探索
                    self.quick_data_exploration(asin=asin)
                    # 数据预处理
                    self.preprocess_data(asin=asin)
                    # 运行Prophet模型
                    self.run_prophet_model(forecast_steps, asin=asin)
                    # 绘制预测图表
                    self.plot_forecast(asin=asin)
                    # 生成报告
                    self.generate_report(f'forecast_report_{asin}.json', asin=asin)
                except Exception as e:
                    print(f"ASIN {asin} 预测失败: {str(e)}")
                    continue
            
            print(f"\n已完成前 {n} 个ASIN的销量预测")
            
            return top_asins
            
        except Exception as e:
            print(f"批量预测错误: {str(e)}")
            raise

def main():
    try:
        forecast_system = ProphetSalesForecast()
        
        # 输入文件路径
        file_path = input("请输入数据文件路径: ")
        
        # 自动加载宽表数据
        print("\n正在自动加载宽表数据...")
        forecast_system.load_data_from_wide_auto(file_path)
        
        # 自动数据探索
        print("\n正在进行自动数据探索...")
        forecast_system.quick_data_exploration()
        
        # 预测未来周期数
        forecast_steps = int(input("\n请输入预测未来的周期数: "))
        
        # 选择增长模型
        growth_model = input("选择增长模型 (linear/logistic/flat, 默认为linear): ") or 'linear'
        
        # 运行Prophet模型
        print("\n正在运行Prophet模型...")
        forecast_system.run_prophet_model(forecast_steps, growth=growth_model)
        
        # 绘制预测图表
        print("\n正在绘制预测图表...")
        forecast_system.plot_forecast()
        
        # 生成报告
        print("\n正在生成预测报告...")
        report_file = f'forecast_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        forecast_system.generate_report(report_file)
        
        print(f"\n销量预测完成！报告已保存为 '{report_file}'")
        
        # 询问是否预测Top ASINs
        top_asin = input("\n是否要为销量最高的前N个ASIN进行预测? (y/n): ")
        if top_asin.lower() == 'y':
            n = int(input("请输入要预测的ASIN数量 (N): "))
            forecast_system.forecast_for_top_asins(n, forecast_steps)
        
    except Exception as e:
        print(f"系统运行错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()