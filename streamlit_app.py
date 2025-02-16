import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import jieba
from wordcloud import WordCloud
import numpy as np
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from prophet.plot import plot_plotly
import os

# 数据加载和处理
def load_and_process_data(files):
    all_data = []
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if '最近上榜' in line:
                        # 解析每行数据
                        title = line.split('微博热搜榜')[0].strip()
                        date_str = line.split('最近上榜:')[1].split('累计在榜')[0].strip()
                        duration = line.split('累计在榜:')[1].split('最高排名')[0].strip()
                        rank = int(line.split('最高排名:')[1].split()[0])

                        # 转换日期格式
                        date = datetime.strptime(date_str, '%Y/%m/%d %H:%M')

                        # 修改类别提取方式（原代码）
                        # category = file.split('.')[0]

                        # 新代码：从文件路径中提取纯文件名
                        category = os.path.splitext(os.path.basename(file))[0]

                        # 转换持续时间为分钟
                        duration_mins = 0
                        if '小时' in duration:
                            hours = float(duration.split('小时')[0])
                            duration_mins = hours * 60
                            if '分' in duration:
                                mins = float(duration.split('小时')[1].split('分')[0])
                                duration_mins += mins
                        elif '分' in duration:
                            duration_mins = float(duration.split('分')[0])

                        all_data.append({
                            'title': title,
                            'date': date,
                            'duration_mins': duration_mins,
                            'rank': rank,
                            'category': category
                        })
        except Exception as e:
            st.error(f"处理文件 {file} 时出错: {str(e)}")

    return pd.DataFrame(all_data)

# 绘制月度趋势图
def plot_monthly_trends(df):
    # 按月份和类别统计热搜数量
    df['month'] = df['date'].dt.to_period('M')
    monthly_counts = df.groupby(['month', 'category']).size().reset_index(name='count')
    monthly_counts['month'] = monthly_counts['month'].astype(str)

    fig = px.line(monthly_counts,
                  x='month',
                  y='count',
                  color='category',
                  title='各类热搜月度趋势',
                  labels={'month': '月份', 'count': '热搜数量'})

    fig.update_layout(xaxis_tickangle=-45)
    return fig

# 绘制排名分布图
def plot_rank_distribution(df):
    fig = px.box(df,
                 y='category',
                 x='rank',
                 orientation='h',
                 title='各类热搜排名分布',
                 labels={'category': '类别', 'rank': '排名'},
                 color='category')  # 添加颜色映射到类别

    # 反转x轴并设置对数刻度
    fig.update_xaxes(
        autorange="reversed",
        type="log",
        tickvals=[1, 2, 3, 5, 10, 20, 50],
        ticktext=["1", "2", "3", "5", "10", "20", "50"]
    )

    # 调整布局和图例
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        height=500,
        hovermode='x unified',  # 添加统一的悬停模式
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            align="left",
            namelength=-1  # 显示完整名称
        ),
        margin=dict(b=100),  # 增加底部边距
        legend=dict(
            title='类别',
            orientation="h",  # 水平图例
            yanchor="bottom",
            y=1.02,  # 图例位于图表上方
            xanchor="right",
            x=1
        )
    )

    # 修改图例显示方式
    fig.for_each_trace(lambda t: t.update(name=t.name.split('=')[-1]))
    return fig

# 绘制持续时间分析图
def plot_duration_analysis(df):
    fig = px.scatter(df,
                     x='rank',
                     y='duration_mins',
                     color='category',
                     title='热搜排名与持续时间关系',
                     labels={'rank': '排名',
                            'duration_mins': '持续时间(分钟)',
                            'category': '类别'})
    return fig

# 生成词云图
def generate_wordcloud(df):
    text = ' '.join(df['title'])
    words = jieba.cut(text)
    word_space_split = ' '.join(words)

    wordcloud = WordCloud(font_path='simhei.ttf',
                         background_color='white',
                         width=800,
                         height=400).generate(word_space_split)

    return wordcloud

def find_significant_periods(dates, slopes, window_size=5, threshold=1.5, merge_days=15):
    """
    找出显著变化的时间区间

    参数:
    dates: 日期序列
    slopes: 斜率序列
    window_size: 滑动窗口大小
    threshold: 标准差倍数阈值
    merge_days: 合并区间的天数阈值

    返回:
    significant_periods: 显著变化区间的列表 [(start_date, end_date, mean_slope), ...]
    """
    # 计算斜率的标准差
    slope_std = slopes.std()
    threshold_value = slope_std * threshold

    # 初始化变量
    periods = []
    current_period = None

    for i in range(len(slopes)):
        slope = slopes.iloc[i]
        date = dates.iloc[i]

        # 如果斜率超过阈值
        if abs(slope) > threshold_value:
            if current_period is None:
                current_period = {
                    'start': date,
                    'slopes': [slope],
                    'dates': [date]
                }
            else:
                # 如果与上一个日期相隔不超过window_size，则延续当前区间
                if (date - current_period['dates'][-1]).total_seconds() <= window_size * 24 * 3600:
                    current_period['slopes'].append(slope)
                    current_period['dates'].append(date)
                else:
                    # 保存当前区间并开始新区间
                    if len(current_period['slopes']) >= 3:  # 至少要有3个点才算一个有效区间
                        periods.append({
                            'start': current_period['start'],
                            'end': current_period['dates'][-1],
                            'mean_slope': np.mean(current_period['slopes'])
                        })
                    current_period = {
                        'start': date,
                        'slopes': [slope],
                        'dates': [date]
                    }
        else:
            # 如果当前有活跃的区间，检查是否应该结束它
            if current_period is not None:
                if len(current_period['slopes']) >= 3:
                    periods.append({
                        'start': current_period['start'],
                        'end': current_period['dates'][-1],
                        'mean_slope': np.mean(current_period['slopes'])
                    })
                current_period = None

    # 处理最后一个区间
    if current_period is not None and len(current_period['slopes']) >= 3:
        periods.append({
            'start': current_period['start'],
            'end': current_period['dates'][-1],
            'mean_slope': np.mean(current_period['slopes'])
        })

    # 新增：合并相邻区间
    merged_periods = []
    if periods:
        # 按时间排序
        sorted_periods = sorted(periods, key=lambda x: x['start'])

        # 初始化第一个区间
        current = sorted_periods[0]

        for next_period in sorted_periods[1:]:
            time_gap = (next_period['start'] - current['end']).total_seconds() / 3600 / 24

            if time_gap <= merge_days:
                # 合并区间
                current['end'] = next_period['end']
                # 加权平均（按时间长度）
                current_duration = (current['end'] - current['start']).total_seconds()
                next_duration = (next_period['end'] - next_period['start']).total_seconds()
                current['mean_slope'] = (current['mean_slope'] * current_duration +
                                        next_period['mean_slope'] * next_duration) / (current_duration + next_duration)
            else:
                merged_periods.append(current)
                current = next_period
        merged_periods.append(current)

        # 再次按显著性排序
        merged_periods.sort(key=lambda x: abs(x['mean_slope']), reverse=True)

    return merged_periods[:5]  # 仍然返回前5个最显著的

# 绘制热搜持续时间横道图
def plot_timeline_gantt(df, selected_category):
    # 筛选特定类别的数据并按时间排序
    category_df = df[df['category'] == selected_category].copy()
    category_df = category_df.sort_values('date')

    # 修改：根据时间分组来分配y轴位置
    time_window = pd.Timedelta(hours=72)
    current_position = 0
    positions = []
    last_end_time = None

    for _, row in category_df.iterrows():
        current_time = row['date']
        current_duration = pd.Timedelta(minutes=row['duration_mins'])

        if last_end_time is None:
            positions.append(current_position)
        else:
            # 如果当前热搜开始时间在上一个热搜结束时间之后超过时间窗口
            # 则回到较低的位置重新开始
            time_gap = current_time - last_end_time
            if time_gap > time_window:
                current_position = max(0, current_position - 1)
            else:
                # 如果时间重叠，则增加位置
                current_position += 1
            positions.append(current_position)  # 移动到这里

        last_end_time = current_time + current_duration

    category_df['y_position'] = positions

    # 计算移动平均和斜率
    window_size = 3  # 移动平均窗口大小改为15天

    # 将时间戳转换为数值（以天为单位）
    dates_num = (category_df['date'] - category_df['date'].min()).dt.total_seconds() / (24*3600)
    y_positions = category_df['y_position']

    # 计算移动平均
    dates_smooth = pd.Series(dates_num).rolling(window=window_size, center=True, min_periods=1).mean()
    y_positions_smooth = pd.Series(y_positions).rolling(window=window_size, center=True, min_periods=1).mean()

    # 计算斜率
    dx = dates_smooth.diff()
    dy = y_positions_smooth.diff()
    slopes = dy / dx  # 斜率（每天的变化率）
    slopes = slopes.rolling(window=window_size, center=True, min_periods=1).mean()  # 平滑斜率

    # 修改斜率归一化方式，保留正负值
    max_abs_slope = slopes.abs().max()
    normalized_slopes = slopes / max_abs_slope if max_abs_slope > 0 else slopes

    # 将日期数值转回datetime以用于绘图
    smooth_dates = category_df['date'].min() + pd.to_timedelta(dates_smooth, unit='D')

    # 找出显著变化的时间区间
    valid_mask = ~slopes.isna()
    significant_periods = find_significant_periods(
        smooth_dates[valid_mask],
        slopes[valid_mask]
    )

    # 修改：应用对数转换和归一化
    scaled_slopes = slopes[valid_mask] / 100  # 保持原有缩放

    # 计算累积趋势（指数型）
    cumulative_trend = [1.0]
    for s in scaled_slopes[1:]:
        cumulative_trend.append(cumulative_trend[-1] * (1 + s))

    # 应用自然对数转换（避免负值）
    log_trend = np.log(np.clip(cumulative_trend, 1e-6, None))  # 防止取log(0)

    # 双重归一化处理（先log再归一化）
    trend_min = log_trend.min()
    trend_max = log_trend.max()
    normalized_log_trend = (log_trend - trend_min) / (trend_max - trend_min)

    # 创建子图布局（保持2行1列）
    fig = make_subplots(rows=2, cols=1,
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.05,
                        shared_xaxes=True,
                        specs=[[{"secondary_y": False}],
                               [{"secondary_y": True}]])  # 允许第二个子图使用次级Y轴

    # 设置颜色映射
    rank_colors = px.colors.sequential.Viridis
    max_rank = category_df['rank'].max()

    # 添加平滑连接线
    fig.add_trace(
        go.Scatter(
            x=smooth_dates,
            y=y_positions_smooth,
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False
        ),
        row=1, col=1
    )

    # 添加热搜时间块
    for idx, row in category_df.iterrows():
        end_time = row['date'] + pd.Timedelta(minutes=float(row['duration_mins']))
        color_idx = int((row['rank'] / max_rank) * (len(rank_colors) - 1))
        bar_color = rank_colors[color_idx]

        # 添加时间块
        fig.add_trace(
            go.Scatter(
                x=[row['date'], end_time],
                y=[row['y_position'], row['y_position']],
                mode='lines',
                line=dict(color=bar_color, width=10),
                name=row['title'],
                hovertext=f"标题: {row['title']}<br>" \
                         f"开始时间: {row['date'].strftime('%Y/%m/%d %H:%M')}<br>" \
                         f"持续时间: {row['duration_mins']:.0f}分钟<br>" \
                         f"最高排名: {row['rank']}",
                hoverinfo="text",
                showlegend=False
            ),
            row=1, col=1
        )

        # 添加端点标记
        fig.add_trace(
            go.Scatter(
                x=[row['date']],
                y=[row['y_position']],
                mode='markers',
                marker=dict(color=bar_color, size=8, line=dict(color='white', width=1)),
                hoverinfo="skip",
                showlegend=False
            ),
            row=1, col=1
        )

    # 在第二个子图中同时绘制变化率和趋势线
    # 原始变化率曲线（左侧Y轴）
    fig.add_trace(
        go.Scatter(
            x=smooth_dates[valid_mask],
            y=normalized_slopes[valid_mask],
            mode='lines',
            line=dict(color='rgba(100,100,100,0.5)', width=2),
            name='变化率',
            showlegend=False
        ),
        row=2, col=1
    )

    # 对数趋势线（右侧Y轴）
    fig.add_trace(
        go.Scatter(
            x=smooth_dates[valid_mask],
            y=normalized_log_trend,
            mode='lines',
            line=dict(color='blue', width=2, dash='dot'),  # 改为蓝色以示区别
            name='对数趋势',
            yaxis='y3'
        ),
        row=2, col=1,
        secondary_y=True
    )

    # 标记显著区间
    for period in significant_periods:
        # 添加显著区间的高亮背景
        fig.add_vrect(
            x0=period['start'],
            x1=period['end'],
            fillcolor="red",
            opacity=0.2,
            layer="below",
            line_width=0,
            row=2, col=1
        )

        # 添加区间标注
        fig.add_annotation(
            x=period['start'],
            y=1,  # 在图表顶部
            text=f"变化率: {period['mean_slope']:.2f}",
            showarrow=True,
            arrowhead=1,
            row=2, col=1
        )

    # 更新布局
    fig.update_layout(
        title=dict(text=f'{selected_category}相关热搜时间线', font=dict(size=20)),
        height=500,  # 增加总高度以容纳两个图表
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False,
        hovermode='x unified',
        xaxis_range=[category_df['date'].min() - pd.Timedelta(hours=6),
                    category_df['date'].max() + pd.Timedelta(hours=6)],
        xaxis2_range=[category_df['date'].min() - pd.Timedelta(hours=6),
                     category_df['date'].max() + pd.Timedelta(hours=6)],
        yaxis2=dict(
            title='变化率',
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        ),
        yaxis3=dict(
            title='对数趋势 (归一化)',
            overlaying='y2',
            side='right',
            showgrid=False,
            range=[0, 1]
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )

    # 更新主图布局
    fig.update_xaxes(
        title='时间',
        type='date',
        tickformat='%Y/%m/%d %H:%M',
        tickangle=-45,
        gridcolor='lightgray',
        showgrid=True,
        row=1, col=1
    )

    fig.update_yaxes(
        showticklabels=False,
        showgrid=True,
        gridcolor='lightgray',
        zeroline=False,
        row=1, col=1
    )

    # 更新斜率图布局
    fig.update_xaxes(
        title='时间',
        type='date',
        tickformat='%Y/%m/%d %H:%M',
        tickangle=-45,
        gridcolor='lightgray',
        showgrid=True,
        row=2, col=1
    )

    fig.update_yaxes(
        title='变化率',
        gridcolor='lightgray',
        showgrid=True,
        zeroline=True,
        zerolinecolor='black',
        zerolinewidth=2,  # 加粗零线
        row=2, col=1
    )

    return fig

# 在现有函数后添加新函数
def plot_prophet_forecast(df, category):
    """使用Prophet进行时间序列预测"""
    try:
        # 准备数据
        monthly_data = df[df['category'] == category].copy()
        monthly_data['month'] = monthly_data['date'].dt.to_period('M').dt.to_timestamp()
        ts_data = monthly_data.groupby('month').size().reset_index()
        ts_data.columns = ['ds', 'y']

        if len(ts_data) < 3:
            st.warning(f"类别 {category} 数据不足（至少需要3个月数据），无法进行预测")
            return None

        # 训练模型
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
        model.fit(ts_data)

        # 生成预测（未来3个月）
        future = model.make_future_dataframe(periods=3, freq='M')
        forecast = model.predict(future)

        # 绘制图表
        fig = plot_plotly(model, forecast)

        # 修改趋势线绘制部分
        # 使用Prophet的趋势组件
        trend_data = forecast[['ds', 'trend']].merge(ts_data, on='ds', how='left')

        # 添加趋势线（使用Prophet计算出的趋势）
        fig.add_trace(
            go.Scatter(
                x=trend_data['ds'],
                y=trend_data['trend'],
                mode='lines',
                line=dict(color='green', width=3, dash='dot'),
                name='Prophet趋势',
                hovertemplate=
                    "<b>日期</b>: %{x|%Y-%m-%d}<br>" +
                    "<b>趋势值</b>: %{y:.1f}<br>" +
                    "<extra></extra>"
            )
        )

        # 添加历史数据点（保持原有代码）
        fig.add_trace(
            go.Scatter(
                x=ts_data['ds'],
                y=ts_data['y'],
                mode='markers',
                name='实际值',
                marker=dict(color='red', size=8),
                hovertemplate=
                    "<b>日期</b>: %{x|%Y-%m-%d}<br>" +
                    "<b>实际值</b>: %{y}<br>" +
                    "<extra></extra>"
            )
        )

        # 修改后的基准计算逻辑
        # 获取数据时间范围
        min_date = ts_data['ds'].min()
        baseline_end = pd.Timestamp('2020-01-01')

        if min_date < baseline_end:
            # 计算从最早数据到2020年前的数据
            baseline_data = ts_data[(ts_data['ds'] >= min_date) &
                                   (ts_data['ds'] < baseline_end)]
            baseline_value = baseline_data['y'].mean()
            date_range_str = f"{min_date.strftime('%Y-%m')}至{baseline_end.strftime('%Y-%m')}"
            baseline_text = f"历史基准（{date_range_str}）: {baseline_value:.2f}"
        else:
            # 如果所有数据都在2020年后，使用全部数据平均
            baseline_value = ts_data['y'].mean()
            baseline_text = f"历史基准（全部数据）: {baseline_value:.2f}"

        # 合并预测数据与实际数据
        full_data = forecast[['ds', 'yhat']].merge(ts_data, on='ds', how='left')
        full_data['multiple'] = full_data['yhat'] / baseline_value

        # 计算倍数数据（使用实际值）
        full_data['actual_multiple'] = full_data['y'] / baseline_value
        full_data['forecast_multiple'] = full_data['yhat'] / baseline_value

        # 创建右侧Y轴
        fig.update_layout(
            yaxis2=dict(
                title="基准倍数",
                overlaying="y",
                side="right",
                showgrid=False,
                rangemode="tozero",
                tickformat=".1f"  # 显示1位小数
            )
        )

        # 添加倍数曲线（使用实际值和预测值）
        fig.add_trace(
            go.Scatter(
                x=full_data['ds'],
                y=full_data['forecast_multiple'].fillna(full_data['actual_multiple']),
                mode='lines',
                line=dict(color='purple', width=2),
                name='基准倍数',
                yaxis='y2',
                hovertemplate=
                    "<b>日期</b>: %{x|%Y-%m-%d}<br>" +
                    "<b>倍数</b>: %{y:.1f}x<extra></extra>"
            )
        )

        # 更新主图表布局（修复时间范围）
        min_date = ts_data['ds'].min()
        max_date = forecast['ds'].max()

        fig.update_layout(
            title=f'{category}热搜数量趋势及预测',
            xaxis=dict(
                range=[min_date - pd.DateOffset(months=1),
                       max_date + pd.DateOffset(months=1)],
                type='date',
                tickformat='%Y-%m'  # 统一日期格式
            ),
            yaxis=dict(
                title='热搜数量',
                rangemode='tozero'
            ),
            yaxis2=dict(
                title="基准倍数",
                overlaying="y",
                side="right",
                showgrid=False,
                rangemode="tozero",
                tickformat=".1f"  # 显示1位小数
            ),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )

        # 修复倍数曲线的时间范围
        full_data = full_data[
            (full_data['ds'] >= min_date - pd.DateOffset(months=1)) &
            (full_data['ds'] <= max_date + pd.DateOffset(months=1))
        ]

        return fig
    except Exception as e:
        st.error(f"预测失败: {str(e)}")
        return None

def plot_hourly_distribution(df):
    # 提取小时信息并统计
    df['hour'] = df['date'].dt.hour
    hourly_counts = df.groupby(['hour', 'category']).size().reset_index(name='count')

    # 计算每个类别的总数量用于标准化
    total_counts = df.groupby('category').size().reset_index(name='total')
    hourly_data = hourly_counts.merge(total_counts, on='category')
    hourly_data['percentage'] = (hourly_data['count'] / hourly_data['total']) * 100

    # 计算每个类别的小时排名
    hourly_data['rank'] = hourly_data.groupby('category')['percentage'].rank(ascending=False, method='min')
    hourly_data['color'] = np.select(
        [hourly_data['rank'] == 1, hourly_data['rank'] == 2, hourly_data['rank'] == 3],
        ['#FF0000', '#FFA500', '#FFFF00'],  # 红、橙、黄
        default='#D3D3D3'  # 灰色
    )

    # 创建分面柱状图（移除color_discrete_map参数）
    fig = px.bar(hourly_data,
                 x='hour',
                 y='percentage',
                 color='color',
                 facet_col='category',
                 facet_col_wrap=4,
                 title='各类热搜24小时分布（标准化百分比）',
                 labels={'hour': '小时', 'percentage': '占比 (%)'},
                 height=600)

    # 添加颜色说明标注（修改标注内容）
    fig.update_layout(
        annotations=[
            dict(
                text="颜色说明: <span style='color:#FF0000'>■</span>Top1 | "
                     "<span style='color:#FFA500'>■</span>Top2 | "
                     "<span style='color:#FFFF00'>■</span>Top3 | "
                     "<span style='color:#D3D3D3'>■</span>其他",
                xref="paper", yref="paper",
                x=0.5, y=1.08,
                showarrow=False,
                font=dict(size=12),
                align="center"
            )
        ]
    )

    # 隐藏自动生成的图例
    fig.update_layout(showlegend=False)

    # 统一坐标轴设置
    fig.update_xaxes(
        tickvals=list(range(24)),
        ticktext=[f"{h:02d}:00" for h in range(24)],
        matches=None
    )

    # 调整布局（新增yaxis配置）
    fig.update_layout(
        margin=dict(t=100),
        yaxis=dict(
            range=[0, hourly_data['percentage'].max()*1.1],
            showticklabels=False,  # 隐藏y轴刻度标签
            title=None            # 移除y轴标题
        ),
        hovermode="x unified",
        legend_title="小时排名",
        annotations=[
            dict(
                text=" ",
                xref="paper", yref="paper",
                x=0.5, y=1.08,
                showarrow=False,
                font=dict(size=12),
                align="center"
            )
        ]
    )

    # 新增：隐藏所有子图的y轴
    fig.update_yaxes(
        showgrid=False,   # 隐藏网格线
        showticklabels=False,  # 确保隐藏刻度标签
        title=None       # 确保移除标题
    )

    # 优化子图标题
    for annotation in fig.layout.annotations:
        if annotation.text:
            # 移除"category="前缀
            annotation.text = annotation.text.split("=")[-1]
            # 增大字体
            annotation.font = dict(size=14, color='#2c3e50')

    return fig

# 主应用
def main():
    st.title('Weibo Health Trend Analysis')
    st.caption('数据来源：[Entobit丨热搜神器Pro](https://www.entobit.cn/hot-search/desktop)')

    # 侧边栏配置
    st.sidebar.header('Data Filtering')

    # 加载数据
    markdown_folder = 'raw-markdown'
    default_files = [
         f'{markdown_folder}/咳嗽.md',f'{markdown_folder}/飞机.md',f'{markdown_folder}/肺炎.md',
        f'{markdown_folder}/感冒.md', f'{markdown_folder}/甲流.md',
        f'{markdown_folder}/离世.md', f'{markdown_folder}/流感.md',
        f'{markdown_folder}/去世.md', f'{markdown_folder}/生病.md',
        f'{markdown_folder}/心梗.md', f'{markdown_folder}/支原体.md',
        f'{markdown_folder}/坠机.md', f'{markdown_folder}/宏.md'
    ]

    # 新增文件上传功能
    uploaded_files = st.sidebar.file_uploader(
        "上传自定义MD文件（支持多个）",
        type=['md'],
        accept_multiple_files=True
    )

    # 合并默认文件和上传文件
    all_files = default_files.copy()
    if uploaded_files:
        # 保存上传文件到临时目录
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            all_files.append(file_path)

    # 加载数据（修改为使用all_files）
    df = load_and_process_data(all_files)

    # 时间范围选择
    date_range = st.sidebar.date_input(
        "选择时间范围",
        [df['date'].min(), df['date'].max()]
    )

    # 类别选择
    categories = st.sidebar.multiselect(
        "选择类别",
        options=df['category'].unique(),
        default=df['category'].unique()
    )

    # 数据筛选
    mask = (df['date'].dt.date >= date_range[0]) & \
           (df['date'].dt.date <= date_range[1]) & \
           (df['category'].isin(categories))
    filtered_df = df[mask]

    # 添加标签页
    tab1, tab2 = st.tabs(["Overall Analysis", "Single Category Analysis"])

    with tab1:
        st.subheader('Monthly Trend Analysis')
        st.plotly_chart(plot_monthly_trends(filtered_df))

        st.subheader('Rank Distribution Analysis')
        st.plotly_chart(plot_rank_distribution(filtered_df))

        st.subheader('Duration Analysis')
        st.plotly_chart(plot_duration_analysis(filtered_df))

        st.subheader('24-hour Distribution Analysis')
        st.plotly_chart(plot_hourly_distribution(filtered_df))

        # 数据统计
        st.subheader('基础统计信息')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="总热搜数量",
                value=f"{len(filtered_df):,}",
                help="筛选条件下的总热搜条目数"
            )
        with col2:
            st.metric(
                label="平均排名",
                value=f"{filtered_df['rank'].mean():.2f}",
                delta="TOP 50" if filtered_df['rank'].mean() < 50 else "低位热搜",
                delta_color="off",
                help="热搜上榜平均最高排名（数值越小排名越高）"
            )
        with col3:
            st.metric(
                label="平均持续时间",
                value=f"{filtered_df['duration_mins'].mean():.2f} 分钟",
                help="热搜在榜平均持续时间"
            )

    with tab2:
        # 修改为单选控件（与下方保持一致）
        selected_category = st.selectbox(
            "选择要分析的类别",
            options=categories,
            index=0 if len(categories) > 0 else None,
            key='category_selector'
        )

        # 显示预测图表（修改为处理单个类别）
        if selected_category:
            forecast_fig = plot_prophet_forecast(filtered_df, selected_category)
            if forecast_fig:
                st.plotly_chart(forecast_fig, use_container_width=True)

        # 显示该类别的时间线图（保持原有代码）
        if selected_category:
            st.plotly_chart(plot_timeline_gantt(filtered_df, selected_category))

        # 显示该类别的统计信息（保持原有代码）
        if selected_category:
            category_df = filtered_df[filtered_df['category'] == selected_category]
            st.subheader(f'{selected_category}类别统计信息')
            st.write(f"热搜数量: {len(category_df)}")
            st.write(f"平均排名: {category_df['rank'].mean():.2f}")
            st.write(f"平均持续时间: {category_df['duration_mins'].mean():.2f}分钟")

            # 新增动态数据表
            st.subheader('实时数据表')

            # 添加日期范围选择（使用两列布局）
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("开始日期",
                    value=category_df['date'].min().to_pydatetime(),
                    min_value=category_df['date'].min().to_pydatetime(),
                    max_value=category_df['date'].max().to_pydatetime())
            with col2:
                end_date = st.date_input("结束日期",
                    value=category_df['date'].max().to_pydatetime(),
                    min_value=category_df['date'].min().to_pydatetime(),
                    max_value=category_df['date'].max().to_pydatetime())

            # 添加多重排序选项
            sort_cols = st.multiselect(
                "排序字段（按优先级顺序）",
                options=['上榜时间', '持续分钟', '最高排名'],
                default=['上榜时间']
            )
            sort_orders = {col: st.selectbox(
                f"{col}排序方向",
                ['升序 ↑', '降序 ↓'],
                key=f"order_{col}"
            ) for col in sort_cols}

            # 处理数据
            dynamic_df = category_df[['title', 'date', 'duration_mins', 'rank']].copy()
            dynamic_df = dynamic_df[
                (dynamic_df['date'].dt.date >= start_date) &
                (dynamic_df['date'].dt.date <= end_date)
            ]

            # 应用多重排序
            if sort_cols:
                sort_ascending = [
                    order == '升序 ↑' for col, order in sort_orders.items()
                ]
                dynamic_df = dynamic_df.sort_values(
                    by=[col.replace('上榜时间', 'date')
                        .replace('持续分钟', 'duration_mins')
                        .replace('最高排名', 'rank') for col in sort_cols],
                    ascending=sort_ascending
                )
            else:
                dynamic_df = dynamic_df.sort_values('date', ascending=False)

            # 格式化显示（保持datetime类型）
            display_df = dynamic_df.copy()
            display_df.columns = ['热搜标题', '上榜时间', '持续分钟', '最高排名']

            # 使用增强版数据表格
            st.data_editor(
                display_df,
                height=400,
                column_config={
                    "上榜时间": st.column_config.DatetimeColumn(
                        format="YYYY-MM-DD HH:mm",
                        help="热搜首次上榜时间"
                    ),
                    "持续分钟": st.column_config.ProgressColumn(
                        format="%d 分钟",
                        help="热搜持续时长",
                        min_value=0,
                        max_value=int(display_df['持续分钟'].max() * 1.2)
                    ),
                    "最高排名": st.column_config.NumberColumn(
                        format="TOP %d",
                        help="历史最高排名"
                    )
                },
                use_container_width=True,
                key=f"datagrid_{selected_category}",
                disabled=True  # 如果不需要编辑功能建议保留
            )

            # 显示该类别的原始数据（保持原有代码）
            if st.checkbox(f'显示{selected_category}原始数据'):
                st.write(category_df)

    # 原始数据展示
    if st.checkbox('显示所有原始数据'):
        st.write(filtered_df)

if __name__ == "__main__":
    main()
