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
import urllib.parse


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

def plot_monthly_distribution(df):
    # 提取月份信息并统计
    df['month'] = df['date'].dt.month  # 改回使用月份数字
    monthly_counts = df.groupby(['month', 'category']).size().reset_index(name='count')

    # 计算每个类别的总数量用于标准化
    total_counts = df.groupby('category').size().reset_index(name='total')
    monthly_data = monthly_counts.merge(total_counts, on='category')
    monthly_data['percentage'] = (monthly_data['count'] / monthly_data['total']) * 100

    # 计算每个类别的月份排名
    monthly_data['rank'] = monthly_data.groupby('category')['percentage'].rank(ascending=False, method='min')
    monthly_data['color'] = np.select(
        [monthly_data['rank'] == 1, monthly_data['rank'] == 2, monthly_data['rank'] == 3],
        ['#FF0000', '#FFA500', '#FFFF00'],  # 红、橙、黄
        default='#D3D3D3'  # 灰色
    )

    # 创建分面柱状图
    fig = px.bar(monthly_data,
                 x='month',
                 y='percentage',
                 color='color',
                 facet_col='category',
                 facet_col_wrap=4,
                 title='各类热搜月份分布（标准化百分比）',
                 labels={'month': '月份', 'percentage': '占比 (%)'},
                 height=600)

    # 调整布局以为图例留出空间
    fig.update_layout(
        margin=dict(l=50, r=150, t=100, b=50),  # 增加右侧边距
    )

    # 隐藏原始数据的图例
    fig.update_traces(showlegend=False)

    # 更新图例样式
    fig.update_layout(
        showlegend=True,
        legend=dict(
            title="<b>颜色说明</b>",
            x=1.02,  # 位于图表右侧
            y=0.98,  # 位于顶部
            xanchor='left',  # 左对齐
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',  # 半透明白色背景
            bordercolor='black',
            borderwidth=1,
            itemsizing='constant',
            itemwidth=30,
            font=dict(size=12),
        )
    )

    # 只添加颜色说明的图例项
    fig.add_trace(
        go.Bar(
            x=[None], y=[None],
            name="最活跃月份",
            marker_color='#FF0000',
            showlegend=True
        )
    )
    fig.add_trace(
        go.Bar(
            x=[None], y=[None],
            name="第二活跃月份",
            marker_color='#FFA500',
            showlegend=True
        )
    )
    fig.add_trace(
        go.Bar(
            x=[None], y=[None],
            name="第三活跃月份",
            marker_color='#FFFF00',
            showlegend=True
        )
    )
    fig.add_trace(
        go.Bar(
            x=[None], y=[None],
            name="其他月份",
            marker_color='#D3D3D3',
            showlegend=True
        )
    )

    # 统一坐标轴设置
    fig.update_xaxes(
        tickvals=list(range(1, 13)),
        ticktext=['一月', '二月', '三月', '四月', '五月', '六月',
                 '七月', '八月', '九月', '十月', '十一月', '十二月'],
        matches=None
    )

    # 隐藏所有子图的y轴
    fig.update_yaxes(
        showgrid=False,
        showticklabels=False,
        title=None
    )

    # 优化子图标题
    for annotation in fig.layout.annotations:
        if annotation.text:
            annotation.text = annotation.text.split("=")[-1]
            annotation.font = dict(size=14, color='#2c3e50')

    return fig

def plot_topic_cooccurrence(df):
    """分析热搜话题共现关系"""
    # ... 现有代码 ...

    # 创建共现矩阵
    df['date_trunc'] = df['date'].dt.floor('D')  # 按天截断时间
    cooccurrence = pd.DataFrame(index=df['category'].unique(), columns=df['category'].unique())

    for date in df['date_trunc'].unique():
        daily_categories = df[df['date_trunc'] == date]['category'].unique()
        for cat1 in daily_categories:
            for cat2 in daily_categories:
                if cat1 != cat2:
                    cooccurrence.loc[cat1, cat2] = cooccurrence.loc[cat1, cat2] + 1 if not pd.isna(cooccurrence.loc[cat1, cat2]) else 1

    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=cooccurrence.values,
        x=cooccurrence.columns,
        y=cooccurrence.index,
        colorscale='Viridis',
        hoverongaps=False))

    fig.update_layout(
        title='热搜话题共现热力图',
        xaxis_title='类别',
        yaxis_title='类别',
        height=600
    )

    return fig

def plot_topic_velocity(df):
    """分析热搜传播速度"""
    # 计算每个热搜从首次出现到达最高排名的时间
    df_velocity = df.copy()
    df_velocity['velocity'] = df_velocity['rank'] / df_velocity['duration_mins']

    # 按类别统计平均传播速度
    velocity_by_category = df_velocity.groupby('category')['velocity'].mean().sort_values()

    # 创建条形图
    fig = go.Figure(data=go.Bar(
        x=velocity_by_category.index,
        y=velocity_by_category.values,
        marker_color='lightblue'
    ))

    fig.update_layout(
        title='各类热搜平均传播速度',
        xaxis_title='类别',
        yaxis_title='传播速度 (排名/分钟)',
        height=500
    )

    return fig

def plot_topic_impact_analysis(df):
    """分析热搜话题影响力"""
    # 计算综合影响力分数
    df_impact = df.copy()

    # 归一化处理
    df_impact['rank_score'] = 1 - (df_impact['rank'] / 50)  # 排名越高分数越高
    df_impact['duration_score'] = df_impact['duration_mins'] / df_impact['duration_mins'].max()

    # 计算综合影响力分数 (考虑排名和持续时间)
    df_impact['impact_score'] = (df_impact['rank_score'] * 0.6 +
                                df_impact['duration_score'] * 0.4)

    # 按类别统计平均影响力
    impact_by_category = df_impact.groupby('category').agg({
        'impact_score': ['mean', 'std'],
        'rank_score': 'mean',
        'duration_score': 'mean'
    }).round(3)

    # 创建多指标条形图
    fig = go.Figure()

    # 添加综合影响力
    fig.add_trace(go.Bar(
        name='综合影响力',
        x=impact_by_category.index,
        y=impact_by_category[('impact_score', 'mean')],
        error_y=dict(
            type='data',
            array=impact_by_category[('impact_score', 'std')],
            visible=True
        ),
        marker_color='rgb(55, 83, 109)'
    ))

    # 添加排名得分
    fig.add_trace(go.Bar(
        name='排名得分',
        x=impact_by_category.index,
        y=impact_by_category[('rank_score', 'mean')],
        marker_color='rgb(26, 118, 255)'
    ))

    # 添加持续时间得分
    fig.add_trace(go.Bar(
        name='持续时间得分',
        x=impact_by_category.index,
        y=impact_by_category[('duration_score', 'mean')],
        marker_color='rgb(158, 202, 225)'
    ))

    fig.update_layout(
        title='各类热搜影响力分析',
        xaxis_title='类别',
        yaxis_title='得分',
        barmode='group',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def plot_sentiment_analysis(df):
    """分析热搜话题情感倾向"""
    # 计算情感得分
    df['sentiment'] = df['title'].apply(lambda x: SnowNLP(x).sentiments)

    # 按类别统计平均情感得分
    sentiment_by_category = df.groupby('category')['sentiment'].agg(['mean', 'std']).round(3)

    # 创建误差条形图
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='情感得分',
        x=sentiment_by_category.index,
        y=sentiment_by_category['mean'],
        error_y=dict(
            type='data',
            array=sentiment_by_category['std'],
            visible=True
        )
    ))

    fig.update_layout(
        title='各类热搜情感分析',
        xaxis_title='类别',
        yaxis_title='情感得分 (0-负面, 1-正面)',
        height=500
    )

    return fig

def plot_word_frequency(df, category=None):
    """分析热搜话题关键词频率"""
    # 过滤特定类别（如果指定）
    if category:
        df = df[df['category'] == category]

    # 分词并统计词频
    text = ' '.join(df['title'])
    words = jieba.cut(text)
    word_freq = pd.Series(' '.join(words).split()).value_counts()

    # 创建条形图
    fig = go.Figure(data=go.Bar(
        x=word_freq.head(20).index,
        y=word_freq.head(20).values,
        marker_color='lightgreen'
    ))

    title = f'{"" if not category else category + "类别"}热搜关键词TOP20'
    fig.update_layout(
        title=title,
        xaxis_title='关键词',
        yaxis_title='出现次数',
        height=500
    )

    return fig

def parse_logic_expression(expression):
    """解析逻辑表达式
    支持格式:
    - 单个词: "发烧"
    - AND: "发烧 and 咳嗽"
    - OR: "发烧 or 咳嗽"
    - 括号: "(发烧 or 咳嗽) and 医院"
    """
    def tokenize(expr):
        # 处理括号
        expr = expr.replace('(', ' ( ').replace(')', ' ) ')
        return expr.lower().split()

    def parse(tokens):
        if not tokens:
            return None

        def parse_term():
            token = tokens.pop(0)
            if token == '(':
                result = parse_expression()
                if tokens and tokens[0] == ')':
                    tokens.pop(0)
                return result
            return token

        def parse_expression():
            terms = [parse_term()]
            while tokens and tokens[0] in ['and', 'or']:
                op = tokens.pop(0)
                terms.append(op)
                terms.append(parse_term())
            return terms

        return parse_expression()

    tokens = tokenize(expression)
    return parse(tokens)

def evaluate_logic(df, parsed_expr):
    """评估逻辑表达式"""
    if not parsed_expr:
        return pd.Series(True, index=df.index)

    def evaluate_term(term):
        if isinstance(term, list):
            return evaluate_logic(df, term)
        if term in ['and', 'or']:
            return term
        return df['title'].str.contains(term)

    if len(parsed_expr) == 1:
        return evaluate_term(parsed_expr[0])

    result = evaluate_term(parsed_expr[0])
    i = 1
    while i < len(parsed_expr):
        op = parsed_expr[i]
        next_term = evaluate_term(parsed_expr[i + 1])
        if op == 'and':
            result = result & next_term
        elif op == 'or':
            result = result | next_term
        i += 2

    return result

def analyze_keyword_patterns(df, category, include_expression=None, exclude_expression=None, must_include_main=True, start_date=None, end_date=None):
    """分析热搜标题中的关键词模式

    Args:
        df: 数据框
        category: 类别名称
        include_expression: 包含关键词的逻辑表达式
        exclude_expression: 排除关键词的逻辑表达式
        must_include_main: 是否必须包含主关键词
        start_date: 开始日期
        end_date: 结束日期
    """
    # 获取该类别的数据
    category_df = df[df['category'] == category].copy()

    # 应用日期筛选
    if start_date and end_date:
        category_df = category_df[
            (category_df['date'].dt.date >= start_date) &
            (category_df['date'].dt.date <= end_date)
        ]

    # 初始化筛选条件
    filtered_df = category_df.copy()

    # 应用主关键词筛选
    if must_include_main:
        filtered_df = filtered_df[filtered_df['title'].str.contains(category)]

    # 应用包含关键词筛选
    if include_expression:
        try:
            parsed_expr = parse_logic_expression(include_expression)
            mask = evaluate_logic(filtered_df, parsed_expr)
            filtered_df = filtered_df[mask]
        except Exception as e:
            st.error(f"包含关键词表达式解析错误: {str(e)}")

    # 应用排除关键词筛选
    if exclude_expression:
        try:
            parsed_expr = parse_logic_expression(exclude_expression)
            mask = evaluate_logic(filtered_df, parsed_expr)
            filtered_df = filtered_df[~mask]
        except Exception as e:
            st.error(f"排除关键词表达式解析错误: {str(e)}")

    # 修改统计信息，添加时间范围说明
    date_range_str = ""
    if start_date and end_date:
        date_range_str = f"({start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')})"

    stats = {
        '符合条件的热搜数': len(filtered_df),
        '占该类别总数比例': f"{(len(filtered_df) / len(category_df) * 100):.1f}%",
        '平均排名': f"{filtered_df['rank'].mean():.1f}",
        '平均持续时间': f"{filtered_df['duration_mins'].mean():.1f}分钟",
        '时间范围': date_range_str
    }

    return filtered_df, stats

def get_top_keywords(df, category, top_n=20):
    """获取某个类别最常见的关键词"""
    category_df = df[df['category'] == category]
    text = ' '.join(category_df['title'])
    words = jieba.cut(text)
    # 过滤掉单字词和类别名本身
    word_freq = pd.Series(
        [w for w in words if len(w) > 1 and w != category]
    ).value_counts()
    return word_freq.head(top_n)

def plot_filtered_timeline(df, title):
    """为筛选后的数据绘制时间线"""
    fig = go.Figure()

    # 按时间排序
    df_sorted = df.sort_values('date')

    # 添加散点图
    fig.add_trace(go.Scatter(
        x=df_sorted['date'],
        y=df_sorted['rank'],
        mode='markers+text',
        text=df_sorted['title'],
        textposition="top center",
        marker=dict(
            size=10,
            color='red',
            symbol='circle'
        ),
        hovertemplate="<b>%{text}</b><br>" +
                     "时间: %{x}<br>" +
                     "排名: %{y}<br>" +
                     "<extra></extra>"
    ))

    # 更新布局
    fig.update_layout(
        title=title,
        xaxis_title='时间',
        yaxis_title='排名',
        yaxis=dict(
            autorange="reversed",  # 反转y轴使排名1在顶部
            tickmode='array',
            ticktext=[f'TOP {i}' for i in range(1, 51, 5)],
            tickvals=list(range(1, 51, 5))
        ),
        height=400,
        showlegend=False
    )

    return fig

def analyze_filtered_results(df):
    """对筛选结果进行深入分析"""
    # 按年统计
    df['year'] = df['date'].dt.year
    yearly_stats = df.groupby('year').agg({
        'title': 'count',
        'rank': ['mean', 'min'],
        'duration_mins': ['mean', 'max']
    }).round(2)
    yearly_stats.columns = ['热搜数量', '平均排名', '最高排名', '平均持续时间', '最长持续时间']

    # 按月份分布
    df['month'] = df['date'].dt.month
    monthly_dist = df.groupby('month')['title'].count()

    # 按小时分布
    df['hour'] = df['date'].dt.hour
    hourly_dist = df.groupby('hour')['title'].count()

    # 创建分析图表
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '年度热搜数量对比',
            '月份分布',
            '每日时段分布',
            '排名与持续时间关系'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )

    # 1. 年度数量对比
    fig.add_trace(
        go.Bar(
            x=yearly_stats.index,
            y=yearly_stats['热搜数量'],
            text=yearly_stats['热搜数量'],
            textposition='auto',
        ),
        row=1, col=1
    )

    # 2. 月份分布
    fig.add_trace(
        go.Bar(
            x=[f"{m}月" for m in monthly_dist.index],
            y=monthly_dist.values,
            text=monthly_dist.values,
            textposition='auto',
        ),
        row=1, col=2
    )

    # 3. 时段分布
    fig.add_trace(
        go.Bar(
            x=[f"{h:02d}:00" for h in hourly_dist.index],
            y=hourly_dist.values,
            text=hourly_dist.values,
            textposition='auto',
        ),
        row=2, col=1
    )

    # 4. 排名与持续时间散点图
    # 修改：使用时间戳而不是直接转换为int
    timestamps = df['date'].astype('int64') // 10**9  # 转换为Unix时间戳
    min_ts = timestamps.min()
    max_ts = timestamps.max()
    # 归一化时间戳到0-1区间，用于颜色映射
    normalized_time = (timestamps - min_ts) / (max_ts - min_ts) if max_ts > min_ts else timestamps * 0

    fig.add_trace(
        go.Scatter(
            x=df['rank'],
            y=df['duration_mins'],
            mode='markers',
            marker=dict(
                size=8,
                color=normalized_time,  # 使用归一化后的时间
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="时间进程",
                    ticktext=["早期", "中期", "后期"],
                    tickvals=[0, 0.5, 1]
                )
            ),
            hovertemplate=(
                "排名: %{x}<br>" +
                "持续时间: %{y}分钟<br>" +
                "时间: " + df['date'].dt.strftime('%Y-%m-%d').astype(str) +
                "<extra></extra>"
            )
        ),
        row=2, col=2
    )

    # 更新布局
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="筛选结果深入分析"
    )

    # 更新x轴
    fig.update_xaxes(title_text="年份", row=1, col=1)
    fig.update_xaxes(title_text="月份", row=1, col=2)
    fig.update_xaxes(title_text="时段", row=2, col=1)
    fig.update_xaxes(title_text="排名", row=2, col=2)

    # 更新y轴
    fig.update_yaxes(title_text="热搜数量", row=1, col=1)
    fig.update_yaxes(title_text="热搜数量", row=1, col=2)
    fig.update_yaxes(title_text="热搜数量", row=2, col=1)
    fig.update_yaxes(title_text="持续时间(分钟)", row=2, col=2)

    return fig, yearly_stats

def predict_yearly_trend(df):
    """预测年度热搜趋势，只使用历史完整年份数据"""
    # 按年月统计热搜数
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    monthly_counts = df.groupby(['year', 'month']).size().reset_index(name='count')

    # 计算2025年的预计数量
    current_year = 2025
    current_data = monthly_counts[monthly_counts['year'] == current_year]

    if not current_data.empty:
        # 只使用历史完整年份的数据进行预测
        yearly_counts = df[df['year'] < current_year].groupby('year').size().reset_index(name='count')

        # 准备Prophet数据
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(yearly_counts['year'].astype(str)),
            'y': yearly_counts['count']
        })

        # 使用Prophet进行预测
        model = Prophet(
            yearly_seasonality=True,
            changepoint_prior_scale=0.5,
            changepoint_range=0.9
        )
        model.fit(prophet_df)

        # 生成预测数据（包括2025年）
        future = model.make_future_dataframe(periods=2, freq='Y')
        forecast = model.predict(future)

        # 获取2025年的预测值
        predicted_total = forecast.loc[forecast['ds'].dt.year == current_year, 'yhat'].iloc[0]

        # 创建图表
        fig = go.Figure()

        # 添加历史数据点
        fig.add_trace(go.Scatter(
            x=yearly_counts['year'],
            y=yearly_counts['count'],
            mode='markers',
            name='历史数据',
            marker=dict(size=10, color='blue')
        ))

        # 添加2025年实际数据点
        actual_count = df[df['year'] == current_year].shape[0]
        months_recorded = len(current_data)
        fig.add_trace(go.Scatter(
            x=[current_year],
            y=[actual_count],
            mode='markers',
            name='2025年当前',
            marker=dict(size=10, color='orange')
        ))

        # 添加趋势线
        fig.add_trace(go.Scatter(
            x=forecast['ds'].dt.year,
            y=forecast['yhat'],
            mode='lines',
            name='趋势线',
            line=dict(color='green', dash='dash')
        ))

        # 添加预测区间
        fig.add_trace(go.Scatter(
            x=forecast['ds'].dt.year.tolist() + forecast['ds'].dt.year.tolist()[::-1],
            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,176,246,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% 置信区间'
        ))

        # 更新布局
        fig.update_layout(
            title=f'年度热搜数量趋势 (2025年预测: {predicted_total:.0f}条)',
            xaxis_title='年份',
            yaxis_title='热搜数量',
            hovermode='x unified',
            height=500
        )

        # 添加注释说明
        fig.add_annotation(
            text=(f"2025年实际情况：<br>" +
                  f"已记录{months_recorded}个月，共{actual_count}条<br>" +
                  f"基于历史数据预测全年：{predicted_total:.0f}条"),
            xref="paper", yref="paper",
            x=1, y=0,
            xanchor='right', yanchor='bottom',
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)"
        )

        return fig, {
            'months_recorded': months_recorded,
            'current_count': actual_count,
            'predicted_total': predicted_total
        }

    return None, None

# 主应用
def main():
    # 新增页面配置
    st.set_page_config(
        page_title="Weibo Health Trend Analysis",
        layout="wide",  # 设置为宽屏模式
        initial_sidebar_state="expanded"
    )

    st.title('Weibo Health Trend Analysis')
    st.caption('数据来源：[Entobit丨热搜神器Pro](https://www.entobit.cn/hot-search/desktop)')

    # 侧边栏配置
    st.sidebar.header('Data Filtering')

    # 加载数据
    markdown_folder = 'raw-markdown'
    default_files = [
         f'{markdown_folder}/咳嗽.md'
    ]
    st.sidebar.caption("[点击查看示例md文件](https://github.com/Feather-2/Wb-Health-Trend-Analysis/blob/main/raw-markdown/%E6%94%AF%E5%8E%9F%E4%BD%93.md)")
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
        # 创建子标签页
        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "趋势分析", "排名分析", "时间分布", "关联分析"
        ])

        with subtab1:
            # 添加基础统计信息
            st.subheader('Base Statistics')
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

            st.subheader('Monthly Trend Analysis')
            st.plotly_chart(plot_monthly_trends(filtered_df))

            # 添加年度趋势预测
            st.subheader('Yearly Trend Prediction')
            trend_fig, trend_stats = predict_yearly_trend(filtered_df)
            if trend_fig and trend_stats:
                # 显示预测统计
                # 显示趋势图
                st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.warning("暂无足够数据进行趋势预测")

        with subtab2:
            st.subheader('Rank Distribution Analysis')
            st.plotly_chart(plot_rank_distribution(filtered_df))

            st.subheader('Duration Analysis')
            st.plotly_chart(plot_duration_analysis(filtered_df))


        with subtab3:
            st.subheader('24-hour Distribution Analysis')
            st.plotly_chart(plot_hourly_distribution(filtered_df))

            st.subheader('Monthly Distribution Analysis')
            st.plotly_chart(plot_monthly_distribution(filtered_df))

        with subtab4:
            st.subheader('话题共现分析')
            st.plotly_chart(plot_topic_cooccurrence(filtered_df))

            st.subheader('传播速度分析')
            st.plotly_chart(plot_topic_velocity(filtered_df))

            st.subheader('话题影响力分析')
            st.plotly_chart(plot_topic_impact_analysis(filtered_df))

    with tab2:
        # 保持类别选择在最上方
        selected_category = st.selectbox(
            "选择要分析的类别",
            options=categories,
            index=0 if len(categories) > 0 else None,
            key='category_selector'
        )

        if selected_category:
            # 创建子标签页
            subtab1, subtab2, subtab3, subtab4 = st.tabs([
                "趋势预测", "时间线分析", "统计信息", "关键词分析"
            ])

            with subtab1:
                st.subheader('热搜数量趋势预测')
                forecast_fig = plot_prophet_forecast(filtered_df, selected_category)
                if forecast_fig:
                    st.plotly_chart(forecast_fig, use_container_width=True)
                else:
                    st.warning("数据量不足，无法进行预测")

            with subtab2:
                st.subheader('相关热搜时间线')
                st.plotly_chart(plot_timeline_gantt(filtered_df, selected_category))

            with subtab3:
                # 统计信息部分
                st.subheader(f'{selected_category}类别统计信息')
                category_df = filtered_df[filtered_df['category'] == selected_category]

                # 使用三列布局显示基础统计
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="热搜数量",
                        value=len(category_df),
                        help="该类别下的总热搜条目数"
                    )
                with col2:
                    st.metric(
                        label="平均排名",
                        value=f"{category_df['rank'].mean():.2f}",
                        delta="TOP 50" if category_df['rank'].mean() < 50 else "低位热搜",
                        delta_color="off",
                        help="该类别热搜的平均最高排名"
                    )
                with col3:
                    st.metric(
                        label="平均持续时间",
                        value=f"{category_df['duration_mins'].mean():.2f} 分钟",
                        help="该类别热搜的平均在榜时间"
                    )

                # 实时数据表
                st.subheader('实时数据表')

                # 日期范围选择
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "开始日期",
                        value=category_df['date'].min().to_pydatetime(),
                        min_value=category_df['date'].min().to_pydatetime(),
                        max_value=category_df['date'].max().to_pydatetime()
                    )
                with col2:
                    end_date = st.date_input(
                        "结束日期",
                        value=category_df['date'].max().to_pydatetime(),
                        min_value=category_df['date'].min().to_pydatetime(),
                        max_value=category_df['date'].max().to_pydatetime()
                    )

                # 排序选项
                sort_cols = st.multiselect(
                    "排序字段（按优先级顺序）",
                    options=['上榜时间', '持续分钟', '最高排名'],
                    default=['上榜时间']
                )

                # 显示增强版数据表格
                display_df = prepare_display_dataframe(category_df, start_date, end_date, sort_cols)
                st.data_editor(
                    display_df,
                    height=400,
                    column_config=get_column_config(display_df),
                    use_container_width=True,
                    key=f"datagrid_{selected_category}",
                    disabled=True
                )

            with subtab4:
                # 关键词分析部分
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader('热搜关键词TOP20')
                    st.plotly_chart(
                        plot_word_frequency(filtered_df, selected_category),
                        use_container_width=True
                    )

                    # 显示热门关键词列表供参考
                    st.write('#### 常见关键词参考')
                    top_keywords = get_top_keywords(filtered_df, selected_category)
                    keyword_text = ', '.join([f"{word}({count}次)" for word, count in top_keywords.items()])
                    st.text_area(
                        "按出现次数排序的关键词列表",
                        keyword_text,
                        height=100,
                        help="可以复制这些关键词用于筛选分析"
                    )

                with col2:
                    st.subheader('关键词模式分析')

                    # 添加日期范围选择
                    date_col1, date_col2 = st.columns(2)
                    with date_col1:
                        kw_start_date = st.date_input(
                            "分析开始日期",
                            value=category_df['date'].min().date(),
                            min_value=category_df['date'].min().date(),
                            max_value=category_df['date'].max().date(),
                            key="kw_start_date"
                        )
                    with date_col2:
                        kw_end_date = st.date_input(
                            "分析结束日期",
                            value=category_df['date'].max().date(),
                            min_value=category_df['date'].min().date(),
                            max_value=category_df['date'].max().date(),
                            key="kw_end_date"
                        )

                    include_expression = st.text_area(
                        '包含关键词表达式',
                        help='输入包含关键词的逻辑表达式，支持 and/or 和括号'
                    )

                    exclude_expression = st.text_area(
                        '排除关键词表达式',
                        help='输入要排除的关键词的逻辑表达式，支持 and/or 和括号'
                    )

                    must_include_main = st.checkbox(
                        f'必须包含主关键词"{selected_category}"',
                        value=True
                    )

                    # 添加示例
                    st.markdown("""
                    #### 示例:
                    - `发烧 and (医院 or 诊所)`：标题中必须包含"发烧"，且包含"医院"或"诊所"之一
                    - `(重症 or 危重) and 治疗`：标题中必须包含"重症"或"危重"之一，且包含"治疗"
                    """)

                # 关键词筛选结果
                if include_expression or exclude_expression:
                    filtered_results, stats = analyze_keyword_patterns(
                        filtered_df,
                        selected_category,
                        include_expression,
                        exclude_expression,
                        must_include_main,
                        kw_start_date,
                        kw_end_date
                    )

                    # 显示筛选统计
                    st.write('#### 筛选结果统计')
                    if stats['时间范围']:
                        st.write(f"时间范围: {stats['时间范围']}")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric('符合条件的热搜数', stats['符合条件的热搜数'])
                    col2.metric('占该类别总数比例', stats['占该类别总数比例'])
                    col3.metric('平均排名', stats['平均排名'])
                    col4.metric('平均持续时间', stats['平均持续时间'])

                    # 显示筛选结果时间线
                    if not filtered_results.empty:
                        st.plotly_chart(
                            plot_filtered_timeline(
                                filtered_results,
                                f'符合条件的{selected_category}类热搜时间线'
                            ),
                            use_container_width=True
                        )

                        # 添加深入分析
                        st.write('#### 筛选结果深入分析')
                        analysis_fig, yearly_stats = analyze_filtered_results(filtered_results)

                        # 显示年度统计表格
                        st.write('##### 年度统计数据')
                        st.dataframe(
                            yearly_stats,
                            use_container_width=True
                        )

                        # 添加年度趋势预测
                        st.write('##### 年度趋势预测')
                        trend_fig, trend_stats = predict_yearly_trend(filtered_results)
                        if trend_fig and trend_stats:
                            # 显示预测统计
                            col1, col2, col3 = st.columns(3)
                            col1.metric(
                                "2025年已记录月数",
                                f"{trend_stats['months_recorded']}个月"
                            )
                            col2.metric(
                                "2025年月均热搜数",
                                f"{trend_stats['current_count']:.1f}条"
                            )
                            col3.metric(
                                "2025年预计总数",
                                f"{trend_stats['predicted_total']:.0f}条"
                            )

                            # 显示趋势图
                            st.plotly_chart(trend_fig, use_container_width=True)
                        else:
                            st.info("筛选结果数据量不足，无法进行趋势预测")

                        # 显示分析图表
                        st.plotly_chart(
                            analysis_fig,
                            use_container_width=True
                        )

                        # 显示热搜详情
                        st.write('#### 符合条件的热搜详情')
                        st.dataframe(
                            filtered_results[['title', 'date', 'rank', 'duration_mins']],
                            use_container_width=True
                        )
                    else:
                        st.warning('没有找到符合条件的热搜')

# 新增辅助函数
def prepare_display_dataframe(df, start_date, end_date, sort_cols):
    """准备用于显示的数据框"""
    # 首先选择需要的列
    display_df = df[['title', 'date', 'duration_mins', 'rank']].copy()

    # 应用日期过滤
    display_df = display_df[
        (display_df['date'].dt.date >= start_date) &
        (display_df['date'].dt.date <= end_date)
    ]

    # 添加链接列
    display_df['link'] = display_df['title'].apply(
        lambda x: f'https://s.weibo.com/weibo?q=%23{urllib.parse.quote(x)}%23'
    )

    # 应用排序
    if sort_cols:
        sort_ascending = [True] * len(sort_cols)
        sort_columns = [
            col.replace('上榜时间', 'date')
               .replace('持续分钟', 'duration_mins')
               .replace('最高排名', 'rank')
            for col in sort_cols
        ]
        display_df = display_df.sort_values(by=sort_columns, ascending=sort_ascending)

    # 重命名列
    display_df.columns = ['热搜标题', '上榜时间', '持续分钟', '最高排名', '实时链接']

    return display_df

def get_column_config(df):
    """获取数据表格的列配置"""
    return {
        "实时链接": st.column_config.LinkColumn(
            help="点击查看实时微博讨论",
            display_text="查看",
            width="small"
        ),
        "热搜标题": st.column_config.Column(
            width="large"
        ),
        "上榜时间": st.column_config.DatetimeColumn(
            format="YYYY-MM-DD HH:mm",
            help="热搜首次上榜时间"
        ),
        "持续分钟": st.column_config.ProgressColumn(
            format="%d 分钟",
            help="热搜持续时长",
            min_value=0,
            max_value=int(df['持续分钟'].max() * 1.2)
        ),
        "最高排名": st.column_config.NumberColumn(
            format="TOP %d",
            help="历史最高排名"
        )
    }

if __name__ == "__main__":
    main()
