# Weibo Hot Search Analysis Tool

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)
一个综合的仪表板，用于分析Weibo热搜趋势，具有交互式可视化和时间序列预测功能。
还提供了下载热搜内容的Chrome/Edge插件。

## Table of Contents

- [Features](#features)
- [Data Sources](#data-sources)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Architecture](#technical-architecture)
- [Notes](#notes)
- [Roadmap](#roadmap)

## Features 🚀

**核心分析功能:**

- 📈 月度趋势分析与移动平均
- 📊 排名分布可视化
- ⏳ 事件持续时间统计 (Top 10 最长热搜)
- ☁️ 动态词云生成
- 🕒 小时分布模式
- 📅 交互式甘特图时间线

**高级功能:**

- 🔮 使用 Facebook Prophet 进行时间序列预测
- 🔍 交互式数据过滤:
  - 日期范围选择
  - 类别过滤
  - 最小持续时间阈值
- 📑 原始数据检查与可排序的列
- 🎨 双标签页布局组织

## Data Sources 📂

数据通过 [Entobit&#39;s Hot Search Pro Tool](https://entobit.com) 收集。用户可以:

1. 使用包含的示例数据 (`hotsearch.md`)
2. 通过界面上传自定义 Markdown 文件

## Installation ⚙️

```bash
# 克隆仓库
git clone [your-repo-url]
cd weibo-hotsearch-analysis

# 安装依赖
pip install -r requirements.txt
```

请确保您的项目根目录下包含 `requirements.txt` 文件，并将以下依赖项添加到 `requirements.txt` 文件中：

```
streamlit
pandas
plotly
jieba
wordcloud
numpy
scikit-learn
prophet
```

## Usage 🖥️

**启动应用:**

```bash
streamlit run analysis.py
```
