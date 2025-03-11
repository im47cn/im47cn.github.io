---
title: Mac系统下matplotlib绘图中文显示问题的解决方案
author: dreambt
date: 2025-03-11 23:59:00 +0800
categories: [Python, 数据可视化]
tags: [Mac, matplotlib, Python, 字体配置]
description: 详细介绍如何解决Mac系统下matplotlib绘图时中文显示乱码的问题
image: /assets/2025/03/image_matplotlib.png
---

## 问题背景

在Mac系统下使用matplotlib进行数据可视化时，经常会遇到中文显示为方块或乱码的问题。这是因为matplotlib默认不支持中文字体，需要我们手动配置中文字体才能正常显示。本文将详细介绍如何解决这个问题。

## 解决方案概述

解决matplotlib中文显示问题主要需要完成以下步骤：

1. 下载中文字体文件（SimHei.ttf）
2. 将字体文件放入matplotlib的字体目录
3. 修改matplotlib配置文件
4. 清除字体缓存

下面我们详细介绍每个步骤的具体操作。

## 1. 下载SimHei字体

首先需要获取中文字体文件。SimHei（黑体）是一个常用的中文字体，可以从网络上下载。建议从可靠的源下载字体文件，确保文件名为`SimHei.ttf`。

```bash
wget "https://github.com/StellarCN/scp_zh/blob/master/fonts/SimHei.ttf"
```

## 2. 定位和配置字体文件

### 2.1 查找matplotlib配置文件路径

使用以下Python代码查找matplotlib的配置文件路径：

```python
import matplotlib
print(matplotlib.matplotlib_fname())
```

这会返回类似 `/xxx/python3.8/site-packages/matplotlib/mpl-data/matplotlibrc` 的路径。

### 2.2 放置字体文件

将下载的`SimHei.ttf`文件复制到matplotlib的字体目录中。字体目录位于 `/xxx/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/`。

## 3. 修改matplotlib配置文件

打开之前找到的`matplotlibrc`文件，修改以下配置项：

```bash
# 启用字体家族设置
font.family:  sans-serif

# 设置sans-serif字体列表，将SimHei放在首位
font.sans-serif: SimHei, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif

# 解决负号显示问题
axes.unicode_minus: False
```

## 4. 清除字体缓存

### 4.1 查找缓存目录

使用以下代码查找matplotlib的缓存目录：

```python
import matplotlib
print(matplotlib.get_cachedir())
```

### 4.2 删除缓存

在终端中执行以下命令删除缓存（将xxx替换为实际的用户名）：

```bash
rm -rf /Users/xxx/.matplotlib
```

## 验证配置

完成上述步骤后，可以使用以下代码验证中文是否正常显示：

```python
import matplotlib.pyplot as plt
import numpy as np

# 创建示例数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 绘制图形
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('正弦波图像')
plt.xlabel('时间（秒）')
plt.ylabel('振幅')
plt.grid(True)
plt.show()
```

## 常见问题

1. 如果配置完成后中文仍然显示为方块，请确保：
   - 下载的SimHei.ttf文件完整性，大约5MB左右
   - SimHei.ttf文件已正确放置在字体目录
   - 配置文件修改正确
   - 已清除字体缓存

2. 如果负号显示异常，检查`axes.unicode_minus`配置是否正确设置为`False`

3. 如果需要使用其他中文字体，可以将对应的.ttf文件放入字体目录，并在配置文件中相应修改`font.sans-serif`的设置

## 总结

通过以上步骤，我们可以解决Mac系统下matplotlib绘图时中文显示的问题。这个配置是一次性的，配置完成后就可以在所有Python程序中正常显示中文了。建议在进行数据可视化项目之前，先确保中文显示的问题得到解决，这样可以避免在后期遇到不必要的麻烦。