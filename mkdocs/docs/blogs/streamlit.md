---
title: Streamlit：用Python写一个简易、美观的App并免费部署！
---
{% set pic_path = https://github.com/Treedy2020/TreeDyStack/tree/master/mkdocs/docs/blogs/streamlit/}

This is the picture path : {{pic_path}}

And this is the picture:

![Ventura]({{pic_path}}/venrura.png)

# Streamlit  
以Python为基础开发语言的开发者，特别是算法工程师们的一个痛点是**开发内容的美观呈现**。我在写一个程序时想完成的不仅是程序的核心逻辑，理想中的状态是，我能通过Python简单的几行代码，就将自己的程序写成一个可交互的美观的App或者网页。:smile:[Streamlit](https://streamlit.io/ 点击进入Streamlit官网)让你可以不用写HTML，CSS，JS，就能轻松实现这个目标，它基于Github的项目仓库进行版本管理，并且可以**免费的进行App部署**。
除此以外，它有诸多优点：

- 开源
- 高度集成易用的API接口
- 丰富的模版库

## 安装和快速使用

### 安装
```
$ pip install streamit  
```

### 快速使用
Streamlit中，一个页面由一个.py脚本创建，因此你的文件树应该是这样的：

    -- ProjectName
        -- HomePage.py
打开HomePage.py，写一个简单的开始界面：

```py title="HomePage.py" linenums="1"
import streamlit as st
st.title("Hello World")
```

结束后保存，然后在命令行运行

```
$ streamlit run HomePage.py
```

默认浏览器就会运行，界面即是刚才写的`HomePage.py`。

!!! note "或不需要创建项目，使用官方提供一个简易的Demo"
    $ streamlit hello

## 使用案例
比如我们训练了一个Seq2seq的文字纠错模型`model`，它接受一个有错的字符串`str`作为输入，返回一个修正之后的字符串`str`，我想要将这个模型可视化的呈现出来，那么我可以这样写：
```py title="Seq2Seq.py" linenums="1"
import streamlit as st

# 为App添加标题
st.title("简单语言纠错模型")

# 显示加载模型的提示
st.write("Loading model...")   
model = torch.load('model.pth') # 显示加载模型的提示
st.write("Model loaded!")    # 显示模型加载完成的提示

```
接着，我们可以为用户写一个输入框，让用户输入有错的字符串，然后将其传入模型中，得到修正之后的字符串，最后将结果显示出来：
```py title="Seq2Seq.py" linenums="1"

# 用户输入框和确认按钮
user_input = st.text_area("请输入您想要修改的句子: ")
enter = st.button("确认")

# 模型预测
if enter:  
    with st.spinner("正在修正..."):
        result = model(user_input)
    st.success("修正结果为：", result)
```
这样，用户每次点击`确认`按钮，就会得到一个新的修正结果，当模型推理结束之后，就会将结果呈现出来；我们保存Seq2Seq.py，然后在命令行运行：

``` command
$ streamlit run Seq2Seq.py
```
默认浏览器就会运行，界面即是刚才写的`Seq2Seq.py`。这么短的代码有这样的效果是不是已经很惊艳了，不过这个网页暂时只在你的本地运行，现在还可以通过Stream Community Cloud将这个简单的App部署到云端，让更多的人使用，你只需要：

1. 将你的项目上传至Github；
2. 注册并登陆[Streamlit Cloud](https://streamlit.io/cloud "前往Streamlit Cloud“)；
3. 点击Streamlit Cloud登陆后界面的`New App`，并将Github Repository的地址填入，并填写`Main file path`为刚才我们所创建的`Seq2Seq.py`点击`Deploy！`，就可以将你的App部署到云端了。   

## 常用API
Streamlit提供了丰富的API接口，你可以在[官方文档](https://docs.streamlit.io/library/api-reference)中查到下边这些API的使用Demo，它们各自都可以用极少的代码创建出对应类型的元素：

- 文本
    - Markdown
    - 代码块
    - 预置格式文本
    - LaTeX
    - 分割线
- 多媒体
    - 图片
    - 音频
    - 视频

- 数据展示
    - Pandas数据帧
    - 静态表格
    - 指标
    - 字典或JSON格式的数据
- 曲线
    - 折线图
    - 面积图
    - 柱状图
    - 散点图
    - Matplotlib、Altair、Vega-Lite等绘图库
    - ...

- 输入组件
    - 按钮
    - 数字编辑器
    - 下载
    - 上传
    - 检查框、选择框、多选框、滑动（选择）条
    - 文本输入条
    - ...


!!! tip "更多API"

    除了这些常用的交互组件以外，Streamlit还提供排版容器(Layouts and containers)、进度条和状态(Progress and status)、流控制(Control flow)，缓存(Cache库)等其他好用的API，除了这些以外，还有一些更复杂的组件可以帮助你进行用户登陆、添加评论区等更复杂的功能，请参见[API文档](https://docs.streamlit.io/library/api-reference)。


## 开源模版

Streamlit还提供了丰富的开源模版，这些模版的类型包括科学和技术介绍、自然语言处理、CV应用等。你可以在[Streamlit Gallery](https://streamlit.io/gallery "前往Streamlit Gallery")中查看这些模版的效果并让它们为你所用，只需要改正少量的代码即可。




​    









