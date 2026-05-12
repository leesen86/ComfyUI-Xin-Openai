# ComfyUI-Xin-Openai

面向 **OpenAI 兼容接口**（`/v1/images/*`、`/v1/chat/completions`）的 ComfyUI 自定义节点，在同一界面内切换「文生图 / 图生图 / 图编辑」与「多模态对话（识图）」两种能力。


## 为什么开发这个组件

本人做设计的,对接企业,有很多生图要求,使用 runninghub.cn 的话经常排队,价格最低也比第三平台贵很多

我现在用的是 [`codeagent.cloud`](https://codeagent.cloud/) 这个大模型平台使用GPT-IMAGE-2生一个图才要几分钱,比 runninghub 便宜多了, 支持API,适合企业批量作图~


## 示例图



<p align="center">
  <img src="./example/example.png" alt="images 模式：双 Load Image 接 image1、image2，Xin OpenAI 节点与 Preview 工作流示例" width="100%" />
</p>








## 安装

1. **安装 ComfyUI**（[官方仓库](https://github.com/comfyanonymous/ComfyUI) 或你正在使用的发行版 / 便携包均可）。
2. **将本仓库克隆到 ComfyUI 的 `custom_nodes` 目录**。在终端中进入该目录后执行：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/leesen86/ComfyUI-Xin-Openai.git
```

若使用 **Windows 便携版**，路径一般为 `ComfyUI_windows_portable\ComfyUI\custom_nodes`，请先 `cd` 到该目录再执行上述 `git clone`（或将仓库解压到该目录下，文件夹名为 `ComfyUI-Xin-Openai` 即可）。

3. **启动 ComfyUI**。在 **`ComfyUI-Xin-Openai`** 分类下添加节点 **`Xin OpenAI - Image/Chat`**。

4. **安装 Python 依赖**（见下一节）：未装齐时节点可能无法加载。

### 安装 Python 依赖（`requirements.txt`）


<p style="color:#c62828;"><strong>注意（Windows 便携版）</strong><br/><br/>
<strong>由于</strong>启动脚本使用 <code>python_embeded\python.exe -s</code>，<code>-s</code> 会禁用用户目录下的 site-packages（例如 <code>%APPDATA%\...\Python312\site-packages</code>），而常见默认 pip 行为又容易把依赖装进该用户目录。<br/>
<strong>导致</strong> ComfyUI 进程里看不到这些包，节点加载时可能报错 <code>No module named 'openai'</code> 等。<br/><br/>
<strong>所以</strong>请在<strong>便携版根目录</strong>（与 <code>python_embeded</code> 同级）使用<strong>下面命令</strong>安装：先设置 <code>PYTHONNOUSERSITE=1</code>，让依赖写入嵌入式环境的 <code>Lib\site-packages</code>；若 pip 仍只认用户目录里的旧包，在命令末尾追加 <code>--force-reinstall</code>。</p>
<br/>

**PowerShell**（路径按你本机修改）：

```powershell
cd ComfyUI_windows_portable
$env:PYTHONNOUSERSITE='1'
.\python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Xin-Openai\requirements.txt
```


**CMD：**

```bat
cd ComfyUI_windows_portable
set PYTHONNOUSERSITE=1
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Xin-Openai\requirements.txt
```
