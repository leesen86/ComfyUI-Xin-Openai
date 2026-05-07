# ComfyUI-Xin-Openai

面向 **OpenAI 兼容接口**（`/v1/images/*`、`/v1/chat/completions`）的 ComfyUI 自定义节点，在同一界面内切换「文生图 / 图生图 / 图编辑」与「多模态对话（识图）」两种能力。

## 示例图

以下为仓库内 [`example/example.png`](example/example.png) 的展示（本地用 IDE 打开本 `readme.md` 时也会渲染；若网页端不显示，请直接打开该图片文件）。

<p align="center">
  <img src="./example/example.png" alt="images 模式：双 Load Image 接 image1、image2，Xin OpenAI 节点与 Preview 工作流示例" width="100%" />
</p>

---

## 为什么开发这个组件

本人做设计的,对接企业,有很多生图要求,使用 runninghub.cn 的话经常排队,价格最低也比第三平台贵很多

我现在用的是 [`uclaude.cc`](https://uclaude.cc/) 这个大模型平台使用GPT-IMAGE-2生一个图才要几分钱,比 runninghub 便宜多了, 支持API,适合企业批量作图~


## 节点入口（功能相同）

在 ComfyUI 中添加 **`Xin OpenAI - Image/Chat`**（内部注册名：`OpenAPIImage` / `OpenAPIVisionChat`，任选其一即可）。

---

## 实例说明（对应上方示例图）

上图工作流为 **`api_mode` = `images`**：左侧两张 **Load Image** 分别接入 **`image1`**（杯子）与 **`image2`**（人物），**prompt** 为「图2的角色，拿着图1的豆浆杯」，经 **`images/edits`** 多图编辑后，右侧 **Preview Image** 为合成结果（截图中网关为 `codeagent.cloud`，模型 `gpt-image-2`，以你实际服务商为准）。

可将 [`example/workflow.json`](example/workflow.json) 拖入 ComfyUI 或从菜单导入以复现该工作流。

---

## 两种工作模式（`api_mode`）

| 模式 | 用途 | 典型模型示例 |
|------|------|----------------|
| **`images`** | 调用图像接口：**无参考图**走文生图（`images/generations`）；**有参考图**走图像编辑（`images/edits`，支持按顺序上传多张） | 各服务商的图像模型名（如 `gpt-image-2`，以网关文档为准） |
| **`chat_completions`** | 调用对话接口：支持 **文本 + 多张参考图** 的多模态输入，输出以文本为主 | 需填支持 **vision / 多模态** 的对话模型；**不要**填仅能用于 `images` 的绘图专用模型名 |

---

## 输入说明（按功能）

### 必选

- **`prompt`**：文本提示；绘图时为画面描述，对话时为对用户问题的说明（可与参考图一起发送）。
- **`base_url`**：服务商 OpenAI 兼容 API 根路径（通常形如 `https://xxx/v1`）。仅填域名时，对部分常见域名会自动补全 `/v1`。
- **`api_key`**：密钥；若留空，会尝试读取环境变量 **`OPENAI_API_KEY`** 或 **`ARK_API_KEY`**。
- **`model`**：由服务商提供的模型 ID（必须与所选 `api_mode` 匹配）。
- **`size`**：生成尺寸档位（映射为接口里的 `size`，含自动与各常用分辨率）；仅在 **`images`** 模式下参与绘图请求。

### 可选参考图（`image1`～`image10`）

| 场景 | 行为要点 |
|------|-----------|
| **`images` + 至少一张图** | 进入 **图像编辑**：按 **image1→image10** 顺序把已连接的图一并作为编辑输入（具体是否支持多图合成取决于服务商与模型）。 |
| **`images` + 不接图** | **文生图**：仅根据 `prompt` 与 `size` / `quality` 生成新图。 |
| **`chat_completions`** | 每张已连接的图都会作为用户消息中的图像内容发送；**至少需要**「非空 `prompt`」或「至少一张图」，否则会报错提示。 |

### 可选 **`quality`**

仅在 **`images`** 绘图相关请求中生效；选项与接口字段对应关系由界面文案标明（自动档通常表示 **不传 `quality`**，交给服务端默认）。

---

## 四个输出（顺序固定）

| 输出名 | 功能说明 |
|--------|-----------|
| **`image`** | **图**：`images` 成功时为生成/编辑结果；`chat` 模式下若有参考图则大致透传首张参考图占位，否则为空白占位图。 |
| **`text`** | **短文本**：如图像接口返回的修订说明、或对话里的回复正文摘要（具体取决于解析逻辑）。 |
| **`request`** | **请求预览**：等价 **curl / JSON** 的调试文本（密钥会被占位，大体积 base64 会截断），便于核对真实请求长什么样。 |
| **`response`** | **原始响应**：接口返回内容的字符串化结果；若执行失败，此处可能包含错误信息与 traceback，便于排查。 |

出错时节点仍会尽量填满四路输出：**`request`** 多为本次构造的请求预览，便于对照网关返回。

---

## 使用提示

1. **网关与模型必须匹配**：同一 `base_url` 下，绘图模型与对话模型名称通常不同，填错会报 4xx/业务错误。
2. **`chat_completions` 不要用纯绘图模型**：节点内会对明显不匹配的模型名给出提示。
3. **多图合成类需求**：优先确认服务商是否支持 **`images/edits` 多图**；否则可改用 **`chat_completions` + 支持多图的多模态模型**做理解与生成链路（取决于你的下游工作流）。
4. **依赖**：需已安装 `openai`、`torch`、`Pillow`、`numpy`（与常规 ComfyUI 环境一致）。

---

## 安装

将整个文件夹放入 ComfyUI 的 `custom_nodes` 目录，重启 ComfyUI。加载后在 **`ComfyUI-Xin-Openai`** 分类下选用 **`Xin OpenAI - Image/Chat`**。
