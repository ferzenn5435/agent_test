# 最小本地代码库分析 Agent

一个纯 Python 实现的本地代码库分析 agent。它通过命令行接收 repo 路径和用户问题，使用 LLM + 只读工具调用循环分析代码库。

## 功能

- 不使用 LangChain、LangGraph、LlamaIndex 等 agent 框架。
- 只提供只读工具，不写文件，不执行 shell 命令。
- 每一步要求模型输出严格 JSON。
- 最大循环步数为 8。
- 每次运行写入 `logs/run_YYYYMMDD_HHMMSS.json`。

## 工具

- `list_dir(path)`：列出 repo 内目录。
- `read_file(path)`：读取 repo 内 UTF-8 文本文件，单文件最大 20KB。
- `search_text(keyword)`：搜索 repo 内文本，跳过二进制文件、隐藏目录、`.git`、`node_modules`、`build`、`dist`。
- `finish(answer)`：输出最终答案。

所有路径都会限制在传入的 repo 根目录内部。

## 配置

LLM 调用封装在 `llm_client.py`，默认使用 OpenAI-compatible Chat Completions 接口。运行前设置环境变量：

```powershell
$env:LLM_BASE_URL = "https://api.openai.com/v1"
$env:LLM_API_KEY = "your_api_key"
$env:LLM_MODEL = "your_model"
```

如果你的供应商已经提供完整 endpoint，也可以传入：

```powershell
$env:LLM_BASE_URL = "https://example.com/v1/chat/completions"
```

## 运行

```powershell
python main.py E:\path\to\repo "这个项目的入口文件在哪里？"
```

程序会打印最终答案，并在 `logs/` 下保存完整运行日志。

## 模型输出协议

模型每一步必须只输出一个 JSON 对象：

```json
{
  "thought": "先查看项目根目录。",
  "tool": "list_dir",
  "args": {
    "path": "."
  }
}
```

完成时调用：

```json
{
  "thought": "已经得到足够信息。",
  "tool": "finish",
  "args": {
    "answer": "最终答案"
  }
}
```

## 替换 LLM 供应商

替换不同模型供应商时，优先只修改 `llm_client.py`，保持 `LlmClient.chat(messages)` 返回模型文本即可。
