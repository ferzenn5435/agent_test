# 最小本地代码库分析 Agent

一个纯 Python 实现的本地代码库分析 agent。它通过命令行接收 repo 路径和用户问题，使用 LLM + 安全工具调用循环分析代码库，并可生成不写入文件的补丁提案。

## 功能

- 不使用 LangChain、LangGraph、LlamaIndex 等 agent 框架。
- 只提供安全工具，不写文件，不执行 shell 命令。
- 每一步要求模型输出严格 JSON。
- 最大循环步数为 8。
- 每次运行写入 `logs/run_YYYYMMDD_HHMMSS.json`。

## 工具

- `list_dir(path)`：列出 repo 内目录。
- `read_file(path)`：读取 repo 内 UTF-8 文本文件，单文件最大 20KB，返回内容会带行号，例如 `12 | code here`。
- `search_text(keyword)`：搜索 repo 内文本，返回文件名、行号和上下文行，跳过二进制文件、隐藏目录、`.git`、`node_modules`、`build`、`dist`。
- `propose_patch(file_path, plan, replacements)`：为 repo 内单个文件生成补丁提案和 unified diff，不修改文件。参数包含 `file_path`、`plan`、`replacements`，其中 `replacements` 的每一项包含 `old_text` 和 `new_text`。
- `finish(answer)`：输出最终答案，答案应尽量引用文件名、函数名和行号。

所有路径都会限制在传入的 repo 根目录内部。`propose_patch` 只返回建议的修改计划和 unified diff，是否在本 agent 外部应用该 diff，由用户自行决定。

## 配置

LLM 调用封装在 `llm_client.py`，默认使用 OpenAI-compatible Chat Completions 接口。

推荐复制 `.env.example` 为 `.env`，然后在 `.env` 中填写本地配置：

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_api_key
LLM_MODEL=your_model
```

`.env` 只保存在本地，已被 `.gitignore` 忽略，不会提交密钥。系统环境变量优先级更高；如果系统环境变量和 `.env` 同时存在，程序会使用系统环境变量。

也可以直接设置系统环境变量：

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

位置参数方式：

```powershell
python main.py E:\path\to\repo "这个项目的入口文件在哪里？"
```

`--repo` 参数方式：

```powershell
python main.py --repo E:\path\to\repo "这个项目的入口文件在哪里？"
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

需要提出补丁时调用：

```json
{
  "thought": "已经确认目标文本，生成补丁提案供用户审查。",
  "tool": "propose_patch",
  "args": {
    "file_path": "prompts.py",
    "plan": "说明本次修改计划。",
    "replacements": [
      {
        "old_text": "原文本",
        "new_text": "新文本"
      }
    ]
  }
}
```

`propose_patch` 只提出 unified diff，不写入文件。最终答案应包含生成的修改计划和 unified diff，供用户审查后在本 agent 外部决定是否应用。

完成时调用：

```json
{
  "thought": "已经得到足够信息。",
  "tool": "finish",
  "args": {
    "answer": "最终答案，例如：入口逻辑在 main.py:25 的 main() 中。"
  }
}
```

## 替换 LLM 供应商

替换不同模型供应商时，优先只修改 `llm_client.py`，保持 `LlmClient.chat(messages)` 返回模型文本即可。
