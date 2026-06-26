# 最小本地代码库分析 Agent

一个纯 Python 实现的本地代码库分析 agent。它通过命令行接收 repo 路径和用户问题，使用 LLM + 安全工具调用循环分析代码库，并支持人工批准后的受控代码修改。

## 功能

- 不使用 LangChain、LangGraph、LlamaIndex、MCP 或多 agent 框架。
- 只提供安全工具，不执行任意 shell 命令，不允许未批准的文件修改。
- 每一步要求模型输出严格 JSON。
- 最大循环步数为 8。
- 每次运行写入 `logs/run_YYYYMMDD_HHMMSS.json`。
- v0.3 修改流程会把补丁保存到 `.repopilot/patches`，应用前展示补丁并等待用户确认，应用时在 `.repopilot/backups` 创建备份，在 `.repopilot/runs` 写入事件日志。

## 工具

- `list_dir(path)`：列出 repo 内目录。
- `read_file(path)`：读取 repo 内 UTF-8 文本文件，单文件最大 20KB，返回内容会带行号，例如 `12 | code here`。
- `read_file_range(path, start_line, end_line)`：读取 repo 内 UTF-8 文本文件的闭区间行范围。
- `search_text(keyword)`：搜索 repo 内文本，返回文件名、行号和上下文行，跳过二进制文件、隐藏目录、`.git`、`node_modules`、`build`、`dist`。
- `propose_patch(instruction, diff)`：保存 unified diff 补丁提案到 `.repopilot/patches`，返回 `patch_id`、预览和影响路径，不修改目标文件。
- `apply_patch(patch_id)`：应用已保存补丁。CLI 模式会先显示补丁路径、影响路径、风险提示和补丁预览，只有用户明确批准后才会修改文件，并会先写入 `.repopilot/backups` 备份。
- `run_tests(command_name)`：执行白名单测试命令，只支持 `unit` 和 `compile`。`unit` 对应 `python -m unittest discover`，`compile` 对应 `python -m compileall .`。
- `finish(answer)`：输出最终答案，答案应尽量引用文件名、函数名和行号。

所有路径都会限制在传入的 repo 根目录内部。工具不会执行任意 shell，不调用外部 patch 程序，也不会在没有人工批准的情况下应用修改。

## v0.3 安全修改流程

1. 先用 `read_file` 或 `read_file_range` 读取相关文件，确认要修改的上下文。
2. 调用 `propose_patch(instruction, diff)` 提交 unified diff，补丁保存到 `.repopilot/patches`。
3. CLI 展示补丁路径、影响路径、风险提示和补丁内容。
4. 用户输入批准词后才允许 `apply_patch(patch_id)` 修改文件。批准词只接受 `yes`、`y`、`approve`，输入会先执行 `strip().lower()`。
5. 空输入、`no` 或任意其他文本都会拒绝应用，agent 收到失败工具结果，目标文件不会被修改。
6. 应用补丁时会在 `.repopilot/backups` 创建备份，并在 `.repopilot/runs` 记录 apply 和确认事件。
7. 成功应用后应调用 `run_tests(command_name)`，可选命令为 `unit` 或 `compile`。

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

需要读取目标片段时调用：

```json
{
  "thought": "先读取待修改区域，确认上下文。",
  "tool": "read_file_range",
  "args": {
    "path": "prompts.py",
    "start_line": 1,
    "end_line": 80
  }
}
```

需要提出补丁时调用：

```json
{
  "thought": "已经确认目标文本，保存补丁提案供用户审查。",
  "tool": "propose_patch",
  "args": {
    "instruction": "更新 prompts.py 的安全流程说明。",
    "diff": "diff --git a/prompts.py b/prompts.py\n--- a/prompts.py\n+++ b/prompts.py\n@@ -1,1 +1,1 @@\n-old text\n+new text\n"
  }
}
```

用户批准后才能应用补丁：

```json
{
  "thought": "用户已经在 CLI 中批准补丁，开始应用保存的补丁。",
  "tool": "apply_patch",
  "args": {
    "patch_id": "20260618_100000_abcd1234ef56"
  }
}
```

应用成功后运行白名单测试：

```json
{
  "thought": "补丁已经应用，运行单元测试验证。",
  "tool": "run_tests",
  "args": {
    "command_name": "unit"
  }
}
```

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
