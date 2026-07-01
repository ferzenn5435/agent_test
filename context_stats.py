"""上下文统计 schema。"""

from __future__ import annotations

from dataclasses import dataclass


def _normalize_posix_path(path: str) -> str:
    """将路径统一为 posix 风格用于跨平台比较。"""
    return path.replace("\\", "/")

@dataclass(frozen=True)
class ContextStats:
    """记录上下文管理相关的基础统计。

字段含义：
- `steps_used`：执行 tool/decision 的步骤计数
- `total_tool_output_chars`：工具输出累计字符数
- `messages_total_chars`：消息累计字符数
- `files_read`：已完整读取的文件相对路径集合
- `ranges_read`：读取区间记录（path/start_line/end_line）
- `search_calls`：search_text 调用次数
- `full_file_reads`：被标记为完整文件读取的路径集合
"""

    steps_used: int = 0
    total_tool_output_chars: int = 0
    messages_total_chars: int = 0
    files_read: tuple[str, ...] = ()
    ranges_read: tuple[dict[str, int | str], ...] = ()
    search_calls: int = 0
    full_file_reads: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """输出可序列化的稳定 JSON schema。

输出阶段会规范化路径为正斜杠，避免 Windows/Unix 平台差异影响评估。
"""

        normalized_ranges: list[dict[str, int | str]] = []
        for range_read in self.ranges_read:
            normalized_range = dict(range_read)
            path_value = normalized_range.get("path")
            if isinstance(path_value, str):
                normalized_range["path"] = _normalize_posix_path(path_value)
            normalized_ranges.append(normalized_range)

        return {
            "steps_used": self.steps_used,
            "total_tool_output_chars": self.total_tool_output_chars,
            "messages_total_chars": self.messages_total_chars,
            "files_read": [
                _normalize_posix_path(file_path)
                for file_path in self.files_read
            ],
            "ranges_read": normalized_ranges,
            "search_calls": self.search_calls,
            "full_file_reads": [
                _normalize_posix_path(file_path)
                for file_path in self.full_file_reads
            ],
        }
