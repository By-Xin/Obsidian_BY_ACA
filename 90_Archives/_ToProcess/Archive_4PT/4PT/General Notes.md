---

---

**0926**
* **Prompt 固化“判定要点清单”并引用关键页**: 把Codebook 的核心表述压缩成 8–10 行加入系统前缀，明确两维度定义与映射关系，并强调禁止编造证据/页码。重点引用：Table 1（p.3）、Table 2（p.5–6）、Coding Challenges（p.27–29）、Appendix B（p.39–42）。
* **清理整理codebook**: 转为 md 格式, 考虑要不要重新整理顺序等, 考虑 school 这一板块是否保留. 
* **Schema 部分**: 后处理与聚合困难；也无法据 Q3/Q7 的两维度**自动映射类型**（而是直接询问 Q11）。建议改为**强制 JSON Schema 输出**，并**程序化映射**，避免模型在 Q11“越权综合”。（你现在的 `compose_prompt` 没有 JSON schema 限制。）
* 修正**用量统计字段**
* 路径模式这里该用两套系统

---
- Codebook.pdf 本身目前在 google ai lab 显示的 token 数目为 78175. 粗略算80k / 0.08 Million
- 就算输出多说20k (一般不会,不过考虑到要比如原文给出 evidence)
- 一个英文单词平均占 1.5 个 token
