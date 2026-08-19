# HANDOVER — Line-Ending (CRLF) Normalization 交接文档

> 生成时间：2026-08-17 · 作者：GitHub Copilot agent
> 用途：把"LF 归一化"任务的当前状态、已做动作、存在的问题完整交给下一位处理者。

---

## 1. 原始问题（用户报障）

- 工作区里 `git status` 显示 **73 个文件被修改**，增删行数均为 **19,193**（完全对称）。
- 根因：磁盘上的文件被重新保存为 **CRLF 换行符**，而 HEAD 里是 **LF** → git 把每个文件当成"全文件重写"。
- 真正的笔记内容**一行都没变**（用 `git diff --ignore-all-space` 验证为空）。
- 仓库无 `.gitattributes`、无 `core.autocrlf`/`core.eol` 配置，git 不做换行符归一化。
- 用户需求：**归一化为 LF 并添加 `.gitattributes`，以后不再出现假改动**。

---

## 2. 我执行过的动作（按顺序）

1. **创建 `.gitattributes`**（内容见下，已 `git add` 暂存，状态 `A`）：
   ```gitattributes
   # Normalize all text files to LF line endings in the repo
   # (prevents Obsidian/editors from creating fake "whole-file changed" diffs)
   * text=auto eol=lf

   # Explicitly treat binary formats as binary (never normalize)
   *.pdf binary
   *.png binary
   *.jpg binary
   *.jpeg binary
   *.gif binary
   *.webp binary
   *.bmp binary
   *.ico binary
   *.zip binary
   *.gz binary
   *.tar binary
   *.bin binary
   *.woff binary
   *.woff2 binary
   *.ttf binary
   *.otf binary
   ```

2. **`git add --renormalize .`** —— 把索引里所有文本文件按 LF 重新规范化。
   - 结果：**256 个文件进入暂存区（staged）**。
   - 这说明：HEAD 里**部分文件本身是 CRLF**（仓库历史上就是混合换行符），归一化会真正改掉这些文件的历史换行符。→ 这是一个真实改动（只影响换行符，不影响内容）。

3. **尝试把磁盘工作区文件转成 LF**（git 本身因文件"已干净"而不重写磁盘，`git restore`/`git checkout-index` 均不转换）：
   - 第一次 sed 循环**有 bug**（`$'\x00'` 在 bash 里是空字符串，导致 grep 条件恒假，**0 个文件被转换**）。
   - 改用**扩展名黑名单**的 sed 循环，把含 CR 的文本文件批量转 LF（跳过 .pdf/.png/.jpg 等二进制扩展名）。
   - 该循环转换了磁盘上的文本文件，但**未做转换后二次全量校验**。

4. **发现并定位问题**：
   - 归一化后出现 `236 个 MM`（暂存+未暂存），且有一个 `.md` 文件 `10_AtomicKnowledge/math.LA_LinearAlgebra/Rank_and_Eigenvalue.md` 被 git 判为二进制（`Bin 10632 -> 10541 bytes`）。
   - 根因：**这个 `.md` 文件在 HEAD 里就含 3263 个 NUL 字节**（伪装成 .md 的二进制/损坏文件，`text=auto` 把它当二进制，归一化时没动它）。但我的扩展名 sed 循环**误伤了它**：把它磁盘副本的 `\r` 剥掉了 → 字节数 10632 → 10541（少 91 字节）。

---

## 3. 当前仓库确切状态（已重新核实）

- **HEAD / index / worktree 三级对比（对文本文件）**：
  - `git diff --cached --ignore-space-at-eol` = **只有 `.gitattributes`（+21 行）**。
    → 所有 256 个已暂存修改**除 .gitattributes 外全是纯换行符变化（LF↔CRLF），零内容变化**。
  - `git diff --ignore-space-at-eol`（工作区 vs 索引）= **只有 `Rank_and_Eigenvalue.md`（Bin 10632 -> 10541）**。
    → 其余所有未暂存差异**全是纯换行符变化，零内容变化**。
- `git status --short` 汇总：`1 A`（.gitattributes）、`92 M`、`236 MM`（共 329 行）。
  - 实际构成：`A` 1 个；`M `（仅暂存）约 19 个；`MM`（暂存+未暂存）236 个；` M`（仅未暂存）73 个（即最初那 73 个文件）。
- **二进制文件安全**：所有含 NUL 的文件（.mkv/.pdf/.png/.jpg 等）在 `.gitattributes` 中均已标记 `binary`，**未被 sed 触碰**（已验证：唯一的意外改动只有那个 `.md`）。

### 安全结论（已验证）
- **没有丢失任何笔记内容**。除 `Rank_and_Eigenvalue.md` 外，所有文件的内容（忽略换行符）与 HEAD 完全一致。
- **唯一的损坏文件**：`10_AtomicKnowledge/math.LA_LinearAlgebra/Rank_and_Eigenvalue.md`
  - HEAD: 10632 字节，含 3263 个 NUL，CRLF
  - 当前 worktree: 10541 字节，CR 被剥掉 91 个
  - 恢复命令：`git restore --worktree "10_AtomicKnowledge/math.LA_LinearAlgebra/Rank_and_Eigenvalue.md"`（或 `git checkout -- <该文件>`）

---

## 4. 任务状态

- ❌ **未完成**：工作区仍有 309 行换行符"假改动"（`MM` 236 + ` M` 73），即用户想消除的假 diff 还在。
- ❌ **未提交**：没有任何 commit 产生；所有改动都在暂存区/工作区。
- ✅ `.gitattributes` 已创建并暂存。
- ✅ 索引已完成 LF 归一化（256 个文件 staged）。

---

## 5. 给下一位处理者的建议（按优先级）

1. **先恢复被误伤的文件**（立即，避免覆盖）：
   ```bash
   git restore --worktree "10_AtomicKnowledge/math.LA_LinearAlgebra/Rank_and_Eigenvalue.md"
   ```
2. **核实剩余差异全是换行符**（可随时用，应只显示该二进制 .md）：
   ```bash
   git diff --ignore-space-at-eol --stat
   git diff --cached --ignore-space-at-eol --stat
   ```
3. **完成磁盘 LF 转换**（关键：用 NUL 检测或 `git check-attr` 判断，**不要用扩展名白名单**，避免再误伤伪装成文本的二进制文件）：
   ```bash
   # 对每个文件：只有不含 NUL 且含 \r 时才 strip CR
   git ls-files -z | while IFS= read -r -d '' f; do
     if ! git show HEAD:"$f" 2>/dev/null | tr -cd '\0' | grep -q . \
        && grep -q $'\r' "$f" 2>/dev/null; then
       sed -i 's/\r$//' "$f"
     fi
   done
   # 注意：对每个文件改用 `git show HEAD` 判断是否有 NUL，避免再误伤
   ```
   （更稳妥的替代方案：直接丢弃所有未暂存改动 `git restore --worktree .`，然后让 git 按 `text=auto eol=lf` 在下一次 checkout/编辑时自动转换。此操作安全，因为已验证未暂存差异全是换行符。）
4. **最终校验**：`git status --short` 应只剩 `.gitattributes`（A）和预期的 staged 归一化（M）。
5. **提交**：建议 `git add . && git commit -m "chore: normalize line endings to LF and add .gitattributes"`（提交前与用户确认）。
6. **后续防护**：`.gitattributes` 的 `* text=auto eol=lf` 生效后，Obsidian/编辑器再保存 CRLF 也不会再产生假 diff。

---

## 6. 注意事项 / 已知坑

- **`text=auto` 与含 NUL 的文件**：`.md` 扩展名不代表是文本文件；判定标准是内容是否含 NUL。任何批量转换**必须**先做 NUL 检测（用 `tr -cd '\0'`，不要用 `grep -P '\x00'`——grep 遇二进制会特殊处理、计数不可靠）。
- **`git restore` 不重写"已干净"的文件**：git 认为索引=工作区时就跳过，不会应用 `eol=lf` 转换；需要强制时才用 checkout/sed。
- **`$'\x00'` 在 bash 中是空字符串**，不能用来检测 NUL（这是第一次 sed 循环失效的原因）。
- 仓库历史上就是 **LF 与 CRLF 混合**（HEAD 里部分文件是 CRLF），归一化必然会产生 256 个文件的 staged 换行符改动，属预期。
- `.gitignore` 也在被改动列表中（文本文件，正常参与归一化）。
