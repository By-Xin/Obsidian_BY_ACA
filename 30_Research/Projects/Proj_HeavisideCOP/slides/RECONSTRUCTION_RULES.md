# Beamer 复原规则

本文件记录当前已经确认的长期要求。后续根据课堂照片继续补充 slides 时，默认遵循这些规则。

## 项目与输出

- 使用英文 Beamer，画面比例为 16:9。
- 主文件为 `main.tex`，使用 XeLaTeX 编译。
- 输出文件为 `main.pdf`。
- 演示文稿标题为 `Heaviside Composite Optimization Problems`。
- 章节结构（用户确认）：
  - `Chapter I: Introduction` —— 原稿 1–9，已复原。
  - `Chapter II: Prerequisites, a second course in optimization` ——
    原稿 10–20，仅有分节页，正文待补照片。
  - `Chapter III: Paradigm changes -- Sources of discontinuity` ——
    原稿 21– ，21–28、36–37 已复原，29–35 待补照片。
- 分节页统一为 `[plain]` 居中两行：第一行 `Chapter N` 用 title 字号加粗，
  第二行为章节描述，降一档字号并去粗（`\mdseries\large`）。
- 使用用户提供的 SimplePlus 模板；模板来源为
  `/Users/xinby/Downloads/BEAMER__Copy_.zip`。
- 保留模板的版式、页码、颜色层级和整体气质，除非用户明确要求调整。

## 课程正文复原

- 以课堂照片中可见内容为课程正文的主要依据，优先忠实复原原始结构、文字、公式和强调关系。
- Slides 顺序以用户提供照片的先后为准，新页依次追加到末尾；不依据照片中的原始页码调整顺序。
- 数学符号必须完整、准确；区间端点、上下标、求和范围、不等号和函数参数均需逐项核对。
- 对照片中无法可靠辨认的内容，不以视觉近似替代数学判断；必要时明确标记并等待进一步照片。
- 原稿中明显的拼写和用词笔误直接订正，不必逐条征求同意；但每次必须在回复中列出改了什么。
  数学内容（符号、区间、上下标、系数）不适用此条，有疑问一律先问。
- Slides 的英文应自然、简洁，并保持适合课堂展示的文字密度。
- **不得自拟标题。** frame title 只能用照片上确实出现的文字；照片上没有标题的页面
  就不写 frametitle，留空即可，不要用章节名或自造的措辞去填补。

## 数学排版约定

- 向量和矩阵统一使用 `\mathbf{}`，例如 `\mathbf{x}`、`\mathbf{A}`、
  `\mathbf{A}_{i\bullet}`。
- 标量保持普通数学字体，例如 `s`、`t`、`a_j`、`x_j`、`b`、`K`、
  `\eta_i`。
- 标量函数可以作用于向量，例如 `f(\mathbf{x})`、`\phi(\mathbf{x})`；
  函数名称本身不加粗。
- 指示函数统一使用 `\mathds{1}`，并严格区分
  `[0,\infty)` 与 `(0,\infty)`。
- 稀疏计数符号使用 `|t|_0` 或 `|x_j|_0`；其中 `x_j` 是标量，不加粗。
- 公式优先使用自然字号和合理换行，不使用不必要的整体缩放。
- 使用 `\usefonttheme[onlymath]{serif}`：beamer 默认把数学重定向到 Computer
  Modern Sans，会让 `\mathbf` 显得过粗。正文保持模板的无衬线体，公式用衬线体。

## 字体与页面密度

- 保持正文、标题和公式的字号关系和谐。
- 避免公式过大；内容较密集时优先精简文字、调整间距或合理分行。
- 不以极小字号强行容纳内容，并保证投影环境下可读。

## 课程正文与补充内容的区分

- 课堂照片中原有内容使用模板的普通白色页面，不额外添加“补充”标签。
- 后续新增的一整页解释或可视化，应沿用其所补充的上一页课程正文的 frame title。
- 整页新增内容放入 `supplementarycallout` 环境，并使用明确的
  `Supplementary ...` 标题。
- 插入课程正文页中的新增说明使用 `supplementarynote` 环境。
- **不要演绎。** 复原照片（含板书）时只还原画面上确实有的公式、文字和标注，
  不添加引导语、连接句、结论或与其他 slides 的关联。用户口述要点时同理，
  只写他说的那一点。宁可短、宁可无标题，也不自行发挥。
- 补充内容使用低饱和度青灰色体系：
  - `SupplementAccent = RGB(90,119,126)`；
  - `SupplementTint = RGB(241,246,246)`。
- 不使用鲜艳红色标记补充内容。课程原稿本身已有的红色关键词或强调应保留。
- 蓝色分两个角色，不可混用：`DarkBlue` 只作模板结构色（frametitle、bullet
  标记，由主题自动套用）；正文里的一切蓝色强调统一用 `MediumBlue`。
- 后续所有新增注释、解释、图示和推论继续沿用这套视觉区分。

## 可视化规则

- 可视化只在能够明显帮助理解数学定义或差异时添加。
- 精确数学图形优先使用 TikZ/PGFPlots，使源文件可编辑且符号一致。
- 阶跃函数必须准确表示阈值处的取值：实心点表示包含，空心点表示不包含。
- 图中的向量、矩阵和标量仍遵循上述数学排版约定。

## 编译与验收

- 每次修改后运行：

  ```bash
  latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
  ```

- 对所有修改过的页面进行全尺寸渲染检查。
- 交付前确认没有以下问题：
  - `Overfull` 或 `Underfull`；
  - `Undefined control sequence`；
  - `Missing character`；
  - 公式、标题、callout 或页码重叠、裁切。
- 每轮工作均保存最新的 `main.tex` 和重新编译后的 `main.pdf`。

