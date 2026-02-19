**为什么**：Codebook 第 B/C 节强调**先判断文本是否属于“政策/治理分析”**，比如纯科学描述或新闻素材往往要标记为 `out_of_scope`（不入样）。[](https://drive.google.com/file/d/11z7FWIfVqs2YruJqa8uqOz8LkoI1k9Ot)

[4ptCodebook](https://drive.google.com/file/d/11z7FWIfVqs2YruJqa8uqOz8LkoI1k9Ot)

**怎么做**

- **Schema**（在 _“Set up prompt and schema”_ 那格）：给顶层加
    
    `"out_of_scope": {"type":"boolean"}, "out_of_scope_reason": {"type":"string"}`
    
- **Q_SET**（在 `compose_prompt()`）：在 Q1、Q2 前加 **Q0**：  
    “该文本是否属于我们要编码的政策/治理分析宇宙？若否，请给出1–2句理由与定位（页码）并将 `out_of_scope=true`。”
    
- **映射规则**：`out_of_scope=true` 则 `primary_type="undetermined"`，`confidence<=0.4`，并**不再映射 4 类**。
    

> 这与 p.27–29 “Coding Challenges / Application”的流程一致：先宇宙→再两维度→再映射。