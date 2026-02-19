如下是我看到的一个best practice. 请你按照这个practice 优化我们的结构安排Generative AI 目录主要包含以下几个核心文件夹：
  •  config/：存放配置文件（如 YAML 格式的模型、提示词模板、日志等配置）。
  •  src/：源代码主目录，采用模块化组织。
  •  data/：数据相关目录，包括缓存、输出、嵌入等。
  •  examples/：示例脚本（如基础补全、聊天会话、链式提示等）。
  •  notebooks/：Jupyter Notebook 文件，用于实验、测试、提示词调优和模型分析。
  •  requirements.txt、README.md、Dockerfile 等标准项目文件。
  src/ 目录下的关键模块
  •  llm/：大语言模型相关实现（初始化、基础客户端、OpenAI/Groq 等客户端、工具调用等）。
  •  prompt_engineering/：提示词工程相关（提示词初始化、模板、few-shot 示例、测试等）。
  •  utils/：工具函数（日志、错误处理、速率限制、令牌计数、缓存等）。
  •  handlers/：输入/输出处理相关。
  最佳实践（Best Practices）
  1.  使用 YAML 文件管理配置。
  2.  正确实现提示词错误处理。
  3.  使用速率限制器保护 API 调用。
  4.  将模型客户端分离。
  5.  适当缓存结果。
  6.  使用 notebooks 进行实验和测试。
  快速开始（Getting Started）
  7.  克隆仓库。
  8.  安装依赖（requirements.txt）。
  9.  配置模型设置。
  10.  查看示例代码。
  11.  通过 notebooks 开始实验。
  开发建议（Development Tips）
  •  遵循模块化设计。
  •  编写组件测试。
  •  使用版本控制。
  •  持续更新依赖。
  •  监控 API 使用情况。
  这个项目模板强调配置与代码分离、模块化组织、实验优先（notebook驱动）和可维护性，非常适合用于快速原型开发、提示词工程实验以及生产级生成式AI应用的搭建。