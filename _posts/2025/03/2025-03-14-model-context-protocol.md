---
title: 深入解析型上下文协议 MCP (Model Context Protocol)
author: dreambt
date: 2025-03-14 23:50:00 +0800
categories: [LLM]
tags: [MCP, LLM, 大模型]
description: 深入解析模型上下文协议 MCP (Model Context Protocol)，包含核心概念、使用方法及架构解析
image: /assets/2025/03/mcp0.webp
---

## 1. 什么是 MCP？

MCP，全称 Model Context Protocol（模型上下文协议），源于 2024 年 11 月 25 日 Anthropic 发布的文章：[Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)。它定义了应用程序与 AI 模型之间交换上下文信息的方式，**旨在实现 AI 模型与外部世界的无缝连接。**

**简单来说，MCP 就像 AI 模型的“通用接口”**。开发者可以使用 MCP 将各种数据源、工具和功能连接到 AI 模型，就像 USB-C 接口统一了不同设备的连接方式一样。MCP 的目标是创建一个通用标准，从而简化 AI 应用程序的开发和集成，促进 AI 生态系统的蓬勃发展。

![mcp 简单理解](/assets/2025/03/mcp2.jpeg)

> 提示：为了提升阅读体验，本文将 MCP Host / Client / Server 的定义延后。初学者/用户可以暂时忽略这些概念，这并不影响对 MCP 的理解和使用。
{: .prompt-info }

## 2. 为什么需要 MCP？

MCP 的出现，我认为是 prompt engineering（提示工程）发展的必然趋势。更结构化、更丰富的上下文信息，能够显著提升 AI 模型的性能。在构建 prompt 的过程中，我们常常希望能够提供更具体、更贴近实际场景的信息，例如本地文件内容、数据库数据，甚至是实时的网络信息，从而帮助模型更好地理解和解决问题。

**试想一下，在 MCP 出现之前，我们通常会怎么做？** 开发者可能需要手动从数据库中筛选数据，或者使用专门的工具检索所需信息，然后将这些信息手动复制粘贴到 prompt 中。 随着待解决问题的复杂性不断提升，这种**“手动拼接”**信息的方式，无疑会变得越来越低效，难以满足实际需求。

为了克服手动构建 prompt 的局限性，许多 LLM 平台（如 OpenAI 和 Google）引入了 `function call` 功能。 这种机制允许模型在需要时调用预定义的函数来获取数据或执行操作，从而在一定程度上提升了自动化水平。

然而，`function call` 也存在一些固有的局限性（我对 function call 与 MCP 的理解可能不够全面，欢迎大家补充和探讨）。我认为，其主要问题在于 **`function call` 的平台依赖性**。 不同的 LLM 平台在 `function call` API 的实现上存在显著差异。例如，OpenAI 的函数调用方式与 Google 的并不兼容，这导致开发者在切换模型或平台时，需要重写大量代码，增加了适配成本。此外，`function call` 在安全性、交互性等方面也存在一些挑战。

![mcp 简单理解](/assets/2025/03/mcp1.jpeg)

**数据和工具本身是客观存在的**，我们真正期望的是，将它们连接到 AI 模型的环节能够更加智能、统一和便捷。Anthropic 基于这一痛点，设计了 MCP。它就像 AI 模型的“万能转接头”，让 LLM 能够轻松获取数据、调用工具，从而扩展其能力边界。 具体来说，MCP 的优势体现在以下几个方面：

*   **生态** - MCP 提供了丰富的现成插件（MCP Servers），可供 AI 模型直接调用，极大地拓展了模型的功能。
*   **统一性** - MCP 不受限于特定的 AI 模型。只要 AI 模型支持 MCP 协议，就可以灵活地切换和使用不同的 MCP Servers，避免了平台锁定。
*   **数据安全** - 用户的敏感数据保留在本地，无需全部上传到云端。（因为我们可以自行设计接口，决定传输哪些数据，实现精细化的数据控制）

## 3. 用户如何使用 MCP？

对于普通用户而言，通常并不需要深入了解 MCP 的具体实现细节，而是关注如何更便捷地使用 MCP 提供的强大功能。

关于 MCP 的具体使用方法，可以参考官方文档：[For Claude Desktop Users](https://modelcontextprotocol.io/quickstart/user)。 这里不再赘述。配置成功后，你可以在 Claude 中进行测试，例如： `Can you write a poem and save it to my desktop?` Claude 会请求你的权限，然后在本地新建一个文件。

此外，开发者们也已经构建了大量现成的 MCP Servers，用户只需选择适合自己需求的工具，然后进行配置和接入即可。

* [Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers)
* [MCP Servers Website](https://mcpservers.org/)
* [Official MCP Servers](https://github.com/modelcontextprotocol/servers)

例如，官方提供的 `filesystem` 工具，允许 Claude 读取和写入文件，就像在本地文件系统中操作一样。 这使得 AI 模型能够直接访问和处理本地文件，极大地扩展了其应用场景。

## 4. MCP 架构解析

下面，我们通过官方给出的架构图，来深入理解 MCP 的核心组件和工作原理。

![MCP 架构图](/assets/2025/03/mcp3.png)

MCP 架构由三个关键组件构成：Host、Client 和 Server。为了更好地理解它们之间的协作关系，让我们通过一个实际的场景来进行分析：

假设你正在使用 Claude Desktop (Host) 询问："我桌面上有哪些文档？"

1.  **Host**：Claude Desktop 作为 Host，负责接收用户的提问，并与 Claude 模型进行交互。 Host 充当着用户与 AI 模型之间的桥梁。
2.  **Client**：当 Claude 模型判断需要访问你的文件系统时，Host 中内置的 MCP Client 会被激活。这个 Client 的职责是与适当的 MCP Server 建立连接，以便获取所需的文件信息。
3.  **Server**：在这个例子中，文件系统 MCP Server 会被调用。它负责执行实际的文件扫描操作，访问你的桌面目录，并返回找到的文档列表。 Server 提供了访问外部资源的接口，例如文件系统、数据库等。

整个流程可以概括如下：用户提问 → Claude Desktop(Host) 接收提问 → Claude 模型分析问题，判断需要文件信息 → 激活 MCP Client → Client 连接到文件系统 MCP Server → 文件系统 MCP Server 执行文件扫描操作 → Server 将扫描结果返回给 Client → Client 将结果传递给 Claude 模型 → Claude 模型生成回答 → 最终结果显示在 Claude Desktop 上。

这种架构设计使得 Claude 可以在不同场景下灵活调用各种工具和数据源，极大地拓展了其应用范围。更重要的是，开发者只需专注于开发对应的 MCP Server，而无需关心 Host 和 Client 的具体实现细节，从而降低了开发门槛，提高了开发效率。

## 5. 原理：模型如何决定使用哪些工具？

在深入学习 MCP 的过程中，一个核心问题引起了我的好奇：**Claude（模型）是如何智能地决定使用哪些工具的呢？** 换句话说，模型是如何判断在特定情境下，应该调用哪些 MCP Server 来辅助解决问题的？

幸运的是，Anthropic 官方为我们提供了详细的[解释](https://modelcontextprotocol.io/quickstart/server#what’s-happening-under-the-hood)：

当用户提出一个问题时，MCP 的工作流程如下：

1.  **客户端（例如 Claude Desktop 或 Cursor）将用户的问题发送给 Claude 模型。**
2.  **Claude 模型会分析用户的问题，并评估可用的工具（即已连接的 MCP Servers）。** 模型会基于其对问题的理解，以及对各个工具功能的认知，来决定是否需要调用特定的工具。
3.  **客户端通过 MCP Server 执行所选的工具。** 这一步涉及了 Client 与 Server 之间的交互，以及 Server 实际执行任务的过程。
4.  **工具的执行结果被送回给 Claude 模型。** Server 执行完毕后，会将结果返回给 Claude 模型，作为其生成回答的依据。
5.  **Claude 模型结合执行结果，构造最终的 prompt 并生成自然语言的回应。** 模型会综合考虑用户的问题、可用的工具以及工具的执行结果，来生成最终的回答。
6.  **最终的回应展示给用户。**

> MCP Server 是由 Claude 模型主动选择并调用的。 值得深入思考的是，Claude 究竟是如何确定该使用哪些工具的呢？ 此外，模型是否会“幻觉”地调用一些不存在的工具呢？ 这些都是值得我们持续关注和研究的问题。
{: .prompt-info }

## 6. 总结

MCP (Model Context Protocol) 代表了 AI 与外部工具和数据交互的标准化和智能化。 通过本文，我们可以了解到：

1.  **MCP 的核心**：它是一个统一的协议标准，使得 AI 模型能够以一致的方式连接各种数据源和工具，从而扩展其能力边界，如同 AI 世界的“通用接口”。
2.  **MCP 的价值**：MCP 解决了传统 `function call` 机制存在的平台依赖性问题，提供了更统一、开放、安全、灵活的工具调用机制，使用户和开发者都能从中受益。
3.  **MCP 的使用与开发**：对于普通用户，MCP 提供了丰富的现成工具，**用户无需了解任何技术细节即可轻松使用**；对于开发者，MCP 提供了清晰的架构和 SDK，降低了工具开发的难度，促进了生态发展。

尽管 MCP 仍处于发展初期，但其潜力是巨大的。基于统一标准构建的生态系统，将正向促进整个 AI 领域的发展，推动 AI 应用的创新和普及。

以上内容涵盖了 MCP 的基本概念、价值和使用方法。对于技术实现感兴趣的读者，以下**附录提供了简易的 MCP Server 开发实践**，帮助你更深入地理解 MCP 的工作原理，并快速上手构建自己的 MCP Server。

## Appendix A：MCP Server 开发实践

在了解 MCP 的核心组件之后，您可能会发现，对于绝大部分 AI 开发者来说，我们最需要关注的其实是 Server 的实现。因此，我将通过一个简单的示例，来介绍如何快速实现一个 MCP Server。

MCP servers 可以提供三种主要类型的功能：

*   Resources（资源）：类似文件的数据，可以被客户端读取（如 API 响应或文件内容）
*   Tools（工具）：可以被 LLM 调用的函数（需要用户批准）
*   Prompts（提示）：预先编写的模板，帮助用户完成特定任务

本教程将主要关注工具（Tools）的实现。

### A.I 使用 LLM 构建 MCP 的最佳实践

在开始之前，Anthropic 官方为我们提供了一个基于 LLM 的 MCP Server 的[最佳开发实践](https://modelcontextprotocol.io/tutorials/building-mcp-with-llms)，总结如下：

*   **引入领域知识 (Domain Knowledge)**：为 LLM 提供关于 MCP Server 开发的范例和资料，帮助其更好地理解任务。
    *   访问 [modelcontextprotocol.io/llms-full.txt](https://modelcontextprotocol.io/llms-full.txt) 并复制完整的文档文本。（实测这个太长了，可以忽略）
    *   导航到 MCP [TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) 或 [Python SDK](https://github.com/modelcontextprotocol/python-sdk) Github 项目中并复制相关内容。
    *   将这些内容作为 prompt 输入到你的 chat 对话中，为 LLM 提供上下文信息。
*   **清晰地描述你的需求**：向 LLM 明确说明你的 MCP Server 需要实现的功能。
    *   你的服务器会开放哪些资源？
    *   它会提供哪些工具？
    *   它应该给出哪些引导或建议？
    *   它需要跟哪些外部系统互动？

下面给出一个 example prompt:

```
... （这里是已经引入的 domain knowledge）

请你打造一个 MCP 服务器，它能够：
- 连接到我的 PostgreSQL 数据库
- 将表格结构作为资源开放出来
- 提供运行只读 SQL 查询的工具
- 包含常见数据分析任务的引导
```

由于篇幅限制，本节不再展开。 推荐大家直接阅读[官方文档](https://modelcontextprotocol.io/tutorials/building-mcp-with-llms)，以获取更详细的实践指导。

### A.II 手动实践

本节内容主要参考了官方文档：[Quick Start: For Server Developers](https://modelcontextprotocol.io/quickstart/server)。您可以选择直接跳过这部分内容，快速浏览或进行一个速读。

这里我准备了一个简单的示例，使用 Python 实现一个 MCP Server，用来**统计当前桌面上的 txt 文件数量和获取对应文件的名字**（您可以理解为，这个示例虽然简单，但它足够清晰，主要目的是为那些难以配置开发环境的读者提供一个快速的实践记录）。 以下实践均运行在我的 MacOS 系统上。

**Step1. 前置工作**

*   安装 Claude Desktop。
*   Python 3.10+ 环境
*   [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk) 1.2.0+

**Step2. 环境配置**

由于我使用的是官方推荐的配置：

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建项目目录
uv init txt_counter
cd txt_counter

# 设置 Python 3.10+ 环境
echo "3.11" > .python-version

# 创建虚拟环境并激活
uv venv
source .venv/bin/activate

# Install dependencies
uv add "mcp[cli]" httpx

# Create our server file
touch txt_counter.py
```

> **Question**: 什么是 `uv` 呢和 `conda` 比有什么区别？
> **Answer**: `uv` 是一个由 Rust 编写的超快速 (100x) Python 包管理器和环境管理工具，由 Astral 开发。它旨在替代 `pip` 和 `venv`， 专注于速度、简单性和现代 Python 工作流。

**Step3. 构造一个 prompt**

```
... （这里是已经引入的 domain knowledge）

请你打造一个 MCP 服务器，它能够：
- 功能：
    - 统计当前桌面上的 txt 文件数量
    - 获取对应文件的名字

要求：
- 不需要给出 prompt 和 resource 相关代码。
- 你可以假设我的桌面路径为 /Users/{username}/Desktop
```

*   Domain Knowledge 复制于 MCP Python SDK 的 [README 文件](https://raw.githubusercontent.com/modelcontextprotocol/python-sdk/main/README.md)

**Step4. 实现 MCP Server**

以下代码由 Claude 3.7 直接生成。 值得注意的是，由于我的需求足够简单，因此 Claude 能够直接生成可用的代码。 当您需要实现一个更复杂的 MCP Server 时，可能需要多轮的引导和调试才能得到最终的代码。

```python
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# 创建 MCP Server
mcp = FastMCP("桌面 TXT 文件统计器")

@mcp.tool()
def count_desktop_txt_files() -> int:
    """Count the number of .txt files on the desktop."""
    # Get the desktop path
    username = os.getenv("USER") or os.getenv("USERNAME")
    desktop_path = Path(f"/Users/{username}/Desktop")

    # Count .txt files
    txt_files = list(desktop_path.glob("*.txt"))
    return len(txt_files)

@mcp.tool()
def list_desktop_txt_files() -> str:
    """Get a list of all .txt filenames on the desktop."""
    # Get the desktop path
    username = os.getenv("USER") or os.getenv("USERNAME")
    desktop_path = Path(f"/Users/{username}/Desktop")

    # Get all .txt files
    txt_files = list(desktop_path.glob("*.txt"))

    # Return the filenames
    if not txt_files:
        return "No .txt files found on desktop."

    # Format the list of filenames
    file_list = "\n".join([f"- {file.name}" for file in txt_files])
    return f"Found {len(txt_files)} .txt files on desktop:\n{file_list}"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run()
```

这个任务非常简单，只需要调用基本的 `os` 模块和 `pathlib` 模块就可以完成。

**Step5. 测试 MCP Server**

```bash
$ mcp dev txt_counter.py
Starting MCP inspector...
Proxy server listening on port 3000

MCP Inspector is up and running at http://localhost:5173
```

之后，您可以在浏览器中打开给出的链接进行操作.

如果一切顺利，您应该能像我一样看到对应的输出（`Tool Result`）～

**Step6. 接入 Claude**

最后一步，就是将我们编写好的 MCP Server 接入到 Claude Desktop 中。 流程如下：

```bash
# 打开 claude_desktop_config.json (MacOS / Linux)
code ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

在配置文件中添加以下内容。 请注意，需要将 `/Users/{username}` 替换为您的实际用户名，以及其他路径替换为您实际的路径。

```json
{
  "mcpServers": {
    "txt_counter": {
      "command": "/Users/{username}/.local/bin/uv",
      "args": [
        "--directory",
        "/Users/{username}/work/mcp-learn/code-example-txt", // 你的项目路径（这里是我的）
        "run",
        "txt_counter.py" // 你的 MCP Server 文件名
      ]
    }
  }
}
```

*   `uv` 最好使用绝对路径，推荐使用 `which uv` 命令获取。

配置完成后，重启 Claude Desktop。 如果没有问题，您应该就能在 Claude 中看到您配置的 MCP Server 了。

**Step7. 实际使用**

接下来，我们通过一个简单的 prompt 进行实际测试：

```
能推测我当前桌面上 txt 文件名的含义吗？
```