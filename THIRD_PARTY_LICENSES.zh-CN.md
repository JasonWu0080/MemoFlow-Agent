# 第三方许可证

[English](THIRD_PARTY_LICENSES.md)

MemoFlow Agent **不包含任何内置、复制或修改的第三方源代码**。它依赖下列包，由你的包管理器单独安装（版本为发布准备时核实的版本；完整许可证文本见各项目）：

| 包 | 用途 | 许可证 |
| --- | --- | --- |
| pydantic、pydantic-core | schema、校验 | MIT |
| httpx | Provider 的 HTTP 客户端 | BSD-3-Clause |
| httpcore、h11、anyio、idna、sniffio | httpx 的依赖 | BSD-3-Clause / MIT |
| certifi | CA 证书包（httpx 的依赖） | MPL-2.0（未修改，作为依赖使用） |
| typing-extensions、typing-inspection、annotated-types | 类型支持 | PSF-2.0 / MIT |
| psycopg、psycopg-binary *（可选组 `postgres`）* | PostgreSQL 存储 | LGPL-3.0-only（动态使用的库；不随本包分发） |

`LICENSE` 文件是 Apache 软件基金会发布的 Apache License 2.0 原文，未作修改。
