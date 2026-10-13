# Third-party licenses

[中文](THIRD_PARTY_LICENSES.zh-CN.md)

MemoFlow Agent contains **no vendored, copied or modified third-party source code**. It depends on the
following packages, installed separately by your package manager (versions as verified at release
preparation; see each project for its full license text):

| Package | Used for | License |
| --- | --- | --- |
| pydantic, pydantic-core | schemas, validation | MIT |
| httpx | HTTP client for providers | BSD-3-Clause |
| httpcore, h11, anyio, idna, sniffio | httpx dependencies | BSD-3-Clause / MIT |
| certifi | CA bundle (httpx dependency) | MPL-2.0 (unmodified, used as a dependency) |
| typing-extensions, typing-inspection, annotated-types | typing support | PSF-2.0 / MIT |
| psycopg, psycopg-binary *(optional extra `postgres`)* | PostgreSQL store | LGPL-3.0-only (dynamically used library; not distributed with this package) |

The `LICENSE` file is the unmodified Apache License 2.0 text from the Apache Software Foundation.
