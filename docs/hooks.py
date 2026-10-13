"""Expand ``--8<-- example "<path>"`` lines into the script's docstring and collapsible source."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

EXAMPLE = re.compile(r'^--8<-- example "(?P<path>[^"]+)"$', re.MULTILINE)


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    """Replace every example directive on the page."""
    root = Path(config.config_file_path).parent
    return EXAMPLE.sub(lambda match: _example(root, match["path"]), markdown)


def _example(root: Path, path: str) -> str:
    source = (root / path).read_text()
    docstring = ast.get_docstring(ast.parse(source))
    if not docstring:
        raise ValueError(f"{path} has no module docstring")
    code = "\n".join(f"    {line}" if line else "" for line in source.splitlines())
    return f'{docstring}\n\n??? example "{path}"\n\n    ```python\n{code}\n    ```\n'
