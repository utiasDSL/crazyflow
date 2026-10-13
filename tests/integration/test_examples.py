import importlib.util
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import available_backends

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
DOCS_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "docs" / "examples"

# Examples that render splats. splax rasterizes with cuda, so these need splax and a GPU.
REQUIRES_SPLAX = (
    "rendering/splat_camera.py",
    "rendering/splat_depth.py",
    "rendering/splat_gradients.py",
    "rendering/splat_viewer.py",
)

requires_splax = pytest.mark.skipif(
    importlib.util.find_spec("splax") is None or "gpu" not in available_backends(),
    reason="requires splax and a CUDA GPU",
)

assert all((EXAMPLES_DIR / name).is_file() for name in REQUIRES_SPLAX), "stale REQUIRES_SPLAX entry"

example_scripts = []
for path in sorted(EXAMPLES_DIR.rglob("*.py")):
    marks = requires_splax if path.relative_to(EXAMPLES_DIR).as_posix() in REQUIRES_SPLAX else ()
    # Parametrize over strings so that pytest prints readable ids. The test converts them back.
    example_scripts.append(pytest.param(str(path), marks=marks))


@pytest.mark.parametrize("example_script", example_scripts)
@pytest.mark.timeout(60)
@pytest.mark.integration
def test_example_main(example_script: str):
    """Dynamically import and execute the main function from an example script."""
    # Add the examples directory to sys.path to resolve imports
    example_script = Path(example_script)
    sys.path.insert(0, str(EXAMPLES_DIR))

    # Dynamically import the module
    spec = importlib.util.spec_from_file_location("example_module", example_script)
    example_module = importlib.util.module_from_spec(spec)
    # dataclasses resolves string annotations through the module in sys.modules
    sys.modules[spec.name] = example_module
    spec.loader.exec_module(example_module)

    # Ensure the script has a main function
    assert hasattr(example_module, "main"), f"{example_script.name} has no main() function."

    # Remove render function to enable headless testing
    with patch("crazyflow.sim.sim.Sim.render", return_value=None):
        example_module.main()

    # Clean up sys.path
    sys.path.remove(str(EXAMPLES_DIR))
    del sys.modules[spec.name]


def test_examples_in_docs():
    """Every example script has a section in the example docs, and every section has a script."""
    directive = re.compile(r'^--8<-- example "examples/(.+)"$', re.MULTILINE)
    pages = DOCS_EXAMPLES_DIR.glob("*.md")
    documented = {path for page in pages for path in directive.findall(page.read_text())}
    scripts = {path.relative_to(EXAMPLES_DIR).as_posix() for path in EXAMPLES_DIR.rglob("*.py")}
    assert not scripts - documented, f"Missing from docs/examples: {sorted(scripts - documented)}"
    assert not documented - scripts, f"Stale in docs/examples: {sorted(documented - scripts)}"
