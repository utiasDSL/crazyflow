import pytest
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"
example_scripts = list(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("example_script", example_scripts)
@pytest.mark.timeout(10) 
def test_example_main(example_script):
    """
    Dynamically import and execute the main function from an example script.
    """
    # Add the examples directory to sys.path to resolve imports
    sys.path.insert(0, str(EXAMPLES_DIR))

    # Dynamically import the module
    spec = importlib.util.spec_from_file_location(
        "example_module", example_script
    )
    example_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example_module)

    # Ensure the script has a main function
    assert hasattr(
        example_module, "main"
    ), f"{example_script.name} has no main() function."

    # Run the main function
    with patch("crazyflow.sim.core.Sim.render", return_value=None): # remove render function to enable headless testing
        example_module.main()
        
    # Clean up sys.path
    sys.path.remove(str(EXAMPLES_DIR))
