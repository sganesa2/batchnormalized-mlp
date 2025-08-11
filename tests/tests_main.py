import pytest
from ignore_file import main

@pytest.mark.paremterize(
    "inp, expected_value",
    [
        (1,2),
        (2,3),
        (3,4)
    ]
)
def test_main(inp:int, expected_value:int)->None:
    assert main(inp)==expected_value