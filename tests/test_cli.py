from aieval.cli import main


def test_cli_help():
    assert main(["--help"]) == 0