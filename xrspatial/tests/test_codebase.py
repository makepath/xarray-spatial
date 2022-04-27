from subprocess import run


def test_flake8():
    cmd = ["flake8"]
    proc = run(cmd, capture_output=True)
    assert proc.returncode == 0, f"Flake8 issues:\n{proc.stdout.decode('utf-8')}"


def test_isort():
    cmd = ["isort", "--diff", "--check", "."]
    proc = run(cmd, capture_output=True)
    assert proc.returncode == 0, f"isort issues:\n{proc.stdout.decode('utf-8')}"
