from drone_security.cli import main


def test_cli_demo_runs_offline(capsys):
    main(["demo", "--offline"])
    out = capsys.readouterr().out
    assert "Person loitering at main gate, 00:01." in out
    assert "Blue Ford F150 spotted at garage, 12:00." in out
    assert "Video summary" in out and "Agent Q&A" in out


def test_cli_query_and_ask(capsys, tmp_path):
    db = str(tmp_path / "i.db")
    main(["run", "--db", db, "--quiet", "--no-colour"])
    main(["query", "--db", db, "--object", "truck"])
    out = capsys.readouterr().out
    assert out.count("Blue Ford F150") >= 6
    main(["ask", "--offline", "what", "objects", "were", "in", "the", "video?"])
    assert "truck" in capsys.readouterr().out
