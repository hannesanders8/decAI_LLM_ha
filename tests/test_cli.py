from typer.testing import CliRunner

from locassess.cli import app
from tests.conftest import CRITERIA_CSV

runner = CliRunner()


def test_init_db_and_load_criteria(tmp_path):
    db = tmp_path / "cli.db"
    r = runner.invoke(app, ["init-db", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert db.exists()
    for _ in range(2):  # idempotent
        r = runner.invoke(app, ["load-criteria", "--db", str(db), "--csv", str(CRITERIA_CSV)])
        assert r.exit_code == 0, r.output
        assert "26 main criteria, 94 sub-criteria" in r.output
