from __future__ import annotations

from pathlib import Path

from agent_black_box.demo_catalog import load_demo_catalog


def test_demo_catalog_entries_point_to_real_fixtures() -> None:
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures"

    demos = load_demo_catalog()

    assert [demo.demo_id for demo in demos] == ["hello-world-static"]
    for demo in demos:
        fixture_dir = fixture_root / demo.fixture_name
        assert fixture_dir.exists()
        assert (fixture_dir / "TASK.md").exists()
        assert (fixture_dir / "package.json").exists()
        assert (fixture_dir / "index.html").exists()
        assert (fixture_dir / "index.css").exists()
        assert (fixture_dir / "index.js").exists()
        assert (fixture_dir / "vite.config.js").exists()


def test_sample_fixture_does_not_overwrite_html_on_load() -> None:
    fixture_dir = Path(__file__).resolve().parents[1] / "fixtures" / "sample_frontend_task"

    html = (fixture_dir / "index.html").read_text(encoding="utf-8")
    script = (fixture_dir / "index.js").read_text(encoding="utf-8")

    assert "<h1>Hello World</h1>" in html
    assert "innerHTML" not in script
