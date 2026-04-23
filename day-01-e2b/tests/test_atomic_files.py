from __future__ import annotations

from agent_black_box.atomic_files import write_bytes_atomic, write_text_atomic


def test_write_text_atomic_replaces_existing_file_without_temp_leftovers(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "status.json"
    target.write_text('{"state":"old"}\n', encoding="utf-8")

    write_text_atomic(target, '{"state":"new"}\n')

    assert target.read_text(encoding="utf-8") == '{"state":"new"}\n'
    assert list(tmp_path.glob(".status.json.*.tmp")) == []


def test_write_bytes_atomic_cleans_temp_file_when_replace_fails(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "existing-dir"
    target.mkdir()

    try:
        write_bytes_atomic(target, b"new")
    except OSError:
        pass
    else:
        raise AssertionError("expected replacing a directory with a file to fail")

    assert target.is_dir()
    assert list(tmp_path.glob(".existing-dir.*.tmp")) == []
