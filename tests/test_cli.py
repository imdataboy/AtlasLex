from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from atlaslex import cli
from atlaslex.config import Settings
from atlaslex.generation import NO_EVIDENCE, GeminiError, QUOTA_MESSAGE


def inputs(monkeypatch, answers):
    answers = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))


def test_menu_loops_on_invalid_selection_and_exits(manager, monkeypatch, capsys):
    inputs(monkeypatch, ["bad", "5"])
    cli.run(manager.settings, manager)
    output = capsys.readouterr().out
    assert output.count("MAIN MENU") == 2
    assert "Goodbye!" in output


def test_key_configuration_is_private_and_persistent(tmp_path, monkeypatch, capsys):
    settings = Settings(home=tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("getpass.getpass", lambda _: "test-placeholder-key")
    cli.configure_key(settings)
    assert Settings.load(tmp_path).api_key == "test-placeholder-key"
    assert settings.env_path.stat().st_mode & 0o777 == 0o600
    assert "test-placeholder-key" not in capsys.readouterr().out
    settings.save_api_key("updated-placeholder-key")
    assert Settings.load(tmp_path).api_key == "updated-placeholder-key"
    assert "test-placeholder-key" not in settings.env_path.read_text()


def test_key_rejects_blank_or_newline(tmp_path):
    settings = Settings(home=tmp_path)
    for key in ["", "key\nOTHER=value", "with space"]:
        with pytest.raises(ValueError):
            settings.save_api_key(key)


def test_missing_key_and_index(manager, capsys):
    cli.chat(manager)
    assert "Option 1" in capsys.readouterr().out
    manager.settings.api_key = "placeholder"
    cli.chat(manager)
    assert "No legal documents are indexed" in capsys.readouterr().out


def test_missing_qdrant_does_not_crash_status(manager, monkeypatch, capsys):
    monkeypatch.setattr(manager, "load", Mock(side_effect=ConnectionError()))
    cli.show_status(manager, models=False)
    assert "Qdrant: Unavailable" in capsys.readouterr().out


def test_chat_continues_after_quota_error(manager, monkeypatch, capsys):
    manager.settings.api_key = "placeholder"
    inputs(monkeypatch, ["", "first question", "second question", "ExIt"])
    answers = iter([GeminiError(QUOTA_MESSAGE), {"answer": "Réponse vérifiée.", "sources": []}])

    def query(*args, **kwargs):
        result = next(answers)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(cli, "rag_query", query)
    gemini = Mock()
    cli.chat(
        manager,
        gemini_factory=lambda _: gemini,
        retriever_factory=lambda _: SimpleNamespace(chunks=[{}]),
    )
    output = capsys.readouterr().out
    assert QUOTA_MESSAGE in output
    assert "Réponse vérifiée" in output
    assert "Returning to main menu" in output
    gemini.close.assert_called_once()


def test_return_from_chat_restores_menu(manager, monkeypatch, capsys):
    inputs(monkeypatch, ["3", "5"])
    monkeypatch.setattr(cli, "chat", lambda _: None)
    cli.run(manager.settings, manager)
    assert capsys.readouterr().out.count("MAIN MENU") == 2


def test_chat_explains_corpus_scope_and_insufficient_evidence(manager, monkeypatch, capsys):
    manager.settings.api_key = "placeholder"
    inputs(monkeypatch, ["question", "exit"])
    monkeypatch.setattr(cli, "rag_query", lambda *_: {"answer": NO_EVIDENCE, "sources": []})
    cli.chat(
        manager,
        gemini_factory=lambda _: Mock(),
        retriever_factory=lambda _: SimpleNamespace(
            chunks=[{"source_document": "cdc.pdf"}, {"source_document": "cdc.pdf"}]
        ),
    )
    output = capsys.readouterr().out
    assert "Indexed documents: cdc.pdf" in output
    assert output.count("cdc.pdf") == 1
    assert NO_EVIDENCE in output
    assert "more specific question" in output
    assert "Option 2" in output


def test_rebuild_requires_confirmation(manager, monkeypatch):
    inputs(monkeypatch, ["3", "n", "4"])
    manager.index = Mock()
    cli.indexing_menu(manager)
    manager.index.assert_not_called()


def test_keyboard_interrupt_is_clean(manager, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", Mock(side_effect=KeyboardInterrupt))
    cli.run(manager.settings, manager)
    assert "Goodbye!" in capsys.readouterr().out


def test_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings.load()
    assert settings.raw_dir.is_absolute()
    assert (settings.home / "pyproject.toml").exists()
