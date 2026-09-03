"""The repository index.

Almost every test here is about the thing it refuses to do. Resolving a name to
the wrong definition is worse than resolving nothing: the reviewer reads it,
believes it, and files a confident finding about code it never saw — and nothing
downstream can tell that happened.
"""

from __future__ import annotations

from pathlib import Path

from loupe.context import build_contexts, estimate
from loupe.index import (
    build,
    called_names,
    enabled,
    gather,
    imported_names,
    module_of,
    render,
    resolve,
)
from loupe.nodes._common import valid_lines
from loupe.schema import FileDiff, Hunk, ReviewRequest

VALIDATORS = '''\
"""Validation."""


def validate(payload):
    if not payload.get("id"):
        raise ValueError("id is required")
    return payload


class Cart:
    def total(self):
        return sum(i.price for i in self.items)
'''

# A second `validate`, so a bare-name lookup has two answers and must give none.
OTHER = '''\
def validate(x):
    return bool(x)
'''

HANDLERS = '''\
from app.validators import validate


def handle(payload):
    checked = validate(payload)
    return checked["id"]
'''


def make_repo(root: Path) -> None:
    (root / "app").mkdir(parents=True)
    (root / "lib").mkdir()
    (root / "app" / "__init__.py").write_text("")
    (root / "app" / "validators.py").write_text(VALIDATORS)
    (root / "app" / "handlers.py").write_text(HANDLERS)
    (root / "lib" / "other.py").write_text(OTHER)


def test_module_names_match_how_an_import_writes_them():
    assert module_of("app/validators.py") == "app.validators"
    assert module_of("app/__init__.py") == "app"


def test_definitions_are_found_including_methods(tmp_path):
    make_repo(tmp_path)
    index = build(tmp_path, max_files=100)

    assert index.files == 4
    assert {d.module for d in index.by_name["validate"]} == {"app.validators", "lib.other"}
    assert index.by_name["Cart"][0].kind == "class"
    assert index.by_name["total"][0].kind == "method"


def test_the_file_cap_is_recorded_not_hidden(tmp_path):
    make_repo(tmp_path)
    index = build(tmp_path, max_files=2)
    assert index.files == 2
    assert index.partial


def test_vendored_and_generated_files_are_not_indexed(tmp_path):
    make_repo(tmp_path)
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "third_party.py").write_text("def validate(a): pass\n")
    index = build(tmp_path, max_files=100)

    assert all("third_party" not in d.path for d in index.by_name["validate"])


def test_an_import_says_which_definition_is_meant(tmp_path):
    """Two functions are called `validate`. The import names one of them."""
    make_repo(tmp_path)
    index = build(tmp_path, max_files=100)
    imports = imported_names(HANDLERS, "app/handlers.py")

    defn = resolve("validate", imports, index, "app/handlers.py", visible={})
    assert defn is not None
    assert defn.path == "app/validators.py"


def test_a_name_imported_from_outside_the_repo_resolves_to_nothing(tmp_path):
    """The dangerous case. `from pydantic import validate` must not be answered
    with this repository's own `validate` just because the name matches."""
    make_repo(tmp_path)
    index = build(tmp_path, max_files=100)
    source = "from pydantic import validate\n\n\ndef handle(p):\n    return validate(p)\n"
    imports = imported_names(source, "app/handlers.py")

    assert resolve("validate", imports, index, "app/handlers.py", visible={}) is None


def test_a_bare_ambiguous_name_resolves_to_nothing(tmp_path):
    make_repo(tmp_path)
    index = build(tmp_path, max_files=100)

    assert resolve("validate", {}, index, "app/handlers.py", visible={}) is None
    # ...while a name only one file defines is safe to answer.
    assert resolve("handle", {}, index, "app/other.py", visible={}) is not None


def test_a_definition_already_on_screen_is_not_sent_again(tmp_path):
    make_repo(tmp_path)
    index = build(tmp_path, max_files=100)
    defn = index.by_name["validate"][0]
    imports = imported_names(HANDLERS, "app/handlers.py")
    shown = {defn.path: {defn.start}}

    assert resolve("validate", imports, index, "app/handlers.py", visible=shown) is None
    # Windowed contexts omit regions, and then it is worth including after all.
    other = {defn.path: {1, 2}}
    assert resolve("validate", imports, index, "app/handlers.py", visible=other) is not None


def test_relative_imports_resolve_within_the_package(tmp_path):
    make_repo(tmp_path)
    index = build(tmp_path, max_files=100)
    source = "from .validators import validate\n\n\ndef handle(p):\n    return validate(p)\n"
    imports = imported_names(source, "app/handlers.py")

    defn = resolve("validate", imports, index, "app/handlers.py", visible={})
    assert defn is not None and defn.path == "app/validators.py"


def test_only_calls_on_changed_lines_are_followed():
    source = "a = one()\nb = two()\nc = three()\n"
    assert called_names(source, {2}) == {"two": 1}
    assert called_names(source, {1, 3}) == {"one": 1, "three": 1}


def test_attribute_calls_are_followed_by_their_last_name():
    assert called_names("x = repo.save(item)\n", {1}) == {"save": 1}


def test_a_file_that_will_not_parse_contributes_nothing():
    assert called_names("def broken(:\n", {1}) == {}


def test_rendered_definitions_are_line_numbered_and_capped(tmp_path):
    make_repo(tmp_path)
    index = build(tmp_path, max_files=100)
    defn = next(d for d in index.by_name["validate"] if d.module == "app.validators")

    whole = render(defn, tmp_path, max_lines=40)
    assert whole.splitlines()[0].strip().startswith(f"{defn.start}|")
    assert "def validate(payload):" in whole

    clipped = render(defn, tmp_path, max_lines=1)
    assert "more line(s)" in clipped


def _request(root: Path, path: str, source: str) -> ReviewRequest:
    """A change to `path`, with every line of it counted as changed."""
    n = len(source.splitlines())
    return ReviewRequest(
        source="local",
        ref="test",
        repo_root=str(root),
        files=[
            FileDiff(
                path=path,
                change_type="modified",
                hunks=[Hunk(old_start=1, old_lines=n, new_start=1, new_lines=n, content="")],
                content_after=source,
            )
        ],
    )


def _gather(request, contexts, **kwargs):
    defaults = {
        "max_files": 100,
        "max_lines": 40,
        "token_budget": 10_000,
        "estimate": estimate,
        "visible_lines": valid_lines,
    }
    return gather(request, contexts, **{**defaults, **kwargs})


def test_a_called_definition_reaches_the_reviewer(tmp_path):
    make_repo(tmp_path)
    request = _request(tmp_path, "app/handlers.py", HANDLERS)
    contexts, _ = build_contexts(request)

    references, stats, _edges = _gather(request, contexts)

    assert [r.definition.path for r in references] == ["app/validators.py"]
    assert "id is required" in references[0].source
    assert stats.resolved == 1
    assert stats.files_indexed == 4


def test_an_unresolvable_call_is_counted_not_guessed(tmp_path):
    make_repo(tmp_path)
    source = "def handle(p):\n    return sorted(p, key=len)\n"
    request = _request(tmp_path, "app/handlers.py", source)
    contexts, _ = build_contexts(request)

    references, stats, _edges = _gather(request, contexts)

    assert references == []
    assert stats.unresolved >= 1


def test_a_credential_in_an_unchanged_file_is_redacted_before_sending(tmp_path):
    """The expansion reads files the change never touched. A key in one of them
    has not been put in front of the reviewer by anybody, and must not leave."""
    make_repo(tmp_path)
    (tmp_path / "app" / "validators.py").write_text(
        'def validate(payload):\n'
        '    api_key = "AKIAIOSFODNN7EXAMPLE"\n'
        '    return payload, api_key\n'
    )
    request = _request(tmp_path, "app/handlers.py", HANDLERS)
    contexts, _ = build_contexts(request)

    references, stats, _edges = _gather(request, contexts)

    assert stats.secrets == 1
    assert "AKIAIOSFODNN7EXAMPLE" not in references[0].source
    assert "REDACTED" in references[0].source


def test_blocking_on_secrets_leaves_the_definition_out_entirely(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "app" / "validators.py").write_text(
        'def validate(payload):\n    key = "AKIAIOSFODNN7EXAMPLE"\n    return key\n'
    )
    request = _request(tmp_path, "app/handlers.py", HANDLERS)
    contexts, _ = build_contexts(request)

    references, stats, _edges = _gather(request, contexts, on_secret="block")  # noqa: S106

    assert references == []
    assert stats.secrets == 1


def test_definitions_over_the_budget_are_reported_not_dropped_silently(tmp_path):
    make_repo(tmp_path)
    request = _request(tmp_path, "app/handlers.py", HANDLERS)
    contexts, _ = build_contexts(request)

    references, stats, _edges = _gather(request, contexts, token_budget=1)

    assert references == []
    assert stats.over_budget == 1


def test_a_language_without_a_parser_here_gets_no_expansion(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "app" / "handlers.ts").write_text("export const handle = (p) => validate(p);\n")
    request = _request(tmp_path, "app/handlers.ts", "export const handle = (p) => validate(p);\n")
    contexts, _ = build_contexts(request)

    references, stats, _edges = _gather(request, contexts)

    assert references == []
    assert stats.files_indexed == 0, "indexed a repository for a change it cannot parse"


def test_expansion_is_off_for_pull_requests_by_default():
    """A pull request's repository is not on this disk. `auto` means local only."""
    assert enabled("auto", "local")
    assert not enabled("auto", "github")
    assert enabled("on", "github")
    assert not enabled("off", "local")


def test_a_definition_the_reviewer_can_already_see_counts_as_shown(tmp_path):
    """Two files change together and one calls the other. Its definition is
    deliberately not injected — the reviewer is already reading that file — but
    it was reaching them, and the edge said the opposite."""
    make_repo(tmp_path)
    both = "from app.validators import validate\n\n\ndef handle(p):\n    return validate(p)\n"
    request = ReviewRequest(
        source="local", ref="test", repo_root=str(tmp_path),
        files=[
            FileDiff(path="app/handlers.py", change_type="modified", content_after=both,
                     hunks=[Hunk(old_start=1, old_lines=5, new_start=1, new_lines=5, content="")]),
            FileDiff(path="app/validators.py", change_type="modified", content_after=VALIDATORS,
                     hunks=[Hunk(old_start=1, old_lines=1, new_start=1, new_lines=1, content="")]),
        ],
    )
    contexts, _ = build_contexts(request)

    references, _stats, edges = _gather(request, contexts)

    edge = next(e for e in edges if e.name == "validate")
    assert edge.definition.path == "app/validators.py"
    assert not any(r.definition.path == "app/validators.py" for r in references), (
        "no need to send source the reviewer is already reading"
    )
    assert edge.shown, "the reviewer could read it; the edge said they could not"


def test_a_file_is_parsed_once_not_once_per_question(monkeypatch):
    """Callers and imports are both read from the same tree."""
    import ast as ast_mod

    import loupe.index as index_mod

    calls = []
    real = ast_mod.parse

    def counted(src, *a, **k):
        calls.append(1)
        return real(src)

    monkeypatch.setattr(index_mod.ast, "parse", counted)

    source = "from json import dumps\n\n\ndef go(x):\n    return dumps(x)\n"
    tree = index_mod.parse(source)
    index_mod.imported_names(source, "a.py", tree)
    index_mod.called_names(source, {5}, tree)

    assert len(calls) == 1
