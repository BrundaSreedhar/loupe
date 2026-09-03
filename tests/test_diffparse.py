import pytest

from loupe.adapters.diffparse import added_line_numbers, parse_unified_diff

DIFF = """\
diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -10,6 +10,7 @@ def handler(req):
     user = lookup(req.user_id)
     if user is None:
         return None
-    return user.name
+    if not user.active:
+        return None
+    return user.display_name
diff --git a/README.md b/README.md
new file mode 100644
--- /dev/null
+++ b/README.md
@@ -0,0 +1,2 @@
+# Title
+body
"""


def test_parses_both_files():
    files = parse_unified_diff(DIFF)
    assert [f.path for f in files] == ["src/app.py", "README.md"]


def test_new_file_marked_added():
    files = parse_unified_diff(DIFF)
    assert files[1].change_type == "added"


def test_changed_lines_are_post_change_numbers():
    app = parse_unified_diff(DIFF)[0]
    # Hunk starts at new line 10 and spans 7 lines.
    assert min(app.changed_lines) == 10
    assert max(app.changed_lines) == 16


def test_added_line_numbers_skips_removals():
    app = parse_unified_diff(DIFF)[0]
    nums = added_line_numbers(app.hunks[0])
    # Three context lines, then one removal (not counted), then three additions.
    assert nums == [13, 14, 15]


def test_binary_files_flagged():
    binary = (
        "diff --git a/logo.png b/logo.png\n"
        "index 1111111..2222222 100644\n"
        "Binary files a/logo.png and b/logo.png differ\n"
    )
    assert parse_unified_diff(binary)[0].is_binary is True


def test_missing_ref_reports_the_ref_not_a_wrong_cause(tmp_path):
    """A bad ref in a repo that has commits must not be blamed on 'no commits'."""
    import subprocess

    from loupe.adapters.local_git import GitError, load

    for cmd in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True, capture_output=True)

    with pytest.raises(GitError) as exc:
        load(ref="no-such-ref", repo_root=str(tmp_path))
    assert "no-such-ref" in str(exc.value)
    assert "no commits" not in str(exc.value)


def test_empty_repo_says_so(tmp_path):
    import subprocess

    from loupe.adapters.local_git import GitError, load

    subprocess.run(
        ["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True, capture_output=True
    )
    for staged in (False, True):
        with pytest.raises(GitError) as exc:
            load(ref="HEAD~1", repo_root=str(tmp_path), staged=staged)
        assert "no commits" in str(exc.value)


def _init_repo(path, files: dict[str, str], message: str = "init") -> None:
    import subprocess

    for cmd in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(cmd, cwd=path, check=True, capture_output=True)
    for name, body in files.items():
        (path / name).write_text(body)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", message], cwd=path, check=True, capture_output=True)


def test_single_commit_repo_explains_itself(tmp_path):
    """A one-commit repo has a HEAD, so the 'no commits' branch does not fire —
    the message must still say why HEAD~1 is unavailable."""
    from loupe.adapters.local_git import GitError, load

    _init_repo(tmp_path, {"a.py": "x = 1\n"})
    with pytest.raises(GitError) as exc:
        load(ref="HEAD~1", repo_root=str(tmp_path))
    msg = str(exc.value)
    assert "only 1 commit" in msg
    assert "--repo-root" in msg


def test_empty_tree_ref_reviews_the_first_commit(tmp_path):
    """The empty-tree hash is suggested in that message, so it has to work — it is
    a tree with no commit behind it, which a `^{commit}` check would reject."""
    import subprocess

    from loupe.adapters.local_git import load

    _init_repo(tmp_path, {"a.py": "x = 1\ny = 2\n"})
    empty = subprocess.run(
        ["git", "hash-object", "-t", "tree", "/dev/null"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    ).stdout.strip()

    request = load(ref=empty, repo_root=str(tmp_path))
    assert [f.path for f in request.reviewable] == ["a.py"]
    assert request.files[0].content_after == "x = 1\ny = 2\n"
