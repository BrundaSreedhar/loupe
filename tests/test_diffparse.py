import pytest

from reviewer.adapters.diffparse import added_line_numbers, parse_unified_diff

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

    from reviewer.adapters.local_git import GitError, load

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

    from reviewer.adapters.local_git import GitError, load

    subprocess.run(
        ["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True, capture_output=True
    )
    for staged in (False, True):
        with pytest.raises(GitError) as exc:
            load(ref="HEAD~1", repo_root=str(tmp_path), staged=staged)
        assert "no commits" in str(exc.value)
