"""GitHub PR adapter.

Uses the REST API directly rather than shelling out to `gh` — one less thing to
install, and it is the same code path in CI where `gh` often isn't present.
"""

from __future__ import annotations

import base64
import os

import httpx

from ..schema import ReviewRequest
from .diffparse import parse_unified_diff

API = "https://api.github.com"


def _headers(token: str, accept: str = "application/vnd.github+json") -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
    }


def load(repo: str, pr_number: int, token: str | None = None) -> ReviewRequest:
    token = token or os.getenv("GITHUB_TOKEN") or ""
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set — the GitHub adapter needs contents:read.")

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        meta = client.get(f"{API}/repos/{repo}/pulls/{pr_number}", headers=_headers(token))
        meta.raise_for_status()
        pr = meta.json()
        head_sha = pr["head"]["sha"]

        diff = client.get(
            f"{API}/repos/{repo}/pulls/{pr_number}",
            headers=_headers(token, "application/vnd.github.v3.diff"),
        )
        diff.raise_for_status()
        files = parse_unified_diff(diff.text)

        for f in files:
            if f.is_binary or f.change_type == "deleted":
                continue
            resp = client.get(
                f"{API}/repos/{repo}/contents/{f.path}",
                params={"ref": head_sha},
                headers=_headers(token),
            )
            if resp.status_code != 200:
                f.is_binary = True
                continue
            payload = resp.json()
            if payload.get("encoding") != "base64":
                f.is_binary = True
                continue
            try:
                f.content_after = base64.b64decode(payload["content"]).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                f.is_binary = True

    return ReviewRequest(
        source="github",
        ref=head_sha,
        title=pr.get("title") or "",
        body=pr.get("body") or "",
        files=files,
        repo=repo,
        pr_number=pr_number,
        head_sha=head_sha,
    )
