#!/usr/bin/env bash
# Install `agentgate` as a command available from any directory.
#
# Uses `uv tool install`, which builds an isolated environment for the package and
# links its entry point into ~/.local/bin — so the tool's dependencies never
# collide with a project you happen to be standing in when you run it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/agentgate"
CONFIG="$CONFIG_DIR/.env"

say()  { printf '  %s\n' "$1"; }
fail() { printf '  error: %s\n' "$1" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || fail "uv is not installed. See https://docs.astral.sh/uv/"

# --editable keeps the command pointed at this checkout, so edits take effect
# without reinstalling. Drop it for a frozen copy that survives moving the repo.
say "installing from $REPO"
uv tool install --editable "$REPO" --force --quiet
say "installed"

# ── config ────────────────────────────────────────────────────────────────
# A globally installed command cannot rely on a .env sitting in the checkout, so
# settings live in a user config file that is found from any directory.
if [ ! -f "$CONFIG" ]; then
  mkdir -p "$CONFIG_DIR"
  cp "$REPO/.env.example" "$CONFIG"
  chmod 600 "$CONFIG"          # it will hold an API key
  say "created $CONFIG — add your API key there"
else
  say "keeping existing $CONFIG"
fi

# ── PATH ──────────────────────────────────────────────────────────────────
BIN="$HOME/.local/bin"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *)
    say ""
    say "$BIN is not on your PATH. Add this to your shell profile:"
    say "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    say "or run: uv tool update-shell"
    ;;
esac

# ── verify ────────────────────────────────────────────────────────────────
say ""
if command -v agentgate >/dev/null 2>&1; then
  say "run 'agentgate doctor' to check your configuration"
else
  say "installed, but 'agentgate' is not on your PATH yet — see the note above"
fi
