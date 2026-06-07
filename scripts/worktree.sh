#!/usr/bin/env bash
# Create a git worktree warm-started with this repo's heavy untracked
# artifacts, so the first build in it is incremental instead of the
# cold 60-min cascading re-elaboration.
#
# What gets carried over (APFS clonefile via `cp -c`: instant, ~zero
# extra disk — blocks are shared copy-on-write):
#   build/                 warm ttc / install / dylib / exec trees (~12G)
#   models/                HF checkpoints (gigabytes, slow to re-download)
#   data/ + packages/pytorch/data/   datasets (MNIST etc.)
#   vendored/              pinned third-party C sources
# Plain copies (small):
#   the four gitignored generated config files (HwConfig.idr,
#   HwExecutors.idr, Test/Config.idr x2) — preserves mtimes so make
#   doesn't consider the cloned ttc stale
#   .claude/               local Claude Code config
#
# Deliberately NOT copied: packages/{pytorch,jupyter}/.venv — Python
# venvs hardcode absolute paths in pyvenv.cfg and console-script
# shebangs; a copied venv half-works. Run `make ref-setup` /
# `make jupyter-install` inside the worktree if you need them.

set -eu

cd "$(dirname "$0")/.."

usage() {
	cat <<EOF
Usage: ./scripts/worktree.sh <name> [branch]

Create a git worktree at .worktrees/<name> warm-started with the
build cache, models, and datasets (APFS copy-on-write — no real disk
cost, no re-download, no cold re-elaboration).

Arguments:
  name      Worktree directory name (created at .worktrees/<name>)
  branch    Optional: existing branch to check out (default: reuses
            an existing branch named <name>, or creates it)

Note: untracked WORK-IN-PROGRESS source files are NOT carried over
(only the known generated configs + cache trees listed above) — copy
any uncommitted .idr you depend on yourself.

Examples:
  ./scripts/worktree.sh feat-tgather          # new branch 'feat-tgather'
  ./scripts/worktree.sh hotfix main           # check out 'main'

Cleanup when done:
  git worktree remove .worktrees/<name>
EOF
	exit 0
}

case "${1:-}" in
	""|-h|--help) usage ;;
esac

NAME="$1"
BRANCH="${2:-}"
WORKTREE_PATH=".worktrees/$NAME"

if [ -d "$WORKTREE_PATH" ]; then
	echo "Error: worktree already exists at $WORKTREE_PATH" >&2
	exit 1
fi

echo "Creating worktree at $WORKTREE_PATH..." >&2

if [ -n "$BRANCH" ]; then
	git worktree add "$WORKTREE_PATH" "$BRANCH" >&2
elif git show-ref --verify --quiet "refs/heads/$NAME"; then
	git worktree add "$WORKTREE_PATH" "$NAME" >&2
else
	git worktree add "$WORKTREE_PATH" -b "$NAME" >&2
fi

# Gitignored generated config files — copied with mtimes (-p) so the
# cloned build/ ttc trees stay newer than their sources and make
# treats them as warm.
for f in \
	packages/idris-ml/src/HwConfig.idr \
	packages/idris-ml/src/HwExecutors.idr \
	packages/idris-ml/src/Test/Config.idr \
	packages/idris-transformers/src/Test/Config.idr; do
	if [ -f "$f" ]; then
		cp -p "$f" "$WORKTREE_PATH/$f"
		echo "Copied $f" >&2
	fi
done

# Local Claude Code config.
if [ -d ".claude" ]; then
	mkdir -p "$WORKTREE_PATH/.claude"
	cp -Rp .claude/. "$WORKTREE_PATH/.claude/"
	echo "Copied .claude directory" >&2
fi

# Heavy untracked trees via APFS clonefile. If the filesystem doesn't
# support COW clones, skip rather than silently duplicating ~16G.
for d in build models data packages/pytorch/data vendored; do
	if [ -d "$d" ]; then
		mkdir -p "$(dirname "$WORKTREE_PATH/$d")"
		if cp -cRp "$d" "$WORKTREE_PATH/$d" 2>/dev/null; then
			echo "Cloned $d (APFS COW)" >&2
		else
			rm -rf "${WORKTREE_PATH:?}/$d"
			echo "Skipped $d (no APFS clonefile support?) — first build will recreate it" >&2
		fi
	fi
done

# Restore mtimes of tracked files that are byte-identical to the parent
# checkout. A fresh checkout stamps every file with "now", which is
# newer than the cloned build/ trees' .library-cache-stamp — so the
# first make would decide "library source changed" and wipe the warm
# ttc. Files that genuinely differ between the two HEADs keep their
# fresh mtimes (their rebuilds are wanted).
PARENT_HEAD="$(git rev-parse HEAD)"
CHANGED="$(mktemp)"
{
	git -C "$WORKTREE_PATH" diff --name-only HEAD "$PARENT_HEAD" || true
	# Parent's dirty files: their on-disk content (which the cloned
	# build/ was built from) differs from what the worktree checked
	# out, so they must keep fresh mtimes (rebuild is correct).
	git diff --name-only HEAD || true
} | sort -u > "$CHANGED"
restored=0
while IFS= read -r f; do
	if [ -f "$f" ] && [ -f "$WORKTREE_PATH/$f" ]; then
		touch -r "$f" "$WORKTREE_PATH/$f"
		restored=$((restored + 1))
	fi
done < <(git -C "$WORKTREE_PATH" ls-files | sort | comm -23 - "$CHANGED")
rm -f "$CHANGED"
echo "Restored mtimes on $restored unchanged tracked files (warm-ttc preservation)" >&2

cat >&2 <<EOF

Worktree ready! To enter it:
  cd $WORKTREE_PATH

The build cache is warm: 'make install' / 'make test' should be
incremental. Python venvs are NOT carried over — run 'make ref-setup'
(pytorch) or 'make jupyter-install' (jupyter) inside the worktree if
needed.

When done, return here and clean up:
  git worktree remove $WORKTREE_PATH        # add --force if it has build artifacts
EOF

echo "$WORKTREE_PATH"
