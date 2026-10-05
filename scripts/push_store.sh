#!/usr/bin/env bash
# store/ (data ブランチ) の変更をコミットして push。並列ジョブ同士の競合は rebase で再試行。
set -e
cd store
git config user.name github-actions; git config user.email github-actions@users.noreply.github.com
git add -A
git diff --cached --quiet && { echo "no changes"; exit 0; }
git commit -qm "$1"
for i in 1 2 3 4 5 6 7 8; do
  git pull -q --rebase origin data && git push -q origin HEAD:data && exit 0
  sleep $((RANDOM % 20 + 5))
done
exit 1
