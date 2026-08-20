# Stripping tool co-author trailers from history

**Date:** 2026-08-20
**Scope:** commit metadata only — messages and one commit's author identity.
**Files changed by the rewrite:** none. Zero. This is provable and was proved; see
[Verification](#verification-before-the-push).

> ## ⚠️ Commit SHAs in earlier session records are stale
>
> **Every session record in this directory dated before 2026-08-20 cites commit
> SHAs from the pre-rewrite history. Those SHAs no longer exist.** The rewrite on
> 2026-08-20 changed 229 of the 232 commits on `main`.
>
> **The content is identical — only the hashes moved.** A record saying "fixed in
> `abac2ad`" is still telling the truth about what was fixed and what the change
> contained; the hash is simply no longer a valid way to look it up.
>
> To resolve an old SHA to its current one, use the mapping preserved beside this
> record:
>
> ```bash
> grep ^abac2ad docs/sessions/commit-map-2026-08-20-strip.txt
> ```
>
> The map covers all 232 commits (`old new`, one pair per line). The full
> pre-rewrite history is also preserved as a git bundle at
> `../AI4CM-pre-strip-20260820.bundle` (outside the repository, 8.6M), which can be
> cloned or fetched from directly if you need the old objects themselves rather
> than just the mapping.

---

## Why

Claude appeared in the repository's GitHub contributors list. The goal was to
remove it. No file content was to change anywhere.

## What the survey found

The brief assumed the problem was co-author trailers. It was, mostly — but not
only, and the survey found the part that would have made the exercise fail.

**Pattern counts on `main`** (232 commits total, case-insensitive):

| Pattern | Commits |
|---|---|
| `Co-Authored-By: Claude` | **18** |
| `Generated with [Claude Code]` | **0** |
| either | **18** |

The trailer appeared in three variants, all repo-wide:

```
Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>   ×20
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>                ×12
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>   ×4
```

(36 repo-wide; 18 of them on `main`, the rest only on stale local branches — see
[Branches](#branches).)

The `Generated with [Claude Code]` pattern **did not occur at all**. It was kept
in the filter anyway, at no cost. One commit message contains the prose "*My
first orientation blocks were generated with*" — it does not contain the
bracketed phrase, so it was never at risk. This was checked specifically before
the rewrite and confirmed intact after it.

### The finding that changed the plan

Stripping trailers alone would **not** have achieved the goal. One commit on
`main` was *authored* by Claude:

```
27620536bd2ab89842577ea6e464ea3b265a98a4
  author:    Claude (pair session) <noreply@anthropic.com>
  committer: Claude (pair session) <noreply@anthropic.com>
  date:      2026-07-29 12:51:05 +0000
  subject:   Gate best-model display on run quality (M-1)
```

Author counts on `main` before the rewrite: 231 `omakhlouk`, **1** `Claude (pair
session)`. GitHub's contributors list is built from commit authorship, so that
single commit would have kept Claude listed after all 18 trailers were gone. This
was surfaced at plan time and the author rewrite was explicitly authorised before
anything ran.

### Tags

`git ls-remote --tags origin` → **0**. Local tags → **0**. Nothing to rewrite or
re-push. No tags were pushed.

## The rewrite

`git-filter-repo` was not installed (no Homebrew, no pipx; the only `pip3` on
PATH belonged to `frontend/.venv`, which was left alone). It was installed into a
throwaway virtualenv outside the repository.

```bash
git-filter-repo --refs main --force \
--message-callback '
drop = (b"co-authored-by: claude", b"generated with [claude code]")
kept = [l for l in message.split(b"\n") if not any(d in l.lower() for d in drop)]
return b"\n".join(kept).rstrip(b"\n") + b"\n"
' \
--name-callback '
return b"omakhlouk" if name == b"Claude (pair session)" else name
' \
--email-callback '
return b"omakhlouk@worldbankgroup.org" if email.lower() == b"noreply@anthropic.com" else email
'
```

The message callback only ever deletes whole lines; the `rstrip` removes the
blank line the trailer left orphaned at the end of the message. Every rewritten
commit **reuses its original tree object** — this is why no file content can
change, and the verification below demonstrates it rather than asserting it.

`--refs main` implies `--partial`, which was chosen deliberately over a full
rewrite for two reasons: the `origin` remote is not removed (so no re-add is
needed), and the backup branch is **not** rewritten in lockstep — which is what
makes `git diff backup-pre-strip-20260820 main` a genuine independent check
rather than a tautology. Under a full-repo rewrite the backup would be rewritten
identically and the diff would be empty no matter what had happened.

Result: `Parsed 232 commits`, new history written in 0.22s.
`675bbe6` → `b988d42`. 229 of 232 SHAs changed; the 3 oldest commits predate the
first match and kept their hashes.

## Backups, taken before anything ran

```bash
git branch backup-pre-strip-20260820 main          # 675bbe61e06de615c2e4092f6821efe51f94a867
git bundle create ../AI4CM-pre-strip-20260820.bundle --all
```

`git bundle verify` → *"The bundle records a complete history."* (8.6M, sha1).
Bundled `--all` rather than just `main` so it is a complete mapping source for
every SHA cited in any older record.

## Verification, before the push

All four required checks, run on the rewritten local `main`:

```
=== 1. TRAILER MATCHES ON MAIN (expect 0) ===
0                       (git log --grep, both patterns, -i)
0                       (raw scan of every message body)

=== 2. CLAUDE AUTHORSHIP/COMMITTERSHIP ON MAIN (expect none) ===
   authors:
 232 omakhlouk <omakhlouk@worldbankgroup.org>
   committers:
 187 omakhlouk <omakhlouk@worldbankgroup.org>
  45 GitHub <noreply@github.com>
   anthropic/claude identity matches: 0

=== 3. COMMIT COUNT (expect 232) ===
232

=== 4. CONTENT DIFF vs BACKUP (expect EMPTY) ===
   diff byte length: 0
   tree hash backup: 97e3229de2893f3970bcf5facdcc4e55361684bf
   tree hash main:   97e3229de2893f3970bcf5facdcc4e55361684bf
```

Two further checks were run beyond the brief, because a HEAD-only tree comparison
proves less than it appears to — it would pass even if two intermediate commits
had swapped content and swapped back:

```
=== PER-COMMIT TREE SEQUENCE (all 232 commits, in order) ===
IDENTICAL: all 232 tree hashes match in order

=== PROSE LINE PRESERVED (expect 1) ===
1
```

The per-commit tree sequence is the real proof of "no content changed": every one
of the 232 commits points at exactly the same tree object as its pre-rewrite
counterpart, in the same order.

## The push

```
+ 675bbe6...b988d42 main -> main (forced update)      # --force-with-lease
```

Branch protection on `main` was lifted in the GitHub UI beforehand and **must be
restored**.

## Branches

All 20 non-`main` branches on `origin` were fully merged ancestors of `main` — 0
commits ahead, nothing unique on any of them. **None was unmerged**, so none
needed a decision. They were deleted rather than rewritten and re-pushed: leaving
them would have pinned the *old*, trailer-bearing commit objects as reachable on
GitHub permanently, defeating the exercise.

Deleted from `origin`: `branch-new`, `ux-lab-models-tooltips`,
`ux-lab-models-tooltips-v2`, `fix-forecast-integrity`, `fix/critical-bugs`,
`fix/forecast-trust`, `fix/test-collection`, `fix/unified-baseline`,
`fix/summary-cdl-integrity`, `fix/e-quantile-honest-eval`,
`fix/residual-rf-intervals`, `feat/backtest-report`, `fix/backtest-report-var`,
`fix/best-model-consistency`, `fix/signal-sentinel-semantics`,
`fix/b-ml-overfitting`, `feat/calendar-features`, `merge/phase1-trust`,
`model/excellence`, `ux/final-polish`.

Eleven **local** branches carried pre-rewrite history from an earlier rewrite
(2026-08-19) and held the other 18 trailer-bearing commits, which existed nowhere
on `origin`. `--refs main` left them untouched; they were then deleted locally so
they cannot confuse future work. `backup-local-main-20260819` was **kept** — it
is a deliberate backup from that earlier rewrite, not stale work.

`origin` now has exactly one branch, `main`, and zero tags.

## Verification, on a fresh clone

Cloned fresh from GitHub after the push:

```
--- remote refs ---
b988d422b1536b63d3d821ca17c15ea5c7123867  refs/heads/main
--- remote tags --- 0

A. trailer matches in FULL log:          0
B. authors:     232 omakhlouk <omakhlouk@worldbankgroup.org>
   committers:  187 omakhlouk <omakhlouk@worldbankgroup.org>
                 45 GitHub <noreply@github.com>
   anthropic/claude identity matches:    0
C. HEAD tree hash, fresh clone: 97e3229de2893f3970bcf5facdcc4e55361684bf
   HEAD tree hash, backup ref:  97e3229de2893f3970bcf5facdcc4e55361684bf
D. commit count:                         232
```

## Known consequences

- **Fourteen short SHAs quoted inside commit messages now dangle.** filter-repo
  reported them: `60976736`, `0b009fd0`, `58082051`, `1043717`, `15fb6ee`,
  `c0fd448`, `772f3d25…`, `1d952420…`, `7e520b4f…`, `2be3596a…`, `9c041497…`,
  `4b480eae`, `8a172f8e`, `0b009fd0…`. They were left as written. Resolve them
  through the commit map like any other stale SHA.
- **Open pull requests referencing the deleted branches are now orphaned** on
  GitHub. All 20 were merged, so nothing is lost, but the PR pages will show
  their branches as deleted and their commit links as unreachable.
- **The force-push left unreachable objects on GitHub's side.** See
  [the support note](#note-for-the-pending-github-support-ticket).

## Future commits

`.claude/settings.json` is now committed with:

```json
{
  "includeCoAuthoredBy": false
}
```

so no future commit made through the tool carries the trailer. This session's own
commits are authored and committed solely by `omakhlouk
<omakhlouk@worldbankgroup.org>`.

## Note for the pending GitHub Support ticket

The 2026-08-20 force-push to `main`, plus the deletion of 20 branches, left the
entire pre-rewrite history **unreachable but not yet garbage-collected** on
GitHub's servers. Until GitHub GCs them, the old commits — trailers, Claude
authorship and all — remain retrievable by direct SHA URL, and may still appear
in cached views.

**The existing GC request should be amended to name these objects too.** The
specific facts to give support:

- Repository: `WBG-ITS-Innovation/AI4CM`
- Old `main` tip, now unreachable: `675bbe61e06de615c2e4092f6821efe51f94a867`
- New `main` tip: `b988d422b1536b63d3d821ca17c15ea5c7123867`
- 229 commits newly unreachable on `main`'s line alone; the full set of old
  object IDs is the left-hand column of
  `docs/sessions/commit-map-2026-08-20-strip.txt`
- 20 deleted branches, tips listed above, all merged ancestors of the old `main`
- Ask: garbage-collect unreachable objects and purge cached views, so no
  unreachable commit remains fetchable by SHA
