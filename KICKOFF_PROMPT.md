# Kickoff prompt for Claude Code

Paste everything below the line, with `PROJECT_CONTEXT.md`, `fantasy_top250.csv`, and
`draft_2026_offense.csv` in the working directory.

---

I'm building a fantasy football prediction system. Read `PROJECT_CONTEXT.md` first and
completely before doing anything else — it is the spec, and it already contains the
decisions, the open questions, and the traps I've found so far.

## How I want to work

**You write the code. All of it.** Don't hand me pseudocode, don't ask me to fill in
blanks, don't stop and wait for me to type something out. Write real, runnable files,
run them, and show me what came back.

I am learning this material, so alongside the code I want to understand it:

- After you build something, walk me through what it does and why you made the choices
  you made — especially anywhere you picked one approach over an alternative.
- When something surprising shows up in the data, show me the actual numbers rather than
  summarizing them away.
- If a concept is easier to see than to read, draw it — a table, an ASCII sketch, a small
  worked example with real values.
- Never state a fact without telling me where it came from. If you aren't sure, say so
  rather than guessing.

Explaining is not the same as making me type. Build it, then teach me what you built.

## Scope for this session

Work through §12 of `PROJECT_CONTEXT.md`, steps 1 through 4, in order:

1. **Profile the `g` column** in `fantasy_top250.csv`
2. **Add `ppg`, `season_games`, `games_missed`** to a new derived file
3. **Build the year-over-year table** joining each player's season *t* to season *t+1*
4. **Race three rankings against the 68.8% baseline** — carry-forward, PPG, PPG × games

Run all four. Don't stop between them to ask whether to continue.

**Do not go past step 4.** Step 5 is me reading the result and deciding whether the
approach is worth building a model on. No machine learning this session — steps 1–4 are
sorting, joining, and set intersection. No nflverse downloads, no LLM work, no Modal.
Those are later stages (§11e) and they depend on this working first.

Stop and check with me before doing anything outside that list.

## What I care about in step 1

Profiling exists to find out whether the data behaves the way §8 assumes it does, before
anything gets built on top of it:

- Does max `g` per season step from 16 to 17 in 2021? (§8a — if it doesn't, something is
  wrong with either the file or my assumption)
- How many rows have `g < 4`, and does that vary by season?
- Are these season totals regular-season only, or do they include playoffs? (§8g — this
  changes every per-game calculation, so it has to be settled before step 2)
- Row count per season — is it exactly 250 every year, and if not, why not?

If profiling surfaces something that contradicts `PROJECT_CONTEXT.md`, say so directly
and stop. The document is my current understanding, not ground truth. Don't code around
a contradiction quietly.

## Things I don't want you to decide for me

`PROJECT_CONTEXT.md` §2 and §15 list open questions. They are open on purpose. Where one
blocks progress, tell me what the options are and what you'd pick and why — then let me
choose. Don't pick silently and build on it. Most likely to come up:

- **pandas vs Polars** (§2a). My notes are pandas, nflreadpy returns Polars. Undecided.
  Everything this session is a local CSV so either works — tell me which you're using and
  why, so it's a deliberate precedent rather than an accident.
- **Whether to keep the single FB** in `draft_2026_offense.csv` (§15).
- **The metric** (§3d). I'm leaning toward probability + calibration, not closed.

## Engineering habits I want in the code from line one

- **`assert`, not `print`, for anything that must be true.** A print inside a 26-season
  loop scrolls past unread; an assert stops the pipeline. State expected counts in
  advance, then assert them (§10).
- **Never a bare `how="left"` merge.** Use `indicator=True` and print the misses (§9).
  Step 3's year-over-year join is exactly where this matters — verify a player appearing
  in 2019 and 2021 but not 2020 doesn't accidentally link across the gap.
- **Never hardcode a season length or week count.** Derive it and check it (§8a, §8b).
- **Don't modify `fantasy_top250.csv`.** Write derived data to new files.
- Keep it readable over clever. I have to be able to follow it.

## Something to be suspicious of

If steps 1–4 run clean on the first try across 26 seasons, treat that as a signal to look
harder rather than a result. Ragged historical data usually complains about something.
Silence often means rows are being dropped quietly. Tell me if it goes too smoothly.

## Before you write anything

Read `PROJECT_CONTEXT.md`, then tell me:

1. Anything in it that looks wrong, internally inconsistent, or unsupported
2. What you'd want to check about `fantasy_top250.csv` that isn't already on my list
3. The one assumption in the document that would do the most damage if it turned out
   false

Then go ahead and build steps 1–4 without waiting for me.
