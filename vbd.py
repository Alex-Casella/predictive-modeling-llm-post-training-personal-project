"""
Value-based drafting (VBD) for fantasy football.

VBD = a player's fantasy points minus the points of the last startable player
at his position. It is the number a draft board should be ordered on, because
overall points ignore positional scarcity.

The point of this module is that league settings are an ARGUMENT, not a
constant. Any league size or lineup can be passed in. Nothing here is specific
to one league.

Usage:
    from vbd import LeagueConfig, ESPN_STANDARD, add_vbd

    league = LeagueConfig(teams=10, **ESPN_STANDARD)
    ranked = add_vbd(players, league, points_key='ppr')

`players` is a list of dicts, each needing a position key and a points key.
It works identically on actual results and on model predictions — that is the
whole reason it is a function rather than a stored column.
"""
from dataclasses import dataclass, field
from collections import defaultdict

# ESPN's default lineup, per ESPN's own settings documentation:
# 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 D/ST, 1 K, 7 bench.
# D/ST and K do not affect skill-position baselines and are omitted.
ESPN_STANDARD = {
    'starters': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1},
    'flex': 1,
    'flex_eligible': ('RB', 'WR', 'TE'),
}

# Fullback is not a fantasy roster slot; treat it as a running back.
POS_ALIAS = {'FB': 'RB'}


@dataclass
class LeagueConfig:
    teams: int
    starters: dict = field(default_factory=lambda: dict(ESPN_STANDARD['starters']))
    flex: int = 1
    flex_eligible: tuple = ('RB', 'WR', 'TE')

    def label(self):
        slots = ', '.join(f'{v} {k}' for k, v in self.starters.items())
        return f'{self.teams}-team | {slots}, {self.flex} flex'


def baselines(players, league, points_key='ppr', pos_key='pos'):
    """Return {position: baseline points} for one season under one league.

    Dedicated starter slots are filled first. Flex slots then go to the best
    remaining flex-eligible players regardless of position. A position's
    baseline is the points of the last player at that position to claim a
    starting slot, including via flex.
    """
    pos_of = lambda p: POS_ALIAS.get(p[pos_key], p[pos_key])
    pts_of = lambda p: float(p[points_key])

    by_pos = defaultdict(list)
    for p in players:
        by_pos[pos_of(p)].append(p)
    for group in by_pos.values():
        group.sort(key=lambda p: -pts_of(p))

    started = defaultdict(list)
    bench = []
    for pos, per_team in league.starters.items():
        n = per_team * league.teams
        started[pos] = by_pos[pos][:n]
        if pos in league.flex_eligible:
            bench += by_pos[pos][n:]

    bench.sort(key=lambda p: -pts_of(p))
    for p in bench[:league.flex * league.teams]:
        started[pos_of(p)].append(p)

    return {pos: min(pts_of(p) for p in group)
            for pos, group in started.items() if group}


def add_vbd(players, league, points_key='ppr', pos_key='pos', prefix='vbd'):
    """Attach VBD score and VBD rank to each player. Returns the same list.

    Works on actual season results or on predicted points — pass whichever
    column holds the number via `points_key`.
    """
    base = baselines(players, league, points_key, pos_key)
    for p in players:
        pos = POS_ALIAS.get(p[pos_key], p[pos_key])
        p[prefix] = round(float(p[points_key]) - base.get(pos, 0.0), 1)
    for i, p in enumerate(sorted(players, key=lambda p: -p[prefix]), 1):
        p[f'{prefix}_rk'] = i
    return players


def startable_counts(players, league, points_key='ppr', pos_key='pos'):
    """How many players at each position are startable. Useful for sanity
    checks — flex slots skew heavily toward one position in most leagues."""
    pos_of = lambda p: POS_ALIAS.get(p[pos_key], p[pos_key])
    base = baselines(players, league, points_key, pos_key)
    counts = defaultdict(int)
    for p in players:
        pos = pos_of(p)
        if pos in base and float(p[points_key]) >= base[pos]:
            counts[pos] += 1
    return dict(counts)


if __name__ == '__main__':
    import csv, sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'fantasy_top250.csv'
    rows = [r for r in csv.DictReader(open(path)) if r['season'] == '2025']
    for teams in (8, 10, 12, 14):
        cfg = LeagueConfig(teams=teams, **ESPN_STANDARD)
        add_vbd(rows, cfg)
        print(f'{cfg.label():<45} startable: {startable_counts(rows, cfg)}')
