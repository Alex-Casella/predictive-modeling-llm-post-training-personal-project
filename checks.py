"""Tiny check harness.

PROJECT_CONTEXT.md §10 asks for `assert`, not `print`, for anything that must be
true. A bare `assert` stops at the *first* failure, which is wrong for profiling --
if three things are broken I want to see three, not one.

So: record every check, print the wall, then raise at the end if any FAILED.
Same stopping guarantee, better diagnostics.
"""


class Checks:
    def __init__(self, title):
        self.title = title
        self.results = []

    def check(self, name, condition, detail=''):
        """Must be true. A failure stops the pipeline at report()."""
        self.results.append(('PASS' if condition else 'FAIL', name, detail))
        return bool(condition)

    def warn(self, name, condition, detail=''):
        """Known discrepancy. Printed loudly, does NOT stop the pipeline.

        Only for things already investigated and written up -- never as a way to
        quiet a check that has not been explained.
        """
        self.results.append(('PASS' if condition else 'WARN', name, detail))
        return bool(condition)

    def report(self):
        print(f'\n=== checks: {self.title} ===')
        for mark, name, detail in self.results:
            line = f'  [{mark}] {name}'
            if detail:
                line += f'  -- {detail}'
            print(line)
        failed = [r for r in self.results if r[0] == 'FAIL']
        warned = [r for r in self.results if r[0] == 'WARN']
        passed = len(self.results) - len(failed) - len(warned)
        print(f'  {passed} passed, {len(warned)} warned, {len(failed)} failed '
              f'(of {len(self.results)})')
        assert not failed, (
            f'{len(failed)} check(s) failed in "{self.title}": '
            + '; '.join(name for _, name, _ in failed)
        )
