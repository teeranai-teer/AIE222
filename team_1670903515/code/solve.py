"""python solve.py <day_dir> <out_dir> [seconds] [--warm-start <plan_dir>]."""
import time

T0 = time.monotonic()
MAX_SECONDS = 120

import argparse
import itertools
import os
import signal
import sys
from random import Random
from tempfile import TemporaryDirectory

sys.dont_write_bytecode = True   # the source folder may be read-only: never write __pycache__ there

from aiconcrete import GRADES, load_day, m2t, read_csv, read_plan, write_csv  # noqa: E402
from solver import auto_shifts, fix_capacity, improve_shifts, optimize, repair_build, Builder, evaluate_rows  # noqa: E402
from dispatch import dispatch  # noqa: E402
from improve import eject, parameters, ranking, rebuild, replay, tune_trucks  # noqa: E402


class DeadlineReached(Exception):
    """Leave the search with its latest completed incumbent."""


def stop_search(_signum, _frame):
    raise DeadlineReached


def save(out_dir, rows, changes, prod):
    plan = [dict(r, load_start=m2t(r["load_start"])) for r in rows]
    plan.sort(key=lambda r: (r["load_start"], r["stream_id"], r["seq"]))
    outputs = (
        ("plan.csv", plan, ["stream_id", "seq", "plant_id", "truck_id", "load_start"]),
        ("order_changes.csv", changes, ["order_id", "new_first_arrival", "reason"]),
        ("production_plan.csv", prod, ["plant_id", "extra_lines", "from", "to"]),
    )
    with TemporaryDirectory(dir=out_dir) as staging:
        for name, data, fields in outputs:
            write_csv(os.path.join(staging, name), data, fields)
        for name, _, _ in outputs:
            os.replace(os.path.join(staging, name), os.path.join(out_dir, name))


def initial_plan(day, total):
    stage_a_end = T0 + 0.6 * total
    chg_s, r_shift = improve_shifts(day, auto_shifts(day), None, T0 + 0.25 * total)     # which flexible orders to move, and to when
    cands = [chg_s]
    if r_shift["sla_pass"]:       # SLA already safe: search shifts for profit, judged by two different greedy builders
        c1, _ = improve_shifts(day, chg_s, None, T0 + 0.34 * total, profit=True)
        c2, _ = improve_shifts(day, chg_s, None, T0 + 0.42 * total, profit=True, bkw=dict(trucks_k=10, co2_w=1.0))
        cands = [c1] + ([c2] if c2 != c1 else [])

    def _k(r):
        return (r["errors"] == 0, r["sla_checks_passed"], r["sla_score"], r["contribution_baht"], -r["km_total"])
    pick = None
    for chg in cands:
        chs = [dict(order_id=o, new_first_arrival=m2t(v), reason="ขอลูกค้าเลื่อนเวลา") for o, v in chg.items()]
        r0, x0, st = repair_build(day, chs, None, T0 + 0.5 * total)
        if x0["sla_pass"]:        # multi-start: same repair buffers, different truck/CO2 weights in the builder
            for k_ in (3, 6, 10, 15, 25):
                for cw in (0.0, 1.0, 2.0, 4.0, 8.0):
                    if time.monotonic() > T0 + 0.55 * total:
                        break
                    r_ = Builder(day, chs, None, st[0], st[1], wait_w=1.0, trucks_k=k_, co2_w=cw).run()
                    x_ = evaluate_rows(day, r_, None, chs)
                    if x_["sla_pass"] and x_["errors"] == 0 and _k(x_) > _k(x0):
                        r0, x0 = r_, x_
        if pick is None or _k(x0) > _k(pick[3]):
            pick = (chg, chs, r0, x0, st)
    chg0, changes0, rows0, res0, state0 = pick
    if res0["sla_pass"]:
        # SLA slack: in grades where one order may miss on-time and GREEN, choose WHICH order(s) to let slip
        by = {}
        for oid, sids in day.orders.items():
            by.setdefault(day.streams[sids[0]]["grade"], []).append(oid)
        slack = []
        for g, os_ in by.items():
            n = len(os_)
            a = max(k for k in range(n + 1) if (n - k) / n >= GRADES[g][2] - 1e-9)
            b = max(k for k in range(n + 1) if (n - k) / n >= GRADES[g][1] - 1e-9)
            if min(a, b) >= 1:
                slack.append(os_)
        combos = list(itertools.islice(itertools.product(*slack), 60)) if slack else []
        bestc = None
        for combo in combos:
            if time.monotonic() > T0 + 0.58 * total:
                break
            st = {sid for o in combo for sid in day.orders[o]}
            r_ = Builder(day, changes0, None, state0[0], state0[1], trucks_k=15, co2_w=1.0, free_on=st, free_gr=st).run()
            x_ = evaluate_rows(day, r_, None, changes0)
            if x_["sla_pass"] and x_["errors"] == 0 and (bestc is None or _k(x_) > _k(bestc[1])):
                bestc = (st, x_, r_)
        if bestc is not None:
            st = bestc[0]
            for k_ in (10, 15, 25):
                for cw in (0.0, 1.0, 4.0):
                    if time.monotonic() > T0 + 0.6 * total:
                        break
                    r_ = Builder(day, changes0, None, state0[0], state0[1], trucks_k=k_, co2_w=cw, free_on=st, free_gr=st).run()
                    x_ = evaluate_rows(day, r_, None, changes0)
                    if x_["sla_pass"] and x_["errors"] == 0 and _k(x_) > _k(bestc[1]):
                        bestc = (st, x_, r_)
            if _k(bestc[1]) > _k(res0):
                rows0, res0 = bestc[2], bestc[1]
    state0 = (state0[0], state0[1], chg0)
    best = ((res0["sla_checks_passed"], res0["contribution_baht"]), (rows0, changes0, [], res0))
    for i in range(6 if not res0["sla_pass"] else 0):                      # restart with another seed until the SLA passes
        left = stage_a_end - time.monotonic()
        if left <= 1:
            break
        out = optimize(day, min(left, 0.2 * total), 1 + i, patience=150, init=state0)
        key = (out[3]["sla_checks_passed"], out[3]["contribution_baht"])
        if best is None or key > best[0]:
            best = (key, out)
        if out[3]["sla_pass"]:
            break
    return best[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('day_dir')
    parser.add_argument('out_dir')
    parser.add_argument('seconds', nargs='?', type=float, default=MAX_SECONDS,
                        help=f'total time in seconds per day (maximum {MAX_SECONDS})')
    parser.add_argument('--warm-start', metavar='PLAN_DIR')
    parser.add_argument('--seed', type=int, help='search seed (default: 37 on holidays, 15 otherwise)')
    args = parser.parse_args()
    if not 0 < args.seconds <= MAX_SECONDS:
        parser.error(f'seconds must be greater than 0 and at most {MAX_SECONDS}')
    if args.warm_start and os.path.realpath(args.warm_start) == os.path.realpath(args.out_dir):
        parser.error('use a separate output folder to keep the warm start intact')
    # Reserve time for writing the CSVs and process teardown.
    search_seconds = args.seconds - min(2.5, args.seconds * 0.1)
    deadline = T0 + search_seconds
    incumbent = None

    def offer(rows, changes, prod, result=None):
        nonlocal incumbent
        if result is None:
            _, result = replay(day, rows, changes, prod)
        if incumbent is None or ranking(result) > ranking(incumbent[3]):
            incumbent = rows, changes, prod, result

    alarm = hasattr(signal, 'setitimer')
    previous_handler = signal.signal(signal.SIGALRM, stop_search) if alarm else None
    try:
        if alarm:
            signal.setitimer(signal.ITIMER_REAL, max(0.000001, deadline - time.monotonic()))
        os.makedirs(args.out_dir, exist_ok=True)
        day = load_day(args.day_dir)
        fix_capacity(day)
        seed = args.seed if args.seed is not None else (37 if day.day_type == 'holiday' else 15)
        if args.warm_start:
            try:
                rows = read_plan(os.path.join(args.warm_start, 'plan.csv'))
                changes = read_csv(os.path.join(args.warm_start, 'order_changes.csv'))
                prod = read_csv(os.path.join(args.warm_start, 'production_plan.csv'))
            except (OSError, ValueError, KeyError) as exc:
                parser.error(f'invalid warm start: {exc}')
            _, result = replay(day, rows, changes, prod)
            if result['errors'] or not rows:
                parser.error('warm start must contain a valid schedule for this day')
            offer(rows, changes, prod, result)
        else:
            default_changes = [dict(order_id=oid, new_first_arrival=m2t(value), reason='เลี่ยงการปิดถนน')
                               for oid, value in auto_shifts(day).items()]
            offer(Builder(day, default_changes).run(), default_changes, [])
            if time.monotonic() < T0 + search_seconds * 0.44:
                rows, changes, prod, _ = initial_plan(day, search_seconds * 0.74)
                offer(rows, changes, prod)
            rng = Random(seed)
            for i in itertools.count():
                if time.monotonic() >= T0 + search_seconds * 0.51:
                    break
                changes = incumbent[1] if i % 3 else default_changes
                prod = incumbent[2]
                offer(dispatch(day, changes, prod, **parameters(rng)), changes, prod)

        rows, changes, prod, result = incumbent
        if rows and not result['errors']:
            rebuild(day, rows, changes, prod, T0 + search_seconds * 0.85, offer, seed)
            if time.monotonic() < T0 + search_seconds * 0.92:
                eject(day, *incumbent[:3], T0 + search_seconds * 0.92, offer, seed)
            if time.monotonic() < deadline:
                tune_trucks(day, *incumbent[:3], deadline, offer, seed)
    except DeadlineReached:
        pass
    finally:
        if alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    if incumbent is None:
        parser.exit(1, 'Time limit reached before a plan was ready.\n')
    rows, changes, prod, result = incumbent
    save(args.out_dir, rows, changes, prod)
    print(f"Day {day.info.get('day')}: SLA {result['sla_checks_passed']}/{result['sla_checks_total']} "
          f"Contribution {result['contribution_baht']:,} THB · after close {result['after_close']} "
          f"· {time.monotonic() - T0:.2f}s")


if __name__ == '__main__':
    main()
