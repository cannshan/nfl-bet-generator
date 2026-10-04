"""One-off backfill for the 2026-10-04 tracking fixes (app/tracking.py).

1. Re-labels season/week on every nfl_predictions row from the row's own
   kickoff, with the same tracking.season_week_for_kickoff() that new rows
   are stamped with. The old labels came from ESPN's "current week" at
   logging time, which a cache-only page view could serve days stale (rows
   for 10/04 games were labeled week 3, some 9/27 games week 1).
2. Runs the current settlement rules over every pending row -- the same
   tracking.compute_settlements() that settle_pending() uses -- which mainly
   voids props on players who didn't play (they used to sit "pending"
   forever, since there's never a box-score row to grade them on).
3. Settles any pending whole tickets (once the nfl_tickets table exists).

Default is a DRY RUN: every change is printed and nothing is written.

  .venv/Scripts/python.exe scripts/backfill_tracking.py            # dry run, cached data only
  .venv/Scripts/python.exe scripts/backfill_tracking.py --live     # dry run, may fetch free nflverse data
  .venv/Scripts/python.exe scripts/backfill_tracking.py --apply    # write the changes

--live lets settlement fetch nflverse data that isn't cached yet (final
scores, box scores, and the snap counts that tell "didn't play" apart from
"played but never touched the ball"). Free GitHub downloads, no API key;
it never calls The Odds API (settlement doesn't use odds). Without it,
nothing outside the app's own Supabase cache is touched.
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import cache_utils, formatting, tracking  # noqa: E402


def _describe(row):
    kickoff = formatting.format_kickoff_et(row.get("commence_time")) or row.get("commence_time")
    return f"#{row['id']} {kickoff} {row['away_team']} @ {row['home_team']} | {row['market']}: {row['selection']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    parser.add_argument("--live", action="store_true", help="allow free nflverse fetches for data not cached yet")
    args = parser.parse_args()
    cache_utils.set_mode("active" if args.live else "passive")

    rows = tracking._select_all(lambda: tracking._sb().table(tracking.TABLE).select("*"))
    print(f"{len(rows)} nfl_predictions rows ({'APPLY' if args.apply else 'DRY RUN'}, "
          f"{'live nflverse allowed' if args.live else 'cached data only'})\n")

    updates = collections.defaultdict(dict)

    # 1. season/week labels
    relabels = collections.Counter()
    print("== season/week relabels ==")
    for row in sorted(rows, key=lambda r: (r.get("commence_time") or "", r["id"])):
        season, week = tracking.season_week_for_kickoff(row.get("commence_time"))
        if season is None:
            print(f"  skip (unparseable kickoff) {_describe(row)}")
            continue
        if (season, week) != (row.get("season"), row.get("week")):
            updates[row["id"]].update({"season": season, "week": week})
            relabels[(row.get("season"), row.get("week"), season, week)] += 1
            print(f"  {row.get('season')}/wk{row.get('week')} -> {season}/wk{week}  {_describe(row)}")
    print(f"  {sum(relabels.values())} rows relabeled")
    for (os_, ow, ns, nw), n in sorted(relabels.items(), key=lambda kv: (kv[0][3] or 0, kv[0][1] or 0)):
        print(f"    week {ow} -> {nw} ({os_}->{ns}): {n}")

    # 2. settlement of pending rows
    print("\n== settlement of pending rows ==")
    pending = [r for r in rows if r["status"] == "pending"]
    explanations = {}
    results_by_season = {}
    settlements = tracking.compute_settlements(
        pending, results_by_season=results_by_season, explanations=explanations,
    )
    by_id = {r["id"]: r for r in rows}
    status_changes = collections.Counter()
    for row_id, payload in sorted(settlements.items()):
        row = by_id[row_id]
        updates[row_id].update(payload)
        status_changes[(row["status"], payload["status"])] += 1
        actual = "" if payload["actual_value"] is None else f" (actual {payload['actual_value']:g})"
        reason = f"  [{explanations[row_id]}]" if row_id in explanations else ""
        print(f"  pending -> {payload['status']}{actual}  {_describe(row)}{reason}")
    still_pending = [r for r in pending if r["id"] not in settlements]
    print(f"  {len(settlements)} settled, {len(still_pending)} still pending")
    for (old, new), n in sorted(status_changes.items()):
        print(f"    {old} -> {new}: {n}")
    for row in still_pending:
        print(f"    still pending: {_describe(row)}")

    # Information only -- nothing is written for these; get_track_record()
    # leaves them out of every number.
    in_play = [r for r in rows if not tracking._logged_before_kickoff(r)]
    print(f"\n== logged after kickoff (excluded from the Track Record, unchanged): {len(in_play)} ==")
    for row in in_play:
        print(f"    logged {row['created_at'][:16]}Z  {_describe(row)}  [{row['status']}]")

    # 3. tickets
    print("\n== tickets ==")
    tickets, leg_rows = tracking._pending_tickets_and_leg_rows()
    ticket_updates = {}
    if tickets is None:
        print("  nfl_tickets table not created yet (run the 2026-10-04 migration in supabase_setup.sql)")
    else:
        # Leg statuses as they'll be AFTER this backfill's own settlements.
        settled_now = {r_id: p["status"] for r_id, p in settlements.items()}
        leg_rows = [{**r, "status": settled_now.get(r["id"], r["status"])} for r in leg_rows]
        ticket_updates = tracking.compute_ticket_settlements(tickets, leg_rows, results_by_season=results_by_season)
        print(f"  {len(tickets)} pending tickets, {len(ticket_updates)} settle now: "
              f"{dict(collections.Counter(p['status'] for p in ticket_updates.values()))}")

    if not args.apply:
        print(f"\nDRY RUN: {len(updates)} prediction rows and {len(ticket_updates)} tickets would change. "
              f"Re-run with --apply to write.")
        return

    written = tracking._update_rows(tracking.TABLE, dict(updates))
    print(f"\nwrote {written}/{len(updates)} prediction rows")
    if ticket_updates:
        written = tracking._update_rows(tracking.TICKETS_TABLE, ticket_updates)
        print(f"wrote {written}/{len(ticket_updates)} tickets")


if __name__ == "__main__":
    main()
