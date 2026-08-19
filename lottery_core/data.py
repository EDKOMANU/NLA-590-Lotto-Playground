"""Data loading and self-update (moved verbatim in behavior from the original predictor.py)."""
import csv
import datetime as dt

from .config import CSVF, NAMES


def load():
    draws = []
    with open(CSVF) as f:
        for r in csv.DictReader(f):
            draws.append({'date': dt.date.fromisoformat(r['date']), 'code': r['code'],
                          'win': [int(r[f'w{i}']) for i in range(1, 6)],
                          'mach': [int(r[f'm{i}']) for i in range(1, 6)] if r['m1'] else []})
    draws.sort(key=lambda d: d['date'])
    return draws


class UpdateError(RuntimeError):
    """Raised when the online refresh can't reach or read the results source. Carries a
    plain-language message and the HTTP status (if any) so callers can show the user
    what happened instead of crashing with a raw traceback."""
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


GAME_DOW = {'MS': 0, 'LT': 1, 'MW': 2, 'FT': 3, 'FB': 4, 'NW': 5, 'SA': 6}

# Site game labels -> our 2-letter codes. theb2blotto also lists non-NLA games (B2B,
# Noon Rush, NLA VAG, Alpha...) which are deliberately NOT mapped: this archive is the
# seven classic NLA games only, exactly as the Kaigee backfill was filtered.
B2B_GAME_NAMES = {
    'Monday Special': 'MS', 'Lucky Tuesday': 'LT', 'Mid Week': 'MW', 'MidWeek': 'MW',
    'Fortune Thursday': 'FT', 'Friday Bonanza': 'FB',
    'National Weekly Lotto': 'NW', 'National Weekly': 'NW', 'Sunday Aseda': 'SA',
}
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')


def _clean(html_fragment):
    import re
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', html_fragment)).strip()


def _valid_draw(iso, game, win, mach):
    """The same acceptance rules the original updater applied, kept verbatim in spirit:
    5 distinct numbers in 1-90, the date's weekday must match the game's fixed draw day
    (this is what catches mislabeled/garbage rows), and machine numbers are only kept
    when they are a clean set of 5 in range -- otherwise recorded as absent rather than
    half-trusted."""
    if len(win) != 5 or len(set(win)) != 5 or any(n < 1 or n > 90 for n in win):
        return None
    try:
        d = dt.date.fromisoformat(iso)
    except ValueError:
        return None
    if d.weekday() != GAME_DOW[game]:
        return None
    if len(mach) != 5 or len(set(mach)) != 5 or any(n < 1 or n > 90 for n in mach):
        mach = []
    return win, mach


def _fetch_b2blotto(pages=6):
    """Primary source: theb2blotto.com's results AJAX endpoint.

    Returns (draws, errors) where draws is [(iso_date, game_code, win, mach)].
    Each page holds ~10 results across all games (roughly 2-3 draw days), so a handful
    of pages comfortably covers the gap since the last refresh. Rows for non-NLA games
    and pagination artifacts are skipped by the game-name mapping."""
    import urllib.request
    import urllib.error
    import re
    out, errors = [], []
    for pn in range(1, pages + 1):
        url = f"https://www.theb2blotto.com/ajax/get_latest_results.php?pn={pn}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            html = urllib.request.urlopen(req, timeout=45).read().decode('utf-8', 'ignore')
        except urllib.error.HTTPError as e:
            errors.append(('theb2blotto', f'page {pn}', e.code, str(e.reason)))
            continue
        except (urllib.error.URLError, OSError) as e:
            errors.append(('theb2blotto', f'page {pn}', None, str(getattr(e, 'reason', e))))
            continue
        for row in re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.S):
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)
            if len(cells) < 3:
                continue
            game = B2B_GAME_NAMES.get(_clean(cells[0]))
            if not game:
                continue
            dm = re.match(r'(\d{1,2})-(\d{1,2})-(\d{4})', _clean(cells[1]))
            if not dm:
                continue
            iso = f"{dm.group(3)}-{int(dm.group(2)):02d}-{int(dm.group(1)):02d}"
            # The numbers cell holds "Win Numbers <5 nums> [Machine Numbers <5 nums>]".
            txt = _clean(cells[2])
            win_part, _, mach_part = txt.partition('Machine Numbers')
            win = [int(x) for x in re.findall(r'\b(\d{1,2})\b', win_part.replace('Win Numbers', ''))]
            mach = [int(x) for x in re.findall(r'\b(\d{1,2})\b', mach_part)]
            ok = _valid_draw(iso, game, win[:5], mach[:5])
            if ok:
                out.append((iso, game, ok[0], ok[1]))
    return out, errors


def _fetch_ghanayello():
    """Fallback source, kept for the day its bot protection is lifted. As of Jul 2026
    ghanayello.com sits behind Cloudflare and returns HTTP 403 to plain requests -- this
    is why it is no longer the primary path."""
    import urllib.request
    import urllib.parse
    import urllib.error
    import re
    GAME_NAMES = {'Monday Special': 'MS', 'Lucky Tuesday': 'LT', 'MidWeek': 'MW',
                  'Fortune Thursday': 'FT', 'Friday Bonanza': 'FB',
                  'National Weekly': 'NW', 'Sunday Aseda': 'SA'}
    MONTHS = {m: i + 1 for i, m in enumerate(['January', 'February', 'March', 'April', 'May', 'June',
                                               'July', 'August', 'September', 'October', 'November', 'December'])}
    today = dt.date.today()
    months = [today.strftime('%Y-%m'),
              (today.replace(day=1) - dt.timedelta(days=1)).strftime('%Y-%m')]
    out, errors = [], []
    for m in months:
        body = urllib.parse.urlencode({'_method': 'POST', 'data[Lottery][name]': '',
                                       'data[Lottery][date]': m}).encode()
        req = urllib.request.Request('https://www.ghanayello.com/lottery/results/history',
                                      data=body, headers={'User-Agent': UA,
                                      'Content-Type': 'application/x-www-form-urlencoded'})
        try:
            html = urllib.request.urlopen(req, timeout=45).read().decode('utf-8', 'ignore')
        except urllib.error.HTTPError as e:
            errors.append(('ghanayello', m, e.code, str(e.reason)))
            continue
        except (urllib.error.URLError, OSError) as e:
            errors.append(('ghanayello', m, None, str(getattr(e, 'reason', e))))
            continue
        for row in re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.S):
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)
            if len(cells) < 5:
                continue
            dm = re.search(r'(\d{1,2})\s+(\w+),\s+(\d{4})', cells[0])
            gm = re.search(r'>([^<]*?)\s*Results?</a>', cells[1])
            if not dm or not gm:
                continue
            game = GAME_NAMES.get(gm.group(1).strip())
            if not game or dm.group(2) not in MONTHS:
                continue
            iso = f"{dm.group(3)}-{MONTHS[dm.group(2)]:02d}-{int(dm.group(1)):02d}"
            win = [int(x) for x in re.findall(r'lotto_no_r[^>]*>\s*(\d+)', cells[2])]
            mach = [int(x) for x in re.findall(r'lotto_no_r[^>]*>\s*(\d+)', cells[3])]
            ok = _valid_draw(iso, game, win, mach)
            if ok:
                out.append((iso, game, ok[0], ok[1]))
    return out, errors


SOURCES = (('theb2blotto', _fetch_b2blotto), ('ghanayello', _fetch_ghanayello))


def update(sources=SOURCES):
    """Refresh the archive from the live results sources and append genuinely new draws.

    Tries each source in order and pools everything they return; a source that is down,
    blocked, or rate-limited contributes nothing but never aborts the run (ghanayello
    has been Cloudflare-blocked since ~Jul 2026, which is exactly why more than one
    source exists). Only (date, game) pairs absent from the CSV are appended, so
    repeated runs and overlapping sources cannot create duplicates -- the same
    (date, code) uniqueness the repair pass enforces.

    Returns {'added', 'errors', 'sources_ok'}; raises UpdateError only when NO source
    could be read at all, so callers can show a clear message instead of a traceback."""
    existing = {(r['date'], r['code']) for r in csv.DictReader(open(CSVF))}
    fetched, errors, sources_ok = [], [], []
    for name, fn in sources:
        try:
            rows, errs = fn()
        except Exception as e:                      # a parser change must not crash the app
            errors.append((name, '-', None, f'{type(e).__name__}: {e}'))
            continue
        errors.extend(errs)
        if rows:
            sources_ok.append(name)
            fetched.extend(rows)

    if not fetched:
        detail = "; ".join(f"{src} {where}: "
                           + (f"HTTP {code} {reason}" if code else str(reason))
                           for src, where, code, reason in errors) or "no rows returned"
        blocked = any(code in (403, 429, 503) for _s, _w, code, _r in errors)
        hint = (" The results sites are blocking automated requests right now. Your local "
                "archive is unaffected and every strategy still works on it."
                if blocked else " The results sites may be temporarily unreachable; try again later.")
        raise UpdateError(f"Couldn't fetch new results ({detail})." + hint,
                          status=next((c for _s, _w, c, _r in errors if c), None))

    # Dedupe across sources: the same draw reported by two sites is one draw. Lists are
    # converted to tuples first -- a list can't live in a set.
    unique = sorted({(i, g, tuple(w), tuple(m)) for i, g, w, m in fetched})
    new_rows = []
    for iso, game, win, mach in unique:
        if (iso, game) in existing:
            continue
        existing.add((iso, game))
        new_rows.append([iso, game, NAMES[game]] + list(win) + (list(mach) or [''] * 5) + [''])
    if new_rows:
        new_rows.sort(key=lambda r: (r[0], r[1]))
        with open(CSVF, 'a', newline='') as f:
            csv.writer(f).writerows(new_rows)

    added = len(new_rows)
    msg = (f"update complete: {added} new draw(s) appended from {', '.join(sources_ok)}"
           if added else f"already up to date (checked {', '.join(sources_ok)})")
    if errors:
        msg += f" [{len(errors)} source error(s)]"
    print(msg)
    return {'added': added, 'errors': errors, 'sources_ok': sources_ok}
