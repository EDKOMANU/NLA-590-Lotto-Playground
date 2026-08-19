"""Clean raw scraped NLA 5/90 draws -> ghana_lotto_history.csv"""
import csv, os
BASE = os.path.dirname(os.path.abspath(__file__))
GAMES = {'MS':'Monday Special','LT':'Lucky Tuesday','MW':'MidWeek','FT':'Fortune Thursday',
         'FB':'Friday Bonanza','NW':'National Weekly','SA':'Sunday Aseda'}
# expected weekday per game (Mon=0)
DOW = {'MS':0,'LT':1,'MW':2,'FT':3,'FB':4,'NW':5,'SA':6}
import datetime as dt

rows = []
seen = set()
bad = []
for part in ['raw_part0a.txt','raw_part0b.txt','raw_part1.txt','raw_part2.txt']:
    for line in open(os.path.join(BASE,part)):
        line=line.strip()
        if not line: continue
        f = line.split('|')
        if len(f)!=5: bad.append(('fields',line)); continue
        date,game,win,mach,ev = f
        if game not in GAMES: bad.append(('game',line)); continue
        try:
            w=[int(x) for x in win.split('.')]
        except ValueError:
            bad.append(('nums',line)); continue
        if len(w)!=5 or len(set(w))!=5 or any(n<1 or n>90 for n in w):
            bad.append(('range',line)); continue
        try:
            m=[int(x) for x in mach.split('.')] if mach else []
            if m and (len(m)!=5 or any(n<1 or n>90 for n in m)): m=[]
        except ValueError:
            m=[]
        d = dt.date.fromisoformat(date)
        # sanity: game day-of-week must match (guards against mislabeled dup rows)
        if d.weekday()!=DOW[game]: bad.append(('dow',line)); continue
        key=(date,game)
        if key in seen: bad.append(('dup',line)); continue
        seen.add(key)
        rows.append([date,game,GAMES[game]]+w+ (m if m else ['','','','','']) + [ev if ev!='-' else ''])

rows.sort(key=lambda r:(r[0],r[1]))
with open(os.path.join(BASE,'ghana_lotto_history.csv'),'w',newline='') as fo:
    wcsv=csv.writer(fo)
    wcsv.writerow(['date','code','game','w1','w2','w3','w4','w5','m1','m2','m3','m4','m5','event'])
    wcsv.writerows(rows)

print('clean rows:', len(rows))
from collections import Counter
print(Counter(r[1] for r in rows))
print('dropped:', len(bad))
for reason,l in bad: print(' -',reason,l[:70])
print('date range:', rows[0][0], '->', rows[-1][0])