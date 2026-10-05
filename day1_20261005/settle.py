import json
P=json.load(open('predictions_locked_20261005.json'));R=json.load(open('results.json'))
st=ret=hit=n=0;rows=[]
for r in P['races']:
    k=f"{r['jcd']}-{r['rno']}"
    if k not in R: continue
    combo,pay=R[k];bet=next((b for b in r['bets'] if b['combo']==combo),None)
    w=pay*b['yen']//100 if (b:=bet) else 0
    n+=1;st+=r['stake'];ret+=w;hit+=bool(w)
    rows.append((r['venue'],r['rno'],combo,pay,w,w-r['stake']))
for x in rows: print(*x)
print(f"settled={n} hits={hit} stake={st} return={ret} pnl={ret-st} roi={ret/st:.1%}")
