import zlib, struct, glob, re, json
from datetime import datetime, timedelta
ev=[]
for f in sorted(glob.glob("datz/*.datZ")):
    raw=zlib.decompress(open(f,"rb").read()); m=raw.find(b"Phases in use:")
    d,t=re.search(rb"Controller Data Log Beginning:,([\d/]+),([\d:.]+)", raw[:m]).groups()
    base=datetime.strptime(d.decode()+" "+t.decode(),"%m/%d/%Y %H:%M:%S.%f")
    p=raw[raw.find(b"\n",m)+1:]; n=len(p)//4; r=struct.unpack(">"+"BBH"*n,p)
    for i,(c,pa,o) in enumerate(zip(r[0::3],r[1::3],r[2::3])):
        if c in (89,90) and pa>=9: ev.append((f[-9:-5], i, base+timedelta(seconds=o/10), c, pa))
# file order is chronological in real time; pair ON then OFF per ped in log order
open_={}; out=[]
for fn,i,ts,c,pa in ev:
    if c==90: open_[pa]=ts
    elif pa in open_: out.append((pa, open_.pop(pa), ts))
recs=[json.loads(l) for l in open("bench.jsonl")][-5:]
exp=[]
for r in recs:
    for p in r["pulses"]:
        w=p["width_s"] + (p.get("shift_s",0) if p["role"]=="set" else 0)
        exp.append((p["ped"], p["role"], p["width_s"], w))
print("%-4s %-9s %8s %9s %9s"%("ped","role","sent","expected","logged"))
for (pa,on,off),(ped,role,sent,w) in zip(out,exp):
    print("%-4d %-9s %8.1f %9.1f %9.1f   %s%s"%(pa,role,sent,w,(off-on).total_seconds(),on.strftime("%H:%M:%S.%f")[:10], "" if pa==ped else "  PED MISMATCH"))
print(len(out),"logged pulses,",len(exp),"sent")
