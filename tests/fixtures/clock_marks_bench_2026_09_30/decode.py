import zlib, struct, sys, glob, re
for f in sorted(glob.glob("datz/*.datZ")):
    raw=zlib.decompress(open(f,"rb").read()); m=raw.find(b"Phases in use:")
    hdr=re.search(rb"Controller Data Log Beginning:,([\d/]+),([\d:.]+)", raw[:m]).group(2).decode()
    piu=raw[m:raw.find(b"\n",m)].decode()
    p=raw[raw.find(b"\n",m)+1:]; n=len(p)//4; r=struct.unpack(">"+"BBH"*n,p)
    print(f.split("/")[-1], hdr, piu)
    for c,pa,o in zip(r[0::3],r[1::3],r[2::3]):
        if c in (89,90,45) or (c in (81,82) and pa>60): print("   off %5.1f code %d param %d"%(o/10,c,pa))
