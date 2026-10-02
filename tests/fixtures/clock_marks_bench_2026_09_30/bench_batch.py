import subprocess, time, json, os, math
PY="/home/hansrkid/econ_itd_tools/.venv/bin/python3"
env=dict(os.environ, EOS_IP="10.70.10.51", EOS_TZ_OFFSET_HOURS="1", EOS_AT_RETRY="0",
         EOS_MARKER_PED_SET="14", EOS_MARKER_PED_BEHIND="15", EOS_MARKER_PED_AHEAD="16",
         EOS_JSONL_LOG=os.path.abspath("bench.jsonl"), EOS_MAX_CORRECTABLE_DRIFT="300")
KEEP=("Controller is","Set ","Waiting","verified","bracket","Pulsed","✓","ERROR","WARNING","Round")
def run(label, shift, *extra):
    t0=time.time()
    p=subprocess.run([PY,"-u","emu_run.py",str(shift),*extra],env=env,capture_output=True,text=True,timeout=400)
    rec=json.loads(open("bench.jsonl").read().splitlines()[-1])
    print(f"\n=== {label}: host shift {shift:+.1f}s  rc={p.returncode}  {time.time()-t0:.1f}s  result={rec.get('result')}")
    for l in (p.stdout+p.stderr).splitlines():
        if any(k in l for k in KEEP): print("   ", l[10:])
    return rec
r=run("B1 drift-check", 0.0, "--drift-check")
r=run("B2 set, bench appears +1.8 ahead", -2.3)
r=run("B3 set back to real time", 0.0)
r=run("B4 set, bench appears ~40 s behind", 40.0)
r=run("B5 restore to real time", 0.0)
