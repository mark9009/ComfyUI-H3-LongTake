"""Checks seam_match on a real clip: redo_one of clip 1 of an already rendered guide project
(same seed -> same clip, corrected at the seam), then measures colour and seam.
Usage: redo_seam.py <project> [seam_match]"""
import os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_style_tests as t

name = sys.argv[1]
seam = sys.argv[2] if len(sys.argv) > 2 else "color"
test = [x for x in t.TESTS if x[0] == name][0]
prompt = t.build(*test)
inp = prompt["7"]["inputs"]
inp.update({"mode": "redo_one", "redo_from_clip": 1, "seam_match": seam})
t.build = lambda *a: prompt  # run_one calls build(*args)
if not t.wait_server():
    sys.exit("server unreachable")
ok, err, dt = t.run_one(*test)
print("result:", "ok" if ok else "ERROR " + err, f"{dt / 60:.1f} min")
if ok:
    print(t.measure(name))
