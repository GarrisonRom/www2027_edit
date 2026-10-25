"""Independent C1 transfer pilot: smoke, 114 updates, then frozen capability audit."""
import json
import subprocess
import sys
from pathlib import Path
import run_cloud_gsu_c2 as c

def main():
    root = c.WS / "exp/runs/exp014_cloud_c1/runs/batch_20261023_01"
    root.mkdir(parents=True, exist_ok=False)
    c.baseline.write(root / "config.json", {"variant": "C1", "contract": c.CONTRACT,
        "change": "No shared-direction span; retain component routing, feedback, F0 guard",
        "source_local_run": "exp010/batch_20261023_092800_Cgroup_restart1/C1",
        "interpretation": "Cross-model transfer, not same-model numerical replay"})
    (root / "patch.diff").write_bytes(subprocess.check_output(["git", "diff"], cwd=c.WS) if (c.WS/".git").exists() else b"Source snapshot archived in train directory; no git metadata on server.\n")
    done = []
    def run(label, command):
        c.baseline.write(root/"status.json", {"status":"running", "current":label, "completed":done})
        with (root/(label+".stdout.log")).open("x") as out, (root/(label+".stderr.log")).open("x") as err:
            subprocess.run(command, cwd=c.WS, stdout=out, stderr=err, check=True, timeout=21600)
        done.append(label)
    try:
        for key, model in c.MODELS:
            for smoke in (True, False):
                label=key+("_C1_smoke" if smoke else "_C1")
                cmd=[sys.executable,str(c.WS/"exp/train/run_cloud_gsu_c2.py"),"--variant","C1","--model",model,"--run_dir",str(root/label)]
                run(label,cmd+(["--smoke"] if smoke else []))
            label=key+"_C1"
            endpoint={"model_key":key,"model_path":model,"run_id":label,"method":"GSU-C1",
                "supervision":"F0","adapter_path":str(root/label/"adapter"),"origin":"exp014",
                "training_metrics_path":str(root/label/"metrics.json")}
            ep=root/(label+"_endpoint.json"); c.baseline.write(ep,endpoint)
            run(label+"_audit",["/root/autodl-tmp/venvs/lm_eval_0411_capability_20261023_v2/bin/python",
                str(c.WS/"exp/train/evaluate_cloud_capabilities.py"),"--endpoint_json",str(ep),
                "--run_dir",str(root/(label+"_audit")),"--wiki_revision","b08601e04326c79dfdd32d625aee71d232d685c3"])
        c.baseline.write(root/"status.json",{"status":"completed","completed":done})
    except BaseException:
        import traceback
        c.baseline.write(root/"status.json",{"status":"failed","completed":done,"traceback":traceback.format_exc()})
        raise

if __name__ == "__main__":
    main()
