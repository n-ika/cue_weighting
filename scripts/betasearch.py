import os
import re
import csv
import glob

LOG_DIR    = "/fs/clip-realspeech/projects/speech_in_noise/me_final/tmp/mdl/out"
OUTPUT_CSV = "/fs/clip-realspeech/projects/speech_in_noise/me_final/outputs/dec_beta_results.csv"

TARGET_EPOCHS = {20, 40, 60, 80, 100}

# Regex patterns
namespace_pattern = re.compile(
    r"Namespace\(.*?beta=([\deE+\-\.]+).*?type_input='(\w+)'.*?sim_num=(\d+)"
    r".*?only_ae_eh=(\w+).*?dec=(\w+)"
)
epoch_pattern = re.compile(
    r"Epoch (\d+)/\d+, Loss: ([-\d\.]+), Recon Loss: ([-\d\.]+), KL Loss: ([-\d\.]+)"
)

rows = []

for filepath in sorted(glob.glob(os.path.join(LOG_DIR, "*.txt"))):
    filename = os.path.basename(filepath)

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Parse Namespace line
    beta = type_input = sim_num = only_ae_eh = dec = None
    for line in lines:
        ns_match = namespace_pattern.search(line)
        if ns_match:
            beta       = float(ns_match.group(1))
            type_input = ns_match.group(2)
            sim_num    = int(ns_match.group(3))
            only_ae_eh = ns_match.group(4) == "True"
            dec        = ns_match.group(5) == "True"
            break

    if beta is None:
        print(f"[SKIP] No Namespace found: {filename}")
        continue

    # Collect losses at each target epoch
    epoch_data = {}
    for line in lines:
        ep_match = epoch_pattern.search(line)
        if ep_match:
            ep = int(ep_match.group(1))
            if ep in TARGET_EPOCHS:
                epoch_data[ep] = {
                    "total_loss": float(ep_match.group(2)),
                    "recon_loss": float(ep_match.group(3)),
                    "kl_loss":    float(ep_match.group(4)),
                }

    if not epoch_data:
        print(f"[SKIP] No target epoch lines found: {filename}")
        continue

    for ep in sorted(epoch_data.keys()):
        rows.append({
            "file":       filename,
            "beta":       beta,
            "type_input": type_input,
            "sim_num":    sim_num,
            "only_ae_eh": only_ae_eh,
            "dec":        dec,
            "epoch":      ep,
            "total_loss": epoch_data[ep]["total_loss"],
            "recon_loss": epoch_data[ep]["recon_loss"],
            "kl_loss":    epoch_data[ep]["kl_loss"],
        })

    found_epochs = sorted(epoch_data.keys())
    print(f"[OK] {filename} — epochs found: {found_epochs}")

if rows:
    fieldnames = ["file", "beta", "type_input", "sim_num", "only_ae_eh",
                  "dec", "epoch", "total_loss", "recon_loss", "kl_loss"]
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {OUTPUT_CSV}")
else:
    print("No valid log files found.")