import os
import re

files_to_update = [
    "AGENTS.md",
    "docs/ROADMAP.md",
    "src/atspm/analysis/detectors.py"
]

for filepath in files_to_update:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Case-sensitive and insensitive replacements

    # "Orphan Pulse" -> "Pulse"
    content = content.replace("Orphan Pulse", "Pulse")
    # "orphan pulse" -> "pulse"
    content = content.replace("orphan pulse", "pulse")
    # "orphan_pulse" -> "pulse"
    content = content.replace("orphan_pulse", "pulse")
    # "orphans" -> "pulses"
    content = content.replace("orphans", "pulses")
    # "Orphans" -> "Pulses"
    content = content.replace("Orphans", "Pulses")
    # "orphan" -> "pulse"
    content = content.replace("orphan", "pulse")
    # "Orphan" -> "Pulse"
    content = content.replace("Orphan", "Pulse")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print("Done")
