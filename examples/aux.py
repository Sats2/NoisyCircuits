import subprocess
import re

file = open("Memory Output.txt", "w")

output = subprocess.run(["/usr/bin/time", "-v", "python3", "test_simulator.py", "nothing"], capture_output=True, text=True)

out = output.stderr
file.write("\n\nBaseline memory usage (nothing):\n")
file.write(out)
file.write("\n\n")

match = re.search("Maximum resident set size \(kbytes\):\s+(\d+)", out)
if match:
    mem = int(match.group(1))
    print(f"Memory usage: {mem / 1024 / 1024} GB")
    file.write(f"Memory usage: {mem / 1024 / 1024} GB\n")

file.flush()

output = subprocess.run(["/usr/bin/time", "-v", "python3", "test_simulator.py", "dm"], capture_output=True, text=True)

out = output.stderr
file.write("\n\nMemory usage with Density Matrix:\n")
file.write(out)
file.write("\n\n")

match = re.search("Maximum resident set size \(kbytes\):\s+(\d+)", out)
if match:
    mem_dm = int(match.group(1))
    print(f"Memory usage: {(mem_dm - mem) / 1024 / 1024} GB")
    file.write(f"Memory usage: {(mem_dm - mem) / 1024 / 1024} GB\n")

file.flush()

output = subprocess.run(["/usr/bin/time", "-v", "python3", "test_simulator.py", "custom"], capture_output=True, text=True)

out = output.stderr
file.write("\n\nMemory usage with Custom Simulator:\n")
file.write(out)
file.write("\n\n")

match = re.search("Maximum resident set size \(kbytes\):\s+(\d+)", out)
if match:
    mem_custom = int(match.group(1))
    print(f"Memory usage: {(mem_custom - mem) / 1024 / 1024} GB")
    file.write(f"Memory usage: {(mem_custom - mem) / 1024 / 1024} GB\n")

file.flush()
file.close()