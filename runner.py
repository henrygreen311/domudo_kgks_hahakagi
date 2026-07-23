import subprocess
import sys

# ==========================
# Enable/Disable scripts
# ==========================
RUN_TRACKER = True
RUN_ANALYZER = True
RUN_TRADER = True

# ==========================
# Show logs for each script
# ==========================
SHOW_TRACKER_LOGS = False
SHOW_ANALYZER_LOGS = True
SHOW_TRADER_LOGS = True

processes = []


def start_script(filename, show_output=False):
    """Start a Python script."""

    if show_output:
        process = subprocess.Popen(
            [sys.executable, "-u", filename],  # Unbuffered output
        )
    else:
        process = subprocess.Popen(
            [sys.executable, "-u", filename],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print(f"✅ {filename} started.")
    processes.append(process)
    return process


tracker = None
analyzer = None
trader = None

if RUN_TRACKER:
    tracker = start_script(
        "tracker.py",
        show_output=SHOW_TRACKER_LOGS
    )

if RUN_ANALYZER:
    analyzer = start_script(
        "analyzer.py",
        show_output=SHOW_ANALYZER_LOGS
    )

if RUN_TRADER:
    trader = start_script(
        "trader.py",
        show_output=SHOW_TRADER_LOGS
    )

try:
    # Wait only for trader if it's running.
    if trader:
        trader.wait()
    else:
        # If trader is disabled, wait for any enabled process.
        for process in processes:
            process.wait()

finally:
    print("\nStopping background processes...")

    for process in processes:
        if process.poll() is None:
            process.terminate()

    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    print("All processes stopped.")