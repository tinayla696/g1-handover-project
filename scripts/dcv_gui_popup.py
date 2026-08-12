import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Open an Isaac Sim GUI window in the DCV X display.")
parser.add_argument("--seconds", type=float, default=30.0, help="Keep the GUI open for this many seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    print("Isaac Sim GUI popup is running in the DCV display.", flush=True)
    print(f"Keeping the window open for {args_cli.seconds:.1f} seconds.", flush=True)
    deadline = time.monotonic() + max(args_cli.seconds, 0.0)
    while simulation_app.is_running() and time.monotonic() < deadline:
        simulation_app.update()
finally:
    simulation_app.close()
