"""
inspect_app.py — Standalone experiment (separate from the main TRINETRA
pipeline). Launches and inspects the window/control structure of a
Windows application, so we can figure out what a UI-automation script
would need to target (button names, text fields, etc.) before writing
one.

This does NOT modify, decompile, or extract anything from the app — it
only reads window titles and control labels that are already visible on
screen, the same information Windows' own Accessibility APIs expose to
screen readers.

Usage:
  # Launch the app directly and inspect its main window:
  python inspect_app.py --launch "C:\\Path\\To\\App.exe"

  # OR, if you already opened it manually, just list open windows:
  python inspect_app.py

  # OR inspect a specific already-open window by title:
  python inspect_app.py --title "QR Code Reader"
"""

import sys
import time
import argparse

try:
    from pywinauto import Desktop
    from pywinauto.application import Application
except ImportError:
    print("pywinauto not installed. Run: python -m pip install pywinauto")
    sys.exit(1)


def launch_app(exe_path, wait_seconds=3):
    """Launches the app and returns its Application object once it's up."""
    print(f"Launching: {exe_path}")
    app = Application(backend="uia").start(exe_path)
    print(f"Waiting {wait_seconds}s for the app window to appear...")
    time.sleep(wait_seconds)
    return app


def list_open_windows():
    print("Currently open top-level windows:\n")
    windows = Desktop(backend="uia").windows()
    for i, w in enumerate(windows):
        try:
            title = w.window_text()
            if title.strip():
                print(f"  [{i}] '{title}'  (class: {w.friendly_class_name()})")
        except Exception:
            continue


def dump_window_controls(title_substring):
    print(f"Looking for a window with title containing: '{title_substring}'\n")
    windows = Desktop(backend="uia").windows()
    target = None
    for w in windows:
        try:
            if title_substring.lower() in w.window_text().lower():
                target = w
                break
        except Exception:
            continue

    if target is None:
        print("No matching window found. Is the app open? Check the exact title with no args first.")
        return

    print(f"Found: '{target.window_text()}'\n")
    print("Control tree (this shows every button/field/label the app exposes):\n")
    target.print_control_identifiers(depth=6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", help="Full path to the .exe to launch directly")
    parser.add_argument("--title", help="Title (or substring) of an already-open window to inspect")
    parser.add_argument("--wait", type=int, default=3, help="Seconds to wait after launching before inspecting (default 3)")
    args = parser.parse_args()

    if args.launch:
        launch_app(args.launch, wait_seconds=args.wait)
        print("\nNow listing open windows (the app should be among them):\n")
        list_open_windows()
        print("\nRe-run with --title \"<exact title from above>\" to see its full control tree.")
    elif args.title:
        dump_window_controls(args.title)
    else:
        list_open_windows()
        print("\nRun again with --title \"<window title>\" to see its controls, or")
        print('--launch "C:\\path\\to\\app.exe" to launch it directly first.')

