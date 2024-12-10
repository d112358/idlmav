"""
This script reproduces an issue that was observed in an environment
consisting of VSCode, WSL and the Python debugger.

Symptom: the debugger would hang at the IDLMAV import statement

Summary of call stack:
* `mavutils` calls `list(pio.renderers)`
* This imports `webbrowser` and calls `webbrowser.get()`
* This executes a command `xdg-settings get default-web-browser`
  in a subprocess and waits for the result on stdout
* The command does not return a result and the wait is indefinite

Resolution:
* This issue only occurs when the following two conditions are both
  true:
  - The `DISPLAY` environment variable is set for X11 forwarding
  - The X-server is not running on the Windows side
* It seems the indefinite wait is the Linux X-client waiting for
  the Windows X-server
  - The wait is also inside the `if os.environ.get("DISPLAY")`
    branch in "webbrowser.py"
* To resolve, perform either of 2 actions:
  - Run the X-server on the Windows side 
    (see "environment_setup.ipynb")
  - Don't set the `DISPLAY` environment variable if not required
    (see "launch.json")
"""

import webbrowser

print('Getting webbrowser')
b = webbrowser.get()
print('Done')
print(f'Default browser: {b.name}')
