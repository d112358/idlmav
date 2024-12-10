"""
This script checks whether X11 forwarding has been configured 
correctly to display matplotlib figures when working in a 
WSL+VSCode environment on Windows.

This is only required to view matplotlib figures when running ".py"
scripts from this library directly. 
* Figures created from notebooks will be displayed inline in the 
  notebook and do not require X11 forwarding
* Plotly figures display in either a notebook or a browser and
  therefore also do not require X11 forwarding. See 
  "test_plotly_browser_setup.py"

For more setup details, refer to "setup_vscode_wsl.ipynb"
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

Year  = [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
Value = [9.8, 12, 8, 7.2, 6.9, 7, 6.5, 6.2, 5.5, 6.3]

plt.plot(Year, Value)
plt.title('Value over time')
plt.xlabel('Year')
plt.ylabel('Value')
plt.show()