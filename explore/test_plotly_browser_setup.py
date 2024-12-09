"""
This script checks whether the BROWSER environment has been 
configured correctly to display plotly figures when working 
in a WSL+VSCode environment on Windows.

This is only required to view plotly figures when running ".py" 
scripts from this library directly. Figures created from notebooks
will be displayed inline in the notebook and do not require this 
setup.

For more setup details, refer to "environment_setup.ipynb"
"""
import plotly.io as pio
import plotly.graph_objects as go

# Sample Data
data = {
    'name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T'],
    'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'y': [10, 15, 13, 17, 11, 13, 18, 14, 12, 19, 17, 15, 10, 19, 18, 13, 12, 17, 14, 15]
}

# Scatter Plot with Plotly
# fig = px.scatter(data, x='x', y='y', hover_name='name')
fig = go.FigureWidget(
    data=go.Scatter(
        x=data['x'], 
        y=data['y'], 
        mode='markers',
        hovertemplate=('name: %{customdata}'),
        customdata=data['name']  # Fields 5 and 6 for custom hover data
    )
)

# Display available renderers
available_renderers = list(pio.renderers)
print(f'Available renderers: {", ".join(available_renderers)}')
print('')

# Display the figure
fig.show()