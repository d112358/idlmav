from bs4 import BeautifulSoup
import jsbeautifier
import json
from IPython.display import display, Markdown

def pretty_html_js(html):
    soup = BeautifulSoup(html, "html.parser")

    # Customize JS indentation size
    js_options = jsbeautifier.default_options()
    js_options.indent_size = 4  # Customize the indentation size

    # Loop through <script> elements
    for script in soup.find_all("script"):
        if script.string:
            pretty_js = jsbeautifier.beautify(script.string, js_options)
            script.string.replace_with(pretty_js)

    # Prettify the entire HTML
    return soup.prettify()


def get_nb_cell_output_sizes(notebook_path):
    """
    Sometimes the kernel becomes slow because a notebook's output is too large.
    This function helps detect which cells in a notebook has large outputs
    The return value is a list of tuples (index, output_size, cell_type, first_line)
    """

    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Analyze each cell
    result = []
    for i, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            outputs = cell.get("outputs", [])
            output_size = sum(len(str(output)) for output in outputs)
            code_lines = cell.get("source", [])
            first_line = ''
            if code_lines: first_line = code_lines[0].strip()
            result.append((i, output_size, "code", first_line))
        elif cell.get("cell_type") == "markdown":
            md_lines = cell.get("source", [])
            first_line = ''
            if md_lines: first_line = md_lines[0].strip()
            result.append((i, 0, "markdown", first_line))
    return result

def display_nb_cell_output_sizes(notebook_path):
    """
    Sometimes the kernel becomes slow because a notebook's output is too large.
    This function helps detect which cells in a notebook has large outputs
    The return value is a list of tuples (index, output_size, code_snippet)
    """

    lines = ['|Cell number|Output size|Cell type  |First line  |',
             '|-----------|-----------|-----------|------------|']
    for index, size, cell_type, first_line in get_nb_cell_output_sizes(notebook_path):
        lines.append(f'|{index}|{size}|{cell_type}|{first_line}|')
    display(Markdown('\n'.join(lines)))