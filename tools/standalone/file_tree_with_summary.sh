import os
import ast

CODE_EXTENSIONS = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.rb', '.go', '.php'}

def is_code_file(filename):
    return any(filename.endswith(ext) for ext in CODE_EXTENSIONS)

def summarize_python(filepath):
    """Summarize a Python file using docstrings, comments, and AST."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        tree = ast.parse(source)
        summary = []
        # Module docstring
        module_doc = ast.get_docstring(tree)
        if module_doc:
            summary.append(f"Module docstring: {module_doc}")
        # Classes and functions
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node)
                summary.append(f"Class '{node.name}': {doc if doc else 'No docstring.'}")
            elif isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node)
                summary.append(f"Function '{node.name}': {doc if doc else 'No docstring.'}")
        # Fallback: first 10 comment lines
        if not summary:
            comments = [line.strip() for line in source.splitlines() if line.strip().startswith('#')]
            summary.extend(comments[:10])
        if not summary:
            summary = ["No descriptive comments or docstrings found."]
        return "\n".join(summary)
    except Exception as e:
        return f"Could not summarize Python file: {e}"

def summarize_other(filepath):
    """Summarize non-Python code files by extracting top comments and function/class names."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        summary = []
        # Top comments
        for line in lines[:30]:
            if line.strip().startswith(("//", "/*", "#")):
                summary.append(line.strip())
        # Function/class names
        for line in lines:
            if any(kw in line for kw in ['function ', 'def ', 'class ', 'public ', 'void ', 'interface ', 'export default']):
                summary.append(line.strip())
        if not summary:
            summary = ["No descriptive comments or docstrings found."]
        return "\n".join(summary[:10])
    except Exception as e:
        return f"Could not summarize file: {e}"

def summarize_code(filepath):
    if filepath.endswith('.py'):
        return summarize_python(filepath)
    else:
        return summarize_other(filepath)

def main(root='.'):
    import datetime
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    md_filename = f'file_tree_with_summary_{timestamp}.md'
    with open(md_filename, 'w', encoding='utf-8') as md:
        md.write(f'# File Tree with Code Summaries\n\nGenerated: {timestamp}\n\n')
        for dirpath, _, filenames in os.walk(root):
            code_files = [f for f in filenames if is_code_file(f)]
            if code_files:
                md.write(f'## {dirpath}\n\n')
                for filename in code_files:
                    filepath = os.path.join(dirpath, filename)
                    md.write(f'### {filename}\n\n')
                    md.write('```text\n')
                    summary = summarize_code(filepath)
                    md.write(summary)
                    md.write('\n```\n\n')
    print(f"Markdown file generated: {md_filename}")

if __name__ == '__main__':
    main()