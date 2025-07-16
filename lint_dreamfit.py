#!/usr/bin/env python3
"""
Simple linter for DreamFit code to catch common ComfyUI integration issues
"""

import ast
import os
import sys
from typing import List, Tuple

class DreamFitLinter(ast.NodeVisitor):
    def __init__(self):
        self.issues = []
        self.current_file = ""
    
    def add_issue(self, node, message, severity="ERROR"):
        self.issues.append({
            'file': self.current_file,
            'line': node.lineno,
            'column': node.col_offset,
            'message': message,
            'severity': severity
        })
    
    def visit_Call(self, node):
        # Check for direct ModelPatcher calls
        if isinstance(node.func, ast.Name):
            if node.func.id == 'model' and len(node.args) >= 2:
                # Look for model(x, timestep) patterns
                self.add_issue(node, 
                    "Possible direct model call - use model.apply_model() or check for ModelPatcher",
                    "WARNING")
        
        # Check for undefined variable usage
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name.startswith('enhanced_') and not hasattr(self, '_defined_enhanced_vars'):
                self.add_issue(node,
                    f"Variable '{func_name}' may be used before definition",
                    "ERROR")
        
        self.generic_visit(node)
    
    def visit_Subscript(self, node):
        # Check for tuple/dict access issues
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            # String indexing into what might be a tuple
            if isinstance(node.value, ast.Name):
                self.add_issue(node,
                    f"String index '{node.slice.value}' on variable '{node.value.id}' - ensure it's a dict not tuple",
                    "WARNING")
        
        self.generic_visit(node)
    
    def visit_Return(self, node):
        # Check return format for samplers
        if isinstance(node.value, ast.Tuple) and len(node.value.elts) == 1:
            # Check if returning (samples,) format
            if isinstance(node.value.elts[0], ast.Name):
                var_name = node.value.elts[0].id
                if 'sample' in var_name.lower():
                    self.add_issue(node,
                        f"Returning ({var_name},) - ensure {var_name} is a dict with 'samples' key",
                        "INFO")
        
        self.generic_visit(node)
    
    def visit_Import(self, node):
        # Check for problematic imports
        for alias in node.names:
            if alias.name == 'comfy.sample_as':
                self.add_issue(node,
                    "Import 'comfy.sample_as' does not exist in ComfyUI",
                    "ERROR")
        
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        # Track variable definitions
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            if var_name.startswith('enhanced_'):
                if not hasattr(self, '_defined_enhanced_vars'):
                    self._defined_enhanced_vars = set()
                self._defined_enhanced_vars.add(var_name)
        
        self.generic_visit(node)

def lint_file(filepath: str) -> List[dict]:
    """Lint a single Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=filepath)
        linter = DreamFitLinter()
        linter.current_file = filepath
        linter.visit(tree)
        
        return linter.issues
    
    except SyntaxError as e:
        return [{
            'file': filepath,
            'line': e.lineno or 0,
            'column': e.offset or 0,
            'message': f"Syntax error: {e.msg}",
            'severity': 'ERROR'
        }]
    except Exception as e:
        return [{
            'file': filepath,
            'line': 0,
            'column': 0,
            'message': f"Failed to parse file: {e}",
            'severity': 'ERROR'
        }]

def lint_directory(directory: str) -> List[dict]:
    """Lint all Python files in a directory"""
    all_issues = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                issues = lint_file(filepath)
                all_issues.extend(issues)
    
    return all_issues

def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "."
    
    if os.path.isfile(target):
        issues = lint_file(target)
    elif os.path.isdir(target):
        issues = lint_directory(target)
    else:
        print(f"Error: {target} is not a valid file or directory")
        sys.exit(1)
    
    # Group issues by severity
    errors = [i for i in issues if i['severity'] == 'ERROR']
    warnings = [i for i in issues if i['severity'] == 'WARNING']
    info = [i for i in issues if i['severity'] == 'INFO']
    
    # Print results
    for issue in errors + warnings + info:
        print(f"{issue['file']}:{issue['line']}:{issue['column']}: {issue['severity']}: {issue['message']}")
    
    print(f"\nSummary: {len(errors)} errors, {len(warnings)} warnings, {len(info)} info")
    
    if errors:
        sys.exit(1)

if __name__ == "__main__":
    main()