#!/usr/bin/env python3
"""
Simple type and error checker for DreamFit code
Checks for common issues like None values, type mismatches, etc.
"""

import ast
import sys
from pathlib import Path

class SimpleTypeChecker(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.errors = []
        self.warnings = []
        
    def visit_Attribute(self, node):
        # Check for potential None attribute access
        if isinstance(node.value, ast.Name):
            if 'noise' in node.value.id.lower():
                self.warnings.append(
                    f"{self.filename}:{node.lineno}: Warning: Accessing attribute on '{node.value.id}' - ensure it's not None"
                )
        self.generic_visit(node)
        
    def visit_Call(self, node):
        # Check for functions that might receive None
        if hasattr(node.func, 'attr'):
            if node.func.attr in ['shape', 'size', 'dtype']:
                self.warnings.append(
                    f"{self.filename}:{node.lineno}: Warning: Calling .{node.func.attr} - ensure object is not None"
                )
                
        # Check for None in keyword arguments
        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Constant) and keyword.value.value is None:
                self.warnings.append(
                    f"{self.filename}:{node.lineno}: Warning: Passing None for '{keyword.arg}' parameter"
                )
                
        self.generic_visit(node)
        
    def check_file(self, filepath):
        with open(filepath, 'r') as f:
            try:
                tree = ast.parse(f.read())
                self.visit(tree)
            except SyntaxError as e:
                self.errors.append(f"{filepath}: Syntax Error: {e}")
                
        return self.errors, self.warnings

def main():
    # Check all Python files in the project
    root = Path(__file__).parent
    files_to_check = [
        root / "nodes" / "dreamfit_sampler_v2.py",
        root / "dreamfit_core" / "models" / "dreamfit_model_wrapper.py",
        root / "nodes" / "dreamfit_unified_v2.py",
    ]
    
    all_errors = []
    all_warnings = []
    
    for filepath in files_to_check:
        if filepath.exists():
            checker = SimpleTypeChecker(str(filepath))
            errors, warnings = checker.check_file(filepath)
            all_errors.extend(errors)
            all_warnings.extend(warnings)
            
    # Print results
    if all_errors:
        print("ERRORS:")
        for error in all_errors:
            print(f"  {error}")
            
    if all_warnings:
        print("\nWARNINGS:")
        for warning in all_warnings:
            print(f"  {warning}")
            
    if not all_errors and not all_warnings:
        print("No issues found!")
        
    return len(all_errors)

if __name__ == "__main__":
    sys.exit(main())