#!/usr/bin/env python3
"""
Structural test for DreamFitUnifiedV2 implementation
Tests file structure and basic syntax without requiring dependencies
"""

import os
import ast
from pathlib import Path


def check_file_exists(filepath):
    """Check if a file exists"""
    return os.path.exists(filepath)


def check_python_syntax(filepath):
    """Check if Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)


def test_file_structure():
    """Test that all required files exist"""
    print("Testing file structure...")
    
    required_files = [
        "nodes/dreamfit_unified_v2.py",
        "dreamfit_types.py",
        "utils/debug_visualization.py",
        "utils/__init__.py",
        "workflows/dreamfit_unified_v2_workflow.json",
        "__init__.py"
    ]
    
    all_exist = True
    for filepath in required_files:
        if check_file_exists(filepath):
            print(f"✓ {filepath} exists")
        else:
            print(f"✗ {filepath} missing")
            all_exist = False
    
    return all_exist


def test_python_syntax():
    """Test that all Python files have valid syntax"""
    print("\nTesting Python syntax...")
    
    python_files = [
        "nodes/dreamfit_unified_v2.py",
        "dreamfit_types.py",
        "utils/debug_visualization.py",
        "utils/__init__.py",
        "__init__.py"
    ]
    
    all_valid = True
    for filepath in python_files:
        if check_file_exists(filepath):
            valid, error = check_python_syntax(filepath)
            if valid:
                print(f"✓ {filepath} has valid syntax")
            else:
                print(f"✗ {filepath} has syntax error: {error}")
                all_valid = False
        else:
            print(f"✗ {filepath} not found")
            all_valid = False
    
    return all_valid


def test_node_registration():
    """Test that DreamFitUnifiedV2 is registered in __init__.py"""
    print("\nTesting node registration...")
    
    init_path = "__init__.py"
    if not check_file_exists(init_path):
        print("✗ __init__.py not found")
        return False
    
    with open(init_path, 'r') as f:
        content = f.read()
    
    checks = [
        ("Import statement", "from .nodes.dreamfit_unified_v2 import DreamFitUnifiedV2"),
        ("NODE_CLASS_MAPPINGS", '"DreamFitUnifiedV2": DreamFitUnifiedV2'),
        ("NODE_DISPLAY_NAME_MAPPINGS", '"DreamFitUnifiedV2": "DreamFit Unified V2"')
    ]
    
    all_found = True
    for check_name, check_string in checks:
        if check_string in content:
            print(f"✓ {check_name} found")
        else:
            print(f"✗ {check_name} not found")
            all_found = False
    
    return all_found


def test_workflow_json():
    """Test that workflow JSON is valid"""
    print("\nTesting workflow JSON...")
    
    import json
    workflow_path = "workflows/dreamfit_unified_v2_workflow.json"
    
    if not check_file_exists(workflow_path):
        print("✗ Workflow file not found")
        return False
    
    try:
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
        print("✓ Workflow JSON is valid")
        
        # Check for DreamFitUnifiedV2 node
        has_unified_v2 = False
        for node in workflow.get("nodes", []):
            if node.get("type") == "DreamFitUnifiedV2":
                has_unified_v2 = True
                break
        
        if has_unified_v2:
            print("✓ Workflow contains DreamFitUnifiedV2 node")
        else:
            print("✗ Workflow doesn't contain DreamFitUnifiedV2 node")
            return False
        
        return True
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}")
        return False


def test_class_structure():
    """Test the basic structure of DreamFitUnifiedV2 class"""
    print("\nTesting class structure...")
    
    filepath = "nodes/dreamfit_unified_v2.py"
    if not check_file_exists(filepath):
        print("✗ File not found")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse the AST
    try:
        tree = ast.parse(content)
    except SyntaxError:
        print("✗ Syntax error in file")
        return False
    
    # Find the DreamFitUnifiedV2 class
    class_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DreamFitUnifiedV2":
            class_found = True
            print("✓ DreamFitUnifiedV2 class found")
            
            # Check for required methods
            methods = [method.name for method in node.body if isinstance(method, ast.FunctionDef)]
            required_methods = ["INPUT_TYPES", "process"]
            
            for method in required_methods:
                if method in methods:
                    print(f"✓ {method} method defined")
                else:
                    print(f"✗ {method} method missing")
                    return False
            
            # Check for class variables
            class_vars = []
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            class_vars.append(target.id)
            
            required_vars = ["RETURN_TYPES", "FUNCTION", "CATEGORY"]
            for var in required_vars:
                if var in class_vars:
                    print(f"✓ {var} defined")
                else:
                    print(f"✗ {var} missing")
            
            break
    
    if not class_found:
        print("✗ DreamFitUnifiedV2 class not found")
        return False
    
    return True


def main():
    """Run all structural tests"""
    print("=== DreamFitUnifiedV2 Structural Test ===\n")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
        ("Node Registration", test_node_registration),
        ("Workflow JSON", test_workflow_json),
        ("Class Structure", test_class_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
        except Exception as e:
            print(f"✗ Test error: {e}")
            result = False
        results.append((test_name, result))
        print()
    
    print("=== Test Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✨ All structural tests passed! DreamFitUnifiedV2 implementation structure is correct.")
        print("\nNote: Runtime testing requires ComfyUI environment with torch and other dependencies.")
    else:
        print("\n⚠️  Some structural tests failed. Please review the implementation.")


if __name__ == "__main__":
    main()