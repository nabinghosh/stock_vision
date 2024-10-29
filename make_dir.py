import os

def get_directory_structure():
    directory = os.getcwd()
    return _get_directory_structure_recursive(directory)

def _get_directory_structure_recursive(directory):
    structure = {}
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and item not in ['.git', '__pycache__', 'venv']:
            structure[item] = _get_directory_structure_recursive(item_path)
        elif os.path.isfile(item_path) and item.endswith(('.py', '.txt', '.md', '.json')):
            structure[item] = None
    return structure
directory_structure = get_directory_structure()
print(directory_structure)