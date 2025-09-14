import os

def get_project_paths():
    """
    Returns a dictionary with paths to important directories in the project.
    
    Returns:
        dict: Dictionary with the following keys:
            - parent_dir: Parent directory of the code folder
            - data_dir: Path to the Data directory
            - plots_dir: Path to the Output/Plots directory
            - tables_dir: Path to the Output/Tables directory
    """
    # Get the parent directory of the current file
    code_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(code_dir)
    
    # Define paths relative to the parent directory
    data_dir = os.path.join(parent_dir, 'Data')
    plots_dir = os.path.join(parent_dir, 'Output', 'Plots')
    tables_dir = os.path.join(parent_dir, 'Output', 'Tables')
    
    return {
        'parent_dir': parent_dir,
        'data': data_dir,
        'plots': plots_dir,
        'tables': tables_dir
    }