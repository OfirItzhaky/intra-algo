from IPython.display import display, HTML
import ipywidgets as widgets

def create_snapshot_counter():
    """
    Create a simple snapshot counter widget for Jupyter notebooks.
    This function creates an interface where users can paste snapshots
    and count how many they've pasted.
    
    Returns:
        None - Displays the counter interface in the notebook
    """
    # Create a text area widget for pasting snapshots
    text_area = widgets.Textarea(
        placeholder='Paste your snapshots here...',
        layout=widgets.Layout(width='100%', height='200px')
    )

    # Create a button to count the snapshots
    button = widgets.Button(
        description='Count Snapshots',
        button_style='primary'
    )

    # Create an output area for the result
    output = widgets.Output()

    # Define a function to count the snapshots
    def count_snapshots(b):
        with output:
            output.clear_output()
            content = text_area.value.strip()
            
            if not content:
                print("No snapshots pasted.")
                return
                
            # A simple heuristic to count snapshots - assuming each snapshot 
            # is separated by at least one blank line
            snapshots = [s for s in content.split('\n\n') if s.strip()]
            count = len(snapshots)
            
            if count == 1:
                print(f"You pasted 1 snapshot.")
            else:
                print(f"You pasted {count} snapshots.")

    # Connect the button to the function
    button.on_click(count_snapshots)

    # Display the widgets
    display(HTML("<h3>Snapshot Counter</h3>"))
    display(HTML("<p>Paste your snapshots below and click the button to count them:</p>"))
    display(text_area)
    display(button)
    display(output)
    
    return None

# Alternative version that uses standard input if widgets don't render properly
def count_snapshots_input():
    """
    A simpler version of the snapshot counter that uses standard input.
    This can be used if the widgets don't render properly in the user's environment.
    
    Returns:
        int: The number of snapshots detected
    """
    print("Please paste your snapshots below and press Ctrl+D (or Ctrl+Z on Windows) when finished:")
    
    snapshots = []
    try:
        while True:
            line = input()
            snapshots.append(line)
    except EOFError:
        # End of input
        pass
    
    # Process the input to count snapshots
    text = '\n'.join(snapshots)
    snapshot_count = len([s for s in text.split('\n\n') if s.strip()])
    
    if snapshot_count == 1:
        print(f"You pasted 1 snapshot.")
    else:
        print(f"You pasted {snapshot_count} snapshots.")
    
    return snapshot_count

# Simple cell version that can be edited directly
def count_snapshots_simple(snapshots_text=None):
    """
    The simplest version of snapshot counting - simply edit the text in the variable.
    
    Args:
        snapshots_text (str, optional): The text containing snapshots to count.
            If None, uses the example text below.
            
    Returns:
        int: The number of snapshots detected
    """
    if snapshots_text is None:
        # Replace this with your snapshots
        snapshots_text = """
        Snapshot 1
        Data for first snapshot
        
        Snapshot 2
        Data for second snapshot
        
        Snapshot 3
        Data for third snapshot
        """
    
    # Count snapshots (assuming each is separated by a blank line)
    snapshot_count = len([s for s in snapshots_text.split('\n\n') if s.strip()])
    
    if snapshot_count == 1:
        print(f"You pasted 1 snapshot.")
    else:
        print(f"You pasted {snapshot_count} snapshots.")
    
    return snapshot_count 