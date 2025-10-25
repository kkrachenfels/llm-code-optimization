import subprocess

def get_program_runtime(executable_path, args=[]):
    """
    Runs the given binary and returns its runtime in seconds.
    
    Parameters:
    - binary_path (str): The path to the binary to be executed.
    - args (list): Additional arguments to pass to the binary.
    
    Returns:
    - float: The runtime of the program in seconds.
    """
    import time
    start_time = time.time()
    subprocess.run([executable_path] + args)
    end_time = time.time()
    return end_time - start_time

def remake_executable():
    """
    *** assumes a makefile is present ***
    Remakes the executable by running the provided make command.
    
    Parameters:
    - make_command (list): The command to run for remaking the executable.
    """
    subprocess.run(['make', 'clean'])
    subprocess.run(['make', 'all'])
    ## and need to add error handling
    ## e.g. if updated code breaks & doesn't compile


def apply_llm_code_changes(file_path, llm_response):
    """
    Applies code changes suggested by the LLM to the specified file.
    """
    pass


