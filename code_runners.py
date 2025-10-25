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

def remake_executable(make_dir):
    """
    *** assumes a makefile is present ***
    Remakes the executable by running the provided make command.
    
    Parameters:
    - make_dir (str): The directory containing the Makefile.
    """
    make_dir = ''
    subprocess.run(['make', 'clean'], cwd=make_dir)
    subprocess.run(['make', 'all'], cwd=make_dir)
    ## and need to add error handling
    ## e.g. if updated code breaks & doesn't compile


def apply_llm_code_changes(file_path, llm_response):
    """
    Applies code changes suggested by the LLM to the specified file.
    """
    pass


