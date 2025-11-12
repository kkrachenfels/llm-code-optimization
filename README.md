## Notes

### Callgrind for hotspot profiling
Callgrind counts function calls and the CPU instructions executed within each call and builds a function callgraph

Resources
- <a href="https://valgrind.org/docs/manual/cl-manual.html">Callgrind documentation</a>
- <a href="https://web.stanford.edu/class/cs107/resources/callgrind">Other helpful notes</a>

Raw callgrind dumps aren't really human readable, so we need to get useful information from them:
- <a href="https://github.com/jrfonseca/gprof2dot">gprof2dot</a>: creates callstack visualization
- <a href="https://man7.org/linux/man-pages/man1/callgrind_annotate.1.html">callgrind_annotate</a>: annotates the dump, various CLI parameters can be provided to specify how to sort and how verbose to print information

### Ctags for tagging the project/program

<a href="https://docs.ctags.io/en/latest/man/ctags.1.html">ctags documentation</a>

ctags can 'tag' all the functions, declarations, and members in a program to help us locate them in the source code after we've run callgrind... this can helps us find the specific code snippets to pass into the LLMs downstream


### Code2Prompt for Baseline & Checking codebase token size

Noticed that the VSCode extension agent I was using via Roo Code would bug out on larger projects and throw context window error limits, so I wanted to check that.

<a href="https://github.com/mufeedvh/code2prompt">Code2Prompt Github</a>

Using a baseline template for code optimization without specific hotspot information (i.e. just throw the entire codebase at the LLM/agent as a prompt):

```code2prompt datasets/quantpp/ --include "*.cpp,*.hpp,*.c,*.h" -t=./baseline_optimization.hbs --line-numbers --output-file=prompts/files_no_hotspots.md```

This prints the token count from all C/C++ source files, and saves the prompt containing the entire codebase to a prompt file. 


### Current ideas and exploration

- Compare results across project difficulty/size levels, i.e. a small/'easy' project, medium, and big project. 
- Investigate and compare various levels of information passed into the LLM or agent:
    1. Provide the hotspot source code only, and ask the LLM to optimize that function.
    2. Provide the callstack trace to and out of the hotspot and ask the LLM to optimize the hotspot, and/or any of its caller functions, and/or any of its called functions.
        - Provide 1 hotspot at a time vs provide all/many hotpsots in one iteration
    3. Provide the full source code for the project, without any Callgrind and hotspot information. Prompt the model to make optimizations based on a prompt template.
    4. Provide the full source code for the project, but also provide more raw Callgrind hotspot information. I.e. provide the full set of callstacks and leave the LLM to pick out significant hotspots. Prompt the model to then optimize based on this.
    5. Provide the full source code for the project, but also provide a limited number of hotspot callstack traces at a time, and ask the model to optimize on any of these.
        - Can also explore if there's a tradeoff between # of hotspots provided and improved at a time; i.e. would it be better to ask the model to improve 1 hotspot at a time, provide the next hotspot, etc, or to give it 10 hotspots and just tell it to modify the code at once? (e.g. similar to breaking this down/CoT approach)


Currently using the following projects 
- quantpp: easy, straightforward optimizations 
- aes: less straightforward optimization
- tinyxml: already coded/touted to be incredibly fast, would be surprised if much targed optimization works on this

script order
- collect_tags.py [-d SOURCE_DIR]
- callgrind.py [-d PROJECT_DIR] [-e EXE] [-a [EXE_ARGS ...]] [-n TOP_N] [-i INCLUDE_THRESHOLD] [-v]
- (optional) prepare_prompts.py [-d PROJECT_DIR]
    - to look at/store the prompt
- llm_target_hotspots.py [-h] [-d PROJECT_DIR] [-n PROMPT_NO]
    - just uses prepare_prompts.py under the hood
