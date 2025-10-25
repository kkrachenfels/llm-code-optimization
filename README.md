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


