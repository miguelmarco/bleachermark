# bleachermark
A pipeline, parallel, interleaving benchmarking framework for Python

## Interface:

Two classes:

 - Benchmark:
   - Created from a list of callable objects (typically functions), each one accepting as input the output of the previous one.
   - The first callable accepts as input some parameters (running instance) and usually acts as a data generator for the rest of the pipeline.
   - Exposes a .run() method that runs the benchmark and returns the timings and values produced by each part of the pipeline.
 - Bleachermark:
   - Essentially a collection of benchmarks
   - Handles the running of them in a smart way (parallelizing, interleaving runs, and so on)
   - Store the timings for each run of each benchmark
   - Fetches the data provided by each run of each benchmark, and stores it
   - Can do simple statistics over the stored data, or return the data to be processed by more specialized tools (pandas?)
   - Allows interrupt and resume gracefully.

## Dependencies:

  - time


