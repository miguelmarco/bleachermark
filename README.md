# bleachermark
A pipeline, parallel, interleaving benchmarking framework for Python

## Interface:

Three classes:

 - Benchmark:
   - Created from a list of callable objects (typically functions), each one accepting as input the output of the previous one.
   - The first callable accepts as input some parameters (running instance) and usually acts as a data generator for the rest of the pipeline.
   - Exposes a .run() method that runs the benchmark and returns the timings and values produced by each part of the pipeline.
 - Bleachermark:
   - Essentially a collection of benchmarks
   - Calls a runner to run them, and gets the results.
   - Store the timings and data for each run of each benchmark
   - Can do simple statistics over the stored data, or return the data to be processed by more specialized tools (pandas?)
   - Allows interrupt and resume gracefully.
 - Runners:
   - Meant to be created by the bleachermark
   - Runs the benchmark according to a strategy specific for each runner.
   - Returns the results according to the iterator protocol.

## Dependencies:

  - time


