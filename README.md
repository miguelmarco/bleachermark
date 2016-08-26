# bleachermark
A pipeline benchmarking framework for python

## Interface:

Two classes:

 - Benchmark:
   - Created from a list of callable objects (typically functions), each one accepting as input the output of the previous one.
   - The first callable accepts as input some parameters (running instance) and acts as a data generator for the rest of the pipeline.
   - Exposes a .run() method that runs the benchmark and returns the timings and values produceb by each piece.
- Bleachermark:
  - Created from a list of benchmarks
  - Handles the running of them in a smart way (parallelizing, distributing times and so on)
  - Fetches the data provided by each run of each benchmark, and stores it
  - Can do simple statistics over the stored data, or return the data to be processed by more specialized tools (pandas?)
  - Allows interrupt and resume gracefully.

