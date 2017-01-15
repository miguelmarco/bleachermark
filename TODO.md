# Johan notes

- The master thread is responsible for handling non-completed runs on
  interruption: the schedulers will assume that any run that was emitted will
  (eventually) be run. This allows using specific data instead of "runids" and
  relying on all of these being eventually run.

- Adaptive strategy := Balance how much time is spent in intervals of doubling sizes.


# Benchmark/Bleachermark design

- Having a Bleachermark with Benchmarks with semantically different pipelines doesn't make sense
- So disallow different lengths and different function labels
- So remove labels from Benchmark and put them on Bleachermark
- User message: Benchmark is mostly an internal thing. Bleachermark is the core external class.
- If so, should we store the data when a Bleachermark is used as an iterator?

# Parallelism

- Suggestions?
- Take a look at https://wiki.python.org/moin/ParallelProcessing.
- Decouple the SageMath parallel framework into a separate package
- It should be possible to make a speed plot across different no. of workers

# Interruption and Resume

- Works?
- How to integrate with parallelism?
- What did David do in #20684

# Out-of-the-Box Statistics

- Remove plural 's' from statistics names?
- Rename stdvs to std?
- Getters for a specific benchmark? (useful for plotting)


#  Other
- Write setup.py and so on to make this a pip installable package
- Decide on memory safety for Bleachermark.__add__ (how do we avoid copying data on add'ing)?


# User stories

- Coding theory decoding setting
    - Plot decoding speeds (x: noise/no. errors, y: speed)
    - Plot decoding success (x: noise/no. errors, y: #correct/#trials)
- Cost of sorting algorithms
    - Plot sorting speed of two sorting algorithms (x: log of list size, y: log of time)
    - Guess asymptotic complexity
    - Adaptively determine size of trials
- Numerics?
