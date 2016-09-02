# Logic for running benchmarks

- Current proposal: Runners

# Parallelism

- Suggestions?
- Take a look at https://wiki.python.org/moin/ParallelProcessing.
- Decouple the SageMath parallel framework into a separate package

# Interruption and Resume

- Works?
- How to integrate with parallelism?
- What did David do in #20684

# Out-of-the-Box Statistics

#  Other
- Write setup.py and so on to make this a pip installable package
- Decide on memory safety for Bleachermark.__add__ (how do we avoid copying data on add'ing)?
