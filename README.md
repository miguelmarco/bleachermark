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

## Usage:

Right now, the intended use case is the following:

- The user creates his benchmarks, by passing a pipeline of functions. Optionally labels can be assigned.
- The bleachermark object is created from the benchmarks
- Then the bleachermark can be told to run the benchmarks and store the information generated. By default
it uses the SerialRunner, that runs the becnhmarks sequentially as many times as told; but other runner can be implemented.
    - After being run, the bleachermark can give the stored data, and also compute basic statistics on it.
- Another option to run the beleachermark is to set a runner, and then use it as an iterator that returns the results of the benchmarks

Here we see an example of using the bleachermark in the first mode (store the data):

    >>> pipeline1 = [lambda i: (i*i, i+1), lambda (a, b): a+b]
    >>> pipeline2 = [lambda i: i, lambda i: 0]
    >>> B1 = Benchmark(pipeline1, label='benchmark 1')
    >>> B2 = Benchmark(pipeline2, label='benchmark 2', fun_labels=['identity', 'zero'])
    >>> B = Bleachermark((B1, B2))
    >>> B.run(3)
    >>> B.fetch_data()   # fetch all the data
    {'benchmark 1': [[(1.9999999993913775e-06, (0, 1)),
    (1.000000000139778e-06, 1)],
    [(1.000000000139778e-06, (1, 2)), (0.0, 3)],
    [(1.000000000139778e-06, (4, 3)), (0.0, 7)]],
    'benchmark 2': [[(2.9999999995311555e-06, 0), (0.0, 0)],
    [(3.000000000419334e-06, 1), (1.000000000139778e-06, 0)],
    [(2.000000000279556e-06, 2), (0.0, 0)]]}
    >>> B.fetch_data(format='flat')  #  We can ask for a format that is better suited for pandas
    [('benchmark 1', '0', 0, 1.9999999993913775e-06, (0, 1)),
    ('benchmark 1', '1', 0, 1.000000000139778e-06, 1),
    ('benchmark 1', '0', 1, 1.000000000139778e-06, (1, 2)),
    ('benchmark 1', '1', 1, 0.0, 3),
    ('benchmark 1', '0', 2, 1.000000000139778e-06, (4, 3)),
    ('benchmark 1', '1', 2, 0.0, 7),
    ('benchmark 2', 'identity', 0, 2.9999999995311555e-06, 0),
    ('benchmark 2', 'zero', 0, 0.0, 0),
    ('benchmark 2', 'identity', 1, 3.000000000419334e-06, 1),
    ('benchmark 2', 'zero', 1, 1.000000000139778e-06, 0),
    ('benchmark 2', 'identity', 2, 2.000000000279556e-06, 2),
    ('benchmark 2', 'zero', 2, 0.0, 0)]
    >>> B.timimgs()   #  Just ask for the timings
    {'benchmark 1': [[1.9999999993913775e-06, 1.000000000139778e-06],
    [1.000000000139778e-06, 0.0],
    [1.000000000139778e-06, 0.0]],
    'benchmark 2': [[2.9999999995311555e-06, 0.0],
    [3.000000000419334e-06, 1.000000000139778e-06],
    [2.000000000279556e-06, 0.0]]}
    >>> B.averages()  # and then ask for basic statistics
    {'benchmark 1': [1.3333333332236446e-06, 3.33333333379926e-07],
    'benchmark 2': [2.6666666667433483e-06, 3.33333333379926e-07]}
    >>> B.stdvs()  # the standard deviations
    {'benchmark 1': [4.7140452043823233e-07, 4.7140452085692363e-07],
    'benchmark 2': [4.71404520647578e-07, 4.7140452085692363e-07]}
    
An example of using the bleachermark as an iterator:

    >>> pipeline1 = [lambda i: (i*i, i+1), lambda (a, b): a+b]
    >>> pipeline2 = [lambda i: i, lambda i: 0]
    >>> B1 = Benchmark(pipeline1, label='benchmark 1')
    >>> B2 = Benchmark(pipeline2, label='benchmark 2', fun_labels=['identity', 'zero'])
    >>> B = Bleachermark((B1, B2))
    >>> B.set_runner(SerialRunner, 10)  # we tell the benchmark to use the serial runnerm with 10 iterations
    >>> B.next() # Then we can iterate over it
    ([(4.999999999810711e-06, (1, 2)), (5.999999999950489e-06, 3)], 0)
    >>> B.next()
    ([(1.000000000139778e-06, 0), (2.9999999995311555e-06, 0)], 1)

    

    


