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

## User Stories:

Essentially, BleacherMark just runs some functions for you, for a certain number
of iterations or until you don't want to wait more. Computation results,
possibly also intermediate values, and timing data is collected.

The clever thing about BleacherMark is that it allows you to do this in many,
many ways, in parallel on your own or many machines, and it allows you to
discover the results during computation and to respond to this by changing your
approach.


### Benchmarking an algorithm

You have an algorithm that you wish to speed test? Just type the following in a
Python or IPython shell:

    def data_gen():
         return some_random_data
    def my_algo(input):
         ....
    B = Bleachermark([ (data_gen, my_algo) ], N =  1000)
    B.run() # runs the function 1000 times


OK, so you wish to know how speed of the algorithm changes with the size of the
data? No problem:

    def data_gen(size):
         return lambda (): some_random_data_of_size(size)
    def my_algo(input):
         ....
    B = Bleachermark([ (data_gen(size), my_algo) for size in range(1, 100) ], N = 100)
    B.run() # runs the function 100 times for each size


Ah, but the above will spend a lot more time on the big sizes than on the
smaller sizes. To even this out, we have an **adaptive** strategy:

    B = Bleachermark([ (data_gen(size), my_algo) for size in range(1, 100) ], adaptive = True, time=60 * 60)
    B.run() # run the function on sizes adaptively for 1 hour.

Oh, but after 15 minutes you get bored and wish to inspect the result? Just go
ahead and interrupt:

    <Press Ctrl-c to interrupt computation>
    B.result_overview()
    <blah blah>
    B.run() # continues running

In fact, back when defining `B`, we didn't even need to set a `time` argument:

    B = Bleachermark([ (data_gen(size), my_algo) for size in range(1, 100) ], adaptive = True)
    B.run() # run the function on sizes adaptively until it's interrupted
    <When you get bored, interrupt using Ctrl-C and inspect the results>


### Pitting two algorithms against each other

    def data_gen():
         return some_random_data
    def algo1(input):
         ....
    def algo2(input):
         ....
    B = Bleachermark([ (data_gen(size), algo1, name="algo1 %s" % size) for size in range(1, 100) ]
                   + [ (data_gen(size), algo2, name="algo2 %s" % size) for size in range(1, 100) ], N = 100)
    B.run() # runs both algorithms 100 times on each size
    g1 = B.plot_speeds([ "algo1 %s" % size for size in range(1,100) ])
    g2 = B.plot_speeds([ "algo2 %s" % size for size in range(1,100) ])
    show(g1 + g2) # requires Sage


### Observing multiple steps

You have a multi-step algorithm and wish to observe the performance of all the
steps? Or perhaps randomness is introduced not only in the beginning but also
the middle of a computation, and you wish to save all the random values for
later reference. No biggie:

    def data_gen():

    def prepare(data):
        return ...
    def rough_solution(data):
        return ...
    def optimize(data):
        return ...
    def pipeline = [ data_gen, prepare, rough_solution, optimize ]
    B = Bleachermark(pipeline, N = 100)
    B.run() # runs the entire pipeline 100 times and saves everything

 
### Finding the cut-off point between basic case and recursive case insehD&C algorithms

A classical example for this is Karatsuba multiplication: an ingenious recursive
step allows multiplying two large integers by 3 multiplications (and some
additions) of smaller integers. A good implementation needs to choose when the
smaller multiplications should be done by Karatsuba recursively, or in a
classical fashion:

    def multiply(cutoff):
        r"""Return a Karatsuba multiplication function with set cutoff point"""
        def karatsuba(a, b):
            r"""Multiply a and b using Karatsuba"""
            def _karatsuba_rec(a, b, k):
                if k < cutoff:
                    return a * b   #classical multiplication
                else:
                    k = ceil(k/2)
                    a1, a2 = split_bits(a, k)
                    b1, b2 = split_bits(b, k)
                    A = mult_karatsuba(a1, b1)
                    B = mult_karatsuba(a2, b2)
                    C = mult_karatsuba(a1 + a2, b1 + b2)
                    return A + shift(C - A - B, k) + shift(B, 2*k)
            return _karatsuba_rec(a, b, max(bitlen(a), bitlen(b)))
        return karatsuba

    B = Bleachermark(lambda cutoff: [ gen_100_bit_numbers, multiply(cutoff) ], adaptive=True)
    B.run()



### Use Bleachermark's parallelisation etc. to find a Needle in a Haystack

You seek an input to your function `f` which makes `f` return `True`, but you
have no idea what this input could be:

    search_space = ( (i,j,k) for i in range(1000) for j in range(1000) for k in range(1000) )
    def f(input):
        return ...
    N = NeedleInHaystack(f, search_space)
    N.run()
    (35, 46, 341)

`NeedleInHaystack` takes many of the same arguments as `Bleachermark`, such as
`parallel, ssh, shuffle`, etc.

