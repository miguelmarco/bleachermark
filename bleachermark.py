r"""
The Bleachermark package

A Bleachermark is a collection of benchmarks. A benchmark is a single pipeline
of functions, often randomized in one or more functions, whose performance and
data one wishes to perform statistics on.

EXAMPLE::

    >>> from bleachermark import *
    >>> import math, random
    >>> def data_gen(i):
    ...     random.seed(i)
    ...     return random.random()
    >>> def f1(x):
    ...     return x*x - x
    >>> pipeline = [data_gen, f1, math.cos, f1, f1, math.sin]
    >>> B = Benchmark(pipeline, label='benchmark 1')
    >>> zero_pipeline = [lambda i: i, lambda i: 0]
    >>> B2 = Benchmark(zero_pipeline, label='stupid benchmark', fun_labels=['identity', 'zero'])
    >>> BB = Bleachermark((B, B2))
    >>> BB.run(2)
    >>> BB.fetch_data("dict")
    {'benchmark 1': [[(1.6000000000016e-05, 0.8444218515250481),
    (1.0000000000287557e-06, -0.1313735881920577),
    (8.000000000008e-06, 0.9913828944313534),
    (0.0, -0.008542851060265422),
    (0.0, 0.0086158313645033),
    (0.0, 0.00861572476904337)],
    [(1.0999999999983245e-05, 0.13436424411240122),
    (1.0000000000287557e-06, -0.11631049401650427),
    (0.0, 0.9932435564834237),
    (0.0, -0.006710793987583674),
    (1.0000000000287557e-06, 0.006755828743527464),
    (1.0000000000287557e-06, 0.006755777352931481)]],
    'stupid benchmark': [[(1.0000000000287557e-06, 0),
    (9.999999999177334e-07, 0)],
    [(0.0, 1), (0.0, 0)]]}
    >>> BB.fetch_data("flat")
    [('benchmark 1', '0', 0, 1.7000000000044757e-05, 0.8444218515250481),
    ('benchmark 1', '1', 0, 9.999999999177334e-07, -0.1313735881920577),
    ('benchmark 1', '2', 0, 6.000000000061512e-06, 0.9913828944313534),
    ('benchmark 1', '3', 0, 0.0, -0.008542851060265422),
    ('benchmark 1', '4', 0, 9.999999999177334e-07, 0.0086158313645033),
    ('benchmark 1', '5', 0, 1.0000000000287557e-06, 0.00861572476904337),
    ('benchmark 1', '0', 1, 2.8000000000028002e-05, 0.13436424411240122),
    ('benchmark 1', '1', 1, 1.0000000000287557e-06, -0.11631049401650427),
    ('benchmark 1', '2', 1, 9.999999999177334e-07, 0.9932435564834237),
    ('benchmark 1', '3', 1, 1.0000000000287557e-06, -0.006710793987583674),
    ('benchmark 1', '4', 1, 2.0000000000575113e-06, 0.006755828743527464),
    ('benchmark 1', '5', 1, 0.0, 0.006755777352931481),
    ('stupid benchmark', 'identity', 0, 9.999999999177334e-07, 0),
    ('stupid benchmark', 'zero', 0, 0.0, 0),
    ('stupid benchmark', 'identity', 1, 0.0, 1),
    ('stupid benchmark', 'zero', 1, 0.0, 0)]
"""

from time import clock
from copy import copy


class Benchmark():
    r"""
    A Benchmark is a pipeline of functions.
    
    """
    def __init__(self, pipeline, label=None, fun_labels = None):
        r"""
        Initializes the Benchmark
        
        INPUT:
        
        -  ``l`` - the list of functions in the Benchmark.
        
        - ``label`` - the name of the benchmark.
        
        - ``fun_labels`` - the names of the functions. If it is nit given,
          they will be named by their index.
        """
        if not isinstance(pipeline, (list, tuple)):
            raise TypeError("Pipeline must be a list or tuple of functions")
        self._pipeline = tuple(pipeline)
        self._label = label
        if fun_labels is None:
            self._fun_labels = tuple(str(i) for i in range(len(self._pipeline)))
        else:
            if not isinstance(fun_labels, (list, tuple)):
                raise TypeError("The labels of the functions must be given in a list or tuple")
            if len(fun_labels) != len(self._pipeline):
                raise ValueError("There must be as many labels as functions")
            self._fun_labels = tuple(fun_labels)
        
    def __repr__(self):
        if self.label():
            return 'Benchmark {}'.format(self.label())
        else:
            return 'Benchmark for a pipeline of {} functions'.format(len(self._pipeline))

    def label(self):
        r"""
        Return the label (name) for this benchmark.
        """
        return self._label

    def _set_label(self, label):
        self._label = label
        
    def function_labels(self):
        r"""
        Return the functions' labels for this benchmark.
        """
        return self._fun_labels

    def pipeline(self):
        r"""
        Return the pipeline of functions of this benchmark.
        """
        return self._pipeline
    
    def run(self, i):
        r"""
        Run the pipeline and return the timings and values produced.
        
        INPUT:
        
        - ``i`` - The input fed to the first element of the pipeline. Typically
                  an identifier of the running instance.
                  
        OUTPUT:
        
        - A list of pairs, with one pair for each part of the pipeline. The
          first element of each pair is the time that this part of the
          pipeline took, and the second is the value it output.
        """
        time_vals = []
        intervalue = i
        for fun in self._pipeline:
            tim = clock()
            intervalue = fun(intervalue)
            time_vals.append( (clock()-tim,  intervalue) )
        return time_vals
        



class Bleachermark:
    def __init__(self, benchmarks):
        r"""
        INPUT:
        
        - ``benchmarks`` - A benchmark, or a list or tuple of them
        """
        if isinstance(benchmarks, Benchmark):
            self._benchmarks = (benchmarks, )
        elif isinstance(benchmarks, (list, tuple)) and all([isinstance(i, Benchmark) for i in benchmarks]):
            self._benchmarks = tuple(benchmarks)
        elif isinstance(benchmarks, (list, tuple)) and all([isinstance(i, (list, tuple)) for i in benchmarks]):
            self._benchmarks = tuple(Benchmark(i) for i in benchmarks)
        else:
            self._benchmarks = (Benchmark(benchmarks),)

        for n, b in zip(range(self.size()), self._benchmarks):
            if b.label() is None:
                b._set_label(str(n))
        self._measurements = { b.label(): [] for b in self._benchmarks }  # benchmark label -> ((timing, value) list) list
    
    def __repr__(self):
        return 'Collection of {} benchmarks'.format(self.size())

    def size(self):
        return len(self._benchmarks)
    
    def run(self, nruns = 100): # This is the part that should need the most work: lots of options about how to run it
        # For the moment it just runs all the benchmarks the same number of times
        # No automatic tweaking, no parallelism, no nothing
        r"""
        Runs the benchmarks in the collection and stores the returned data
        
        INPUT:
        
        - ``nruns`` - The number of times each 
        """
        labels = [ l if l is not None else str(i) for (i,l) in
                    zip(range(self.size()),[ b.label() for b in self._benchmarks]) ]
        measurements  = self._measurements
        
        for n in range(nruns):
            for i in range(len(self._benchmarks)):
                benchmark = self._benchmarks[i]
                label = benchmark.label()
                m = benchmark.run(n)
                measurements[label].append(m)
                
    def clear(self):
        r"""
        Forget all measurements.
        """
        self._measurements = []

    def fetch_data(self, format="dict"):
        r"""
        Return all the measured data.

        INPUT:

        - ``format`` - (optional, default: "dict") specify the format to return
          the data in. If set to "dict", return a dictionary from Benchmark
          label to list of measurements, each measurement being as the output of
          ``Benchmark.run``. If set to "flat", return a list of tuples of the
          format ``(benchmark label, run-no, pipeline-part, timing, output)``.
        """
        measurements = self._measurements
        if format == "dict":
            #TODO: Really copy?
            return copy(measurements)
        elif format == "flat":
            data = []
            for benchmark in self._benchmarks:
                label = benchmark.label()
                fun_labels = benchmark.function_labels()
                for run in range(len(measurements[label])):
                    for i in range(len(benchmark.pipeline())):
                        m = measurements[label][run][i]
                        data.append( (label, fun_labels[i], run, m[0], m[1]) )
            return data
        else:
            raise ValueError("Invalid argument to format: %s".format(format))

    def timings(self):
        r"""
        Return all measured timings.
        
        OUTPUT: 
          
          - a dictionary whose keys are the labels of the benchmarks.
          The value for each benchmark is a list corresponding to the runs.
          For each run, there is a list of the timings of the different 
          components of the pipeline.
         
        """
        di = self.fetch_data()
        return {bm:[[t[0] for t in run] for run in di[bm]] for bm in di.keys()}
    
    def averages(self):
        r"""
        Return the averages of the timings.
        
        OUTPUT:
        
          - A dictionary whose keys are the benchmarks. The value for each benchmark
          is a list with the averages of the corresponding parts of the pipeline.
    
        """
        timings = self.timings()
        res = {}
        for bm in timings.keys():
            l = len(timings[bm][0])
            totals = [0 for i in range(l)]
            for run in timings[bm]:
                for i in range(l):
                    totals[i] += run[i]
            res[bm] = [t / len(timings[bm]) for t in totals]
        return res
    
    def pipeline_data(self):
        r"""
        Get the data through the pipeline of all benchmarks and runs.
        """
        raise NotImplementedError
