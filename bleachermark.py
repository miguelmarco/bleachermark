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
    >>> BB.fetch_data()
    (('benchmark 1', 0, 0, 5.3999999999998494e-05, 0.8444218515250481),
    ('benchmark 1', 1, 0, 2.9999999999752447e-06, -0.1313735881920577),
    ('benchmark 1', 2, 0, 5.999999999950489e-06, 0.9913828944313534),
    ('benchmark 1', 3, 0, 5.000000000254801e-06, -0.008542851060265422),
    ('benchmark 1', 4, 0, 2.000000000279556e-06, 0.0086158313645033),
    ('benchmark 1', 5, 0, 2.000000000279556e-06, 0.00861572476904337),
    ('stupid benchmark', 'identity', 0, 2.000000000279556e-06, 0),
    ('stupid benchmark', 'zero', 0, 6.000000000394579e-06, 0),
    ('benchmark 1', 0, 1, 6.100000000008876e-05, 0.13436424411240122),
    ('benchmark 1', 1, 1, 1.000000000139778e-06, -0.11631049401650427),
    ('benchmark 1', 2, 1, 1.000000000139778e-06, 0.9932435564834237),
    ('benchmark 1', 3, 1, 1.000000000139778e-06, -0.006710793987583674),
    ('benchmark 1', 4, 1, 4.000000000115023e-06, 0.006755828743527464),
    ('benchmark 1', 5, 1, 2.000000000279556e-06, 0.006755777352931481),
    ('stupid benchmark', 'identity', 1, 1.000000000139778e-06, 1),
    ('stupid benchmark', 'zero', 1, 1.9999999998354667e-06, 0))



"""

from time import clock


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
        self._blabel = label
        if fun_labels is None:
            self._fun_labels = tuple(str(i) for i in range(len(self._pipeline)))
        else:
            if not isinstance(fun_labels, (list, tuple)):
                raise TypeError("The labels of the functions must be given in a list or tuple")
            if len(fun_labels) != len(self._pipeline):
                raise ValueError("There must be as many labels as functions")
            self._fun_labels = tuple(fun_labels)
        
    def __repr__(self):
        return 'Benchmark for a pipeline of {} functions'.format(len(self._pipeline))
    
    def _label(self):
        return self._blabel
        
    def _function_labels(self):
        return self._fun_labels
    
    def run(self, i):
        r"""
        Run the pipeline and return the timings and values produced.
        
        INPUT:
        
        - ``i`` - The input fed to the first element of the pipeline. Typically
                  an identifier of the running instance.
                  
        OUTPUT:
        
        - A list of the time that each element of the pipeline took, and a tuple
          of the values produced by the functions in the pipeline.
        """
        timings = []
        values = []
        intervalue = i
        for fun in self._pipeline:
            tim = clock()
            intervalue = fun(intervalue)
            timings.append(clock()-tim)
            values.append(intervalue)
        return timings, values
        



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
            self._benchmark = (Benchmark(benchmarks),)
        self._stored_data = []
        self._runs = {i: 0 for i in range(len(self._benchmarks))}
    
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
        for n in range(nruns):
            for i in range(len(self._benchmarks)):
                bm = self._benchmarks[i]
                label = bm._label()
                if label is None:
                    label = str(i)
                obtdata = bm.run(self._runs[i])
                fun_labels = bm._function_labels()
                for j in range(len(fun_labels)):
                    data = (label, fun_labels[j], self._runs[i], obtdata[0][j], obtdata[1][j])
                    self._stored_data.append(data)
                self._runs[i] += 1
                
    
    def clean_data(self):
        r"""
        Cleans the stored data
        """
        self._stored_data = []
    
    def fetch_data(self):
        r"""
        Get the stored data
        """
        return tuple(self._stored_data)
