r"""
The Bleachermark package

A Bleachermark is a collection of benchmarks. A benchmark is a single pipeline
of functions, often randomized in one or more functions, whose performance and
data one wishes to perform statistics on.

EXAMPLE::

    >>> import math
    >>> def gener(i):
    ...     random.seed(i)
    ...     return random.random()
    >>> def f1(x):
    ...     return x*x - x
    >>> B = Benchmark([gener, f1, math.cos, f1, f1, math.sin], name = 'becnhmark 1')
    >>> B2 = Benchmark([lambda i: i, lambda i: 0], name='stupid benchmark', funnames=['identity', 'zero']
    >>> BB = Bleachermark((B, B2))
    >>> BB.run(2)
    >>> BB.fetch_data()
    (('becnhmark 1', 0, 0, 5.3999999999998494e-05, 0.8444218515250481),
    ('becnhmark 1', 1, 0, 2.9999999999752447e-06, -0.1313735881920577),
    ('becnhmark 1', 2, 0, 5.999999999950489e-06, 0.9913828944313534),
    ('becnhmark 1', 3, 0, 5.000000000254801e-06, -0.008542851060265422),
    ('becnhmark 1', 4, 0, 2.000000000279556e-06, 0.0086158313645033),
    ('becnhmark 1', 5, 0, 2.000000000279556e-06, 0.00861572476904337),
    ('stupid benchmark', 'identity', 0, 2.000000000279556e-06, 0),
    ('stupid benchmark', 'zero', 0, 6.000000000394579e-06, 0),
    ('becnhmark 1', 0, 1, 6.100000000008876e-05, 0.13436424411240122),
    ('becnhmark 1', 1, 1, 1.000000000139778e-06, -0.11631049401650427),
    ('becnhmark 1', 2, 1, 1.000000000139778e-06, 0.9932435564834237),
    ('becnhmark 1', 3, 1, 1.000000000139778e-06, -0.006710793987583674),
    ('becnhmark 1', 4, 1, 4.000000000115023e-06, 0.006755828743527464),
    ('becnhmark 1', 5, 1, 2.000000000279556e-06, 0.006755777352931481),
    ('stupid benchmark', 'identity', 1, 1.000000000139778e-06, 1),
    ('stupid benchmark', 'zero', 1, 1.9999999998354667e-06, 0))



"""

from time import clock


class Benchmark():
    r"""
    A Benchmark is a pipeline of functions.
    
    """
    def __init__(self, l, name = None, funnames = None):
        r"""
        Initializes the Benchmark
        
        INPUT:
        
        -  ``l`` - the list of functions in the Benchmark.
        
        - ``name`` - the name of the benchmark.
        
        - ``funnames`` - the names of the functions. If it is nit given,
          they will be named by their index.
        """
        if not isinstance(l, (list, tuple)):
            raise TypeError("A list or tuple of functions must be provided")
        self._pipeline = tuple(l)
        self._bname = name
        if funnames is None:
            self._funnames = tuple(str(i) for i in range(len(self._pipeline)))
        else:
            if not isinstance(funnames, (list, tuple)):
                raise TypeError("The names of the functions must be given in a list or tuple")
            if len(funnames) != len(self._pipeline):
                raise ValueError("There must be as many names as functions")
            self._funnames = tuple(funnames)
        
    def __repr__(self):
        return 'Benchmark for a pipeline of {} functions'.format(len(self._pipeline))
    
    def _name(self):
        return self._bname
        
    def _function_names(self):
        return self._funnames
    
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
        elif instance(benchmarks, (list, tuple)) and all [isinstance(i, Benchmark) for i in benchmarks]:
            self._benchmarks = tuple(benchmarks)
        elif instance(benchmarks, (list, tuple)) and all [isinstance(i, (list, tuple)) for i in benchmarks]:
            self._benchmarks = tuple(Benchmark(i) for i in benchmarks)
        else:
            self._benchmark = (Benchmark(benchmarks),)
        self._stored_data = []
        self._runs = {i: 0 for i in range(len(self._benchmarks))}
    
    def __repr__(self):
        return 'Collection of {} benchmarks'.format(len(self._benchmarks))
    
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
                name = bm._name()
                if name is None:
                    name = str(i)
                obtdata = bm.run(self._runs[i])
                funnames = bm._function_names()
                for j in range(len(funnames)):
                    data = (name, funnames[j], self._runs[i], obtdata[0][j], obtdata[1][j])
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
