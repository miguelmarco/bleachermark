from bleachermark.benchmark import benchmark

r"""
A Bleachermark is formed by a collection of benchmarks.

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
    >>> BB.run(4)
    >>> BB.fetch_data()
    (('becnhmark 1', 0, 1, 1.999999999990898e-05, 0.8444218515250481),
    ('becnhmark 1', 1, 1, 2.0000000000575113e-06, -0.1313735881920577),
    ('becnhmark 1', 2, 1, 7.000000000090267e-06, 0.9913828944313534),
    ('becnhmark 1', 3, 1, 1.9999999998354667e-06, -0.008542851060265422),
    ('becnhmark 1', 4, 1, 9.999999999177334e-07, 0.0086158313645033),
    ('becnhmark 1', 5, 1, 9.999999999177334e-07, 0.00861572476904337),
    ('stupid benchmark', 'identity', 1, 9.999999999177334e-07, 0),
    ('stupid benchmark', 'zero', 1, 0.0, 0),
    ('becnhmark 1', 0, 2, 1.6000000000016e-05, 0.13436424411240122),
    ('becnhmark 1', 1, 2, 1.000000000139778e-06, -0.11631049401650427),
    ('becnhmark 1', 2, 2, 9.999999999177334e-07, 0.9932435564834237),
    ('becnhmark 1', 3, 2, 2.0000000000575113e-06, -0.006710793987583674),
    ('becnhmark 1', 4, 2, 2.9999999999752447e-06, 0.006755828743527464),
    ('becnhmark 1', 5, 2, 0.0, 0.006755777352931481),
    ('stupid benchmark', 'identity', 2, 1.000000000139778e-06, 1),
    ('stupid benchmark', 'zero', 2, 9.999999999177334e-07, 0),
    ('becnhmark 1', 0, 3, 1.5000000000098268e-05, 0.9560342718892494),
    ('becnhmark 1', 1, 3, 9.999999999177334e-07, -0.042032742862442185),
    ('becnhmark 1', 2, 3, 1.000000000139778e-06, 0.9991167543148527),
    ('becnhmark 1', 3, 3, 0.0, -0.0008824655622069466),
    ('becnhmark 1', 4, 3, 0.0, 0.0008832443076754278),
    ('becnhmark 1', 5, 3, 4.000000000115023e-06, 0.0008832441928359328),
    ('stupid benchmark', 'identity', 3, 1.9999999998354667e-06, 2),
    ('stupid benchmark', 'zero', 3, 9.999999999177334e-07, 0),
    ('becnhmark 1', 0, 4, 1.8000000000073513e-05, 0.23796462709189137),
    ('becnhmark 1', 1, 4, 0.0, -0.18133746334490844),
    ('becnhmark 1', 2, 4, 9.999999999177334e-07, 0.9836033674136959),
    ('becnhmark 1', 3, 4, 9.999999999177334e-07, -0.016127783026133824),
    ('becnhmark 1', 4, 4, 1.000000000139778e-06, 0.016387888411471874),
    ('becnhmark 1', 5, 4, 0.0, 0.01638715489155228),
    ('stupid benchmark', 'identity', 4, 1.000000000139778e-06, 3),
    ('stupid benchmark', 'zero', 4, 1.000000000139778e-06, 0))


"""

class Bleachermark:
    def __init__(self, benchmarks):
        r"""
        INPUT:
        
        - ``benchmarks`` - A benchmark, or a list or tuple of them
        """
        if isinstance(benchmarks, Benchmark):
            self._benchmarks = (benchmarks, )
        else:
            self._benchmarks = tuple(benchmarks)
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