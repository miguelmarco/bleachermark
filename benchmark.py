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
        