from bleachermark.benchmark import benchmark

r"""
A Bleachermark is formed by a collection of benchmarks.

"""

class Bleachermark:
    def __init__(self, benchmarks):
        r"""
        
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
        r"""
        Runs the benchmarks in the collection and stores the returned data
        """
        for n in range(nruns):
            for i in range(len(self._benchmarks)):
                bm = self._benchmarks[i]
                name = bm._name()
                if name is None:
                    name = str(i)
                obtdata = bm.run(self._runs[i])
                self._runs[i] += 1
                funnames = bm._function_names()
                for j in range(len(funnames)):
                    data = (name, funnames[j], self._runs[i], obtdata[0][j], obtdata[1][j])
                    self._stored_data.append(data)
    
    def fetch_data(self):
        r"""
        Get the stored data
        """
        return tuple(self._stored_data)