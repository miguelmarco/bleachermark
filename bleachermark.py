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
    {'benchmark 1': [[0, (1.6000000000016e-05, 0.8444218515250481),
    (1.0000000000287557e-06, -0.1313735881920577),
    (8.000000000008e-06, 0.9913828944313534),
    (0.0, -0.008542851060265422),
    (0.0, 0.0086158313645033),
    (0.0, 0.00861572476904337)],
    [1, (1.0999999999983245e-05, 0.13436424411240122),
    (1.0000000000287557e-06, -0.11631049401650427),
    (0.0, 0.9932435564834237),
    (0.0, -0.006710793987583674),
    (1.0000000000287557e-06, 0.006755828743527464),
    (1.0000000000287557e-06, 0.006755777352931481)]],
    'stupid benchmark': [[0, (1.0000000000287557e-06, 0),
    (9.999999999177334e-07, 0)],
    [1, (0.0, 1), (0.0, 0)]]}
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

#This part handles the ctrl-c interruption.
#class CTRLC(Exception):
    #def __init__(self):
        #pass

import signal

def signal_ctrl_c(signal, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, signal_ctrl_c)

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

        - A list  whose first element is the input passed to the first function
          of the pipeline. The rest of the elements are pairs, with one pair for
          each part of the pipeline. The first element of each pair is the time
          that this part of the pipeline took, and the second is the value it output.
        """
        time_vals = [i]
        intervalue = i
        for fun in self._pipeline:
            tim = clock()
            intervalue = fun(intervalue)
            time_vals.append( (clock()-tim,  intervalue) )
        return time_vals



def _make_autolabel(n):
    r"""
    Return a generic label for a benchmark with count n
    """
    return "[%s]" % n

_autolabel_regex = r"\[[0-9]*\]"


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
                b._set_label(_make_autolabel(n))
        # benchmark label -> ((timing, value) list) list
        # (time, value) = self._measurements[bm.label()][run_no][pipeline_part]
        self.clear()
        self._current_runner = None

    def __repr__(self):
        return 'Collection of {} benchmarks'.format(self.size())

    def size(self):
        return len(self._benchmarks)

    def __iter__(self):   # Implement iterator behaviour
        return self

    def next(self):      # Should we store the result?
        return self._current_runner.next()

    __next__ = next

    def set_runner(self, runner, *args, **kwargs):
        r"""
        Set the runner to be used when the bleachermark is used as an iterator:

        INPUT:

        - ``runner`` - the constructor of the runner to be set.

        - ``*args`` - the arguments to be passed to the constructor of the runner.

        - ``**kwargs`` - the ketword arguments to be pased to the constructor of the runner.

        EXAMPLES::

            >>> from bleachermark import *
            >>> import math, random
            >>> def data_gen(i):
            ...     random.seed(i)
            ...     return random.random()
            >>> def f1(x):
            ...     return x*x - x
            >>> pipeline = [data_gen, f1, math.cos]
            >>> B = Benchmark(pipeline, label='benchmark 1')
            >>> zero_pipeline = [lambda i: i, lambda i: 0]
            >>> B2 = Benchmark(zero_pipeline, label='stupid benchmark', fun_labels=['identity', 'zero'])
            >>> BB = Bleachermark((B, B2))
            >>> BB.set_runner(SerialRunner, 100)
            >>> BB.next()
            (0, (0, (2.2999999999884224e-05, 0.8444218515250481),
              (2.0000000000575113e-06, -0.1313735881920577),
              (2.9999999999752447e-06, 0.9913828944313534)))
            >>> BB.next()
            (0, (1, (3.999999999892978e-06, 0), (2.9999999999752447e-06, 0)))

        This way, the bleachermark can be used as part of a bigger pipeline (for instance,
        feeding output to another engine that makes statistical analyisis, or plots.
        """
        runner = runner(self, *args, **kwargs)
        self._current_runner = runner

    def run(self, nruns = 100): # Refactored to the runnners model
        # This function should create a runner according to the passed parameters and run it
        # For the moment it just uses the serial runner with the parameter nruns
        # No automatic tweaking, no parallelism, no nothing
        r"""
        Runs the benchmarks in the collection and stores the returned data

        INPUT:

        - ``nruns`` - The number of times each
        """
        runner = SerialRunner(self, nruns)
        self._current_runner = runner
        self.resume()

    def resume(self):
        r"""
        Resumes the run of the current runner and stores the measurements produced.
        """
        labels = [ l if l is not None else str(i) for (i,l) in
                    zip(range(self.size()),[ b.label() for b in self._benchmarks]) ]
        measurements = self._measurements
        while True:
            try:
                r = self._current_runner.next()
                label = labels[r[0]]
                measurements[label].append(r[1])
            except (StopIteration, KeyboardInterrupt):
                return


    def clear(self):
        r"""
        Forget all measurements.
        """
        self._measurements = { b.label(): [] for b in self._benchmarks }  # benchmark label -> ((timing, value) list) list

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
                for run in measurements[label]:
                    for i in range(len(benchmark.pipeline())):
                        m = run[i+1]
                        data.append( (label, fun_labels[i], run[0], m[0], m[1]) )
            return data
        else:
            raise ValueError("Invalid argument to format: %s".format(format))

    def timings(self, transposed=False):
        r"""
        Return all measured timings.

        INPUT:

         - ``transposed`` - (default = False), determines weather the data must
            be transposed

        OUTPUT:

          - a dictionary whose keys are the labels of the benchmarks.
          The value for each benchmark is a list corresponding to the runs.
          For each run, there is a list of the timings of the different
          components of the pipeline.

        """
        di = self.fetch_data()
        res =  {bm:[[t[0] for t in run[1:]] for run in di[bm]] for bm in di.keys()}
        if not transposed:
            return res
        return {bm:[[row[i] for row in res[bm]] for i in range(len(res[bm][0]))] for bm in res.keys()}

    def averages(self):
        r"""
        Return the averages of the timings.

        OUTPUT:

          - A dictionary whose keys are the benchmarks. The value for each benchmark
          is a list with the averages of the corresponding parts of the pipeline.

        """
        timings = self.timings(transposed=True)
        res = {}
        for bm in timings.keys():
            totals = map(lambda a: sum(a)/len(a), timings[bm])
            res[bm] = totals
        return res

    def variances(self):
        r"""
        Return the variances of the timings of the benchmarks
        """
        timings = self.timings(transposed=True)
        averages = self.averages()
        res = {}
        for bm in timings.keys():
            timbm = timings[bm]
            avgbm = averages[bm]
            lbm = []
            for (timpart, avgpart) in zip(timbm, avgbm):
                deviations = [(l - avgpart)**2 for l in timpart]
                lbm.append(sum(deviations)/len(deviations))
            res[bm] = lbm
        return res

    def stdvs(self):
        r"""
        Return the standard deviations of the timings.
        """
        variances = self.variances()
        import math
        return {bm:map(math.sqrt, variances[bm]) for bm in variances}

    def maxes(self):
        r"""
        Return the maximum running times of the benchmarks run.
        """
        timings = self.timings(transposed=True)
        return {bm:map(max, timings[bm]) for bm in timings}

    def mins(self):
        r"""
        Return the minimum running times of the benchmarks run.
        """
        timings = self.timings(transposed=True)
        return {bm:map(min, timings[bm]) for bm in timings}

    def pipeline_data(self):
        r"""
        Get the data through the pipeline of all benchmarks and runs.
        """
        raise NotImplementedError

    def __add__(self, other):
        r"""
        Add two Bleachermarks, concatenating their benchmarks.

        Note that this does not lose existing measurement data.
        """
        raise NotImplementedError("Not verified since Runner system")
        import re
        my_labels = set( b.label() for b in self._benchmarks )
        ot_labels = set( b.label() for b in other._benchmarks )
        collisions = my_labels.intersect(ot_labels)
        if collisions:
            autolabel = re.compile(_autolabel_regex)
            counter = self.size()
            for label in collisions:
                if autolabel.match(label):
                    #Change name of other's benchmark
                    other.benchmark(label)._set_label(_make_autolabel(counter))
                    counter += 1
                else:
                    raise ValueError("Collision on label %s" % label)

        #Now benchmarks can just be concatenated
        self._benchmarks.extend(other._benchmarks)
        for b in other._benchmarks:
            assert not b.label() in self._measurements
            self._measurements[b.label()] = copy(other._measurements[b.label()]) #TODO: deepcopy?

        return self




#RUNNERS
# Runners are essentially iterators that produce the data that the bleachermark will store.
#They should support the following interface:
# They are created by passing the bleachermark that created them, and the specific parameters of the runner
# They act as iterators that yield the results of the benchmarks.
# - The format of the return is a tuple (index, results), where
#   - index is the index of the benchmark
#   - results is the result of running the benchmark
# It is the runners work to decide how to order the benchmarks, call them in parallel and so on


class SerialRunner:
    r"""
    Example of Runner. It just runs the each benchmark of the bleachermark as many times as indicated.
    It most likey will be the default one
    """
    def __init__(self, bleachermark, iterations):
        r"""
        INPUT:

        - ``bleachermark`` - The bleachermark that calls it.

        - ``iterations`` - The number of
        """
        self._bleachermark = bleachermark
        self._niter = iterations
        self._benchmarks = bleachermark._benchmarks
        self._current_iter = 0
        self._current_benchmark = 0

    def __iter__(self):
        return self

    def __repr__(self):
        return 'Serial Runner of {} instances for {}'.format(self._niter, self._bleachermark)

    def next(self):
        if self._current_iter >= self._niter:
            raise StopIteration()
        else:
            runid = self._current_iter
            benchmarkid = self._current_benchmark
            res = (benchmarkid, self._benchmarks[benchmarkid].run(runid))
            if self._current_benchmark == len(self._benchmarks) - 1:
                self._current_benchmark = 0
                self._current_iter += 1
            else:
                self._current_benchmark += 1
            return res
    __next__ = next

class ListRunner:
    r"""
    Runner based on a list. You just pass a list of the runid's you want to pass
    to your benchmarks, and it takes care of it.

    EXAMPLE::


    """
    def __init__(self, bleachermark, runids):
        r"""
        INPUT:

            - ``bleachermark`` - the bleachermark that calls this runner

            - ``runids`` - a list, tuple or other iterable with the runids to
            be passed to the benchmarks.
        """
        self._bleachermark = bleachermark
        self._idqueue = runids.__iter__()
        self._benchmarks = bleachermark._benchmarks
        self._nbench = len(self._benchmarks) - 1
        self._currentid = None

    def __iter__(self):
        return self

    def __repr__(self):
        return 'List Runner for {}'.format(self._bleachermark)

    def next(self):
        if self._nbench == len(self._benchmarks) - 1:
            self._nbench = 0
            self._currentid = self._idqueue.next()
        else:
            self._nbench += 1
        return (self._nbench, self._benchmarks[self._nbench].run(self._currentid))

    __next__ = next

class ParallelRunner:
    r"""
    This runner uses sage parallel utility.
    It profiles each benchmark and decides how many runs can fit in a chunk of about two
    seconds. Then it computes these chunks in parallel.

    As input, it takes a list or tuple of the inputs that will be given to the benchmarks.
    """
    def __init__(self, bleachermark, runs):
        from sage.parallel.decorate import parallel
        from sage.functions.other import ceil, floor
        self._benchmarks = bleachermark._benchmarks
        # profiling we run each benchmark once
        self._totaltime = reduce(lambda a,b: a+b, [r[0] for bm in self._benchmarks for r in bm.run(runs[0])[1:]])
        #divide the runs in chunks
        self._chunksize = ceil(2.0 / self._totaltime)
        self._nchunks = floor(len(runs)/self._chunksize)
        self._chunks = [runs[i*self._chunksize:(i+1)*self._chunksize] for i in range(self._nchunks)]
        if (self._nchunks)*self._chunksize < len(runs):
            self._chunks.append(runs[(self._nchunks)*self._chunksize:])
        # we define the parallel function
        @parallel
        def f(indices):
            results = []
            for frun in indices:
                for i in range(len(self._benchmarks)):
                    bm = self._benchmarks[i]
                    res = bm.run(frun)
                    results.append((i, res))
            return results
        self._getchunks = f(self._chunks)
        self._currentchunk = []

    def __iter__(self):
        return self

    def next(self):
        if not self._currentchunk:
            self._currentchunk = self._getchunks.next()[1]
        res = self._currentchunk.pop()
        return res

    __next__ = next




if __name__ == "__main__":
    import doctest
    doctest.testmod()
