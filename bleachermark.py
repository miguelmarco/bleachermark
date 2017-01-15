r"""
The Bleachermark package

A Bleachermark is a collection of benchmarks. A benchmark is a single pipeline
of functions, often randomized in one or more functions, whose performance and
data one wishes to perform statistics on.

EXAMPLE::

    TODO
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


class Scheduler(object):
    r"""
    A Scheduler is basically an iterator which returns Benchmark runs.

    A Benchmark is the simplest type of Scheduler which simply infinitely emits
    a run of this Benchmark.
    """
    def __init__(self):
        self._started_runs = set()
        self._completed_runs = set()

    def __iter__(self):
        return self

    def _next(self):
        # Override this in subclasses
        raise NotImplemented

    def next(self):
        r"""
        Return the next run from this scheduler, or raise a `StopIteration`.
        """
        run = self._next()
        self._started_runs.add(run)
        return run

    __next__ = next

    def register_completed(self, run, result):
        r"""
        Register to this scheduler (and any children) that the given run was completed.
        """
        try:
            self._started_runs.remove(run)
            self._pass_register_completed(self, run, result)
            self._completed_runs.add(run)
        except KeyError:
            #TODO: How to handle this?
            raise ValueError("Run was not started by this scheduler (%s) or already completed: %s" % (self, run))

    def _pass_register_completed(self, run, result):
        r"""
        Pass a call to `self.register_completed` to any sub-schedulers, and
        possibly do special handling in relation to register completed.
        """
        # Override this in subclasses
        raise NotImplemented


class RunN(Scheduler):
    r"""
    Returns at most `n` runs of the input scheduler.
    """
    def __init__(self, scheduler, n):
        self._n = n
        self._scheduler = scheduler
        super(RunN, self).__init__()

    def _next(self):
        if len(self._completed_runs) + len(self._started_runs) >= self._n:
            raise StopIteration
        else:
            return self._scheduler.next()

    def _pass_register_completed(self, run, result):
        self._scheduler.register_completed(run, result)


class AggregateScheduler(Scheduler):
    r"""
    Base class for schedulers which aggregate a finite list of schedulers in some way.

    Sub-classes' `_next()` method should return both a run and the scheduler
    which issued it.
    """
    def __init__(self, schedulers):
        self._schedulers = schedulers
        self._which_scheduler = dict() # map not-completed-run -> scheduler

    def next(self):
        r"""
        Return the next run from this scheduler, or raise a `StopIteration`.
        """
        (run, scheduler) = self._next()
        self._which_scheduler[run] = scheduler
        self._started_runs.add(run)
        return run

    def _pass_register_completed(self, run, result):
        self._which_scheduler[run].register_completed(run, result)
        del self._which_scheduler[run]



class Sequential(Scheduler):
    r"""
    Run an interable of schedulers in sequence: that is, the first scheduler is
    completely exhausted before proceeding to the second scheduler.

    The iterable of schedulers can be infinite.
    """
    def __init__(self, schedulers):
        self._current = None
        super(Sequential, self).__init__(iter(schedulers))

    def _next(self):
        if self._current is None:
            self._current = self._schedulers.next()
        else:
            try:
                run = self._current.next()
                return (run, self._current)
            except StopIteration:
                self._current = None
                return self.next()


class BalanceRuns(Scheduler):
    r"""
    Run an iterable of schedulers, balancing how many runs each scheduler is run.

    The iterable of schedulers must be finite.
    """
    def __init__(self, schedulers):
        self._next_scheduler = 0
        self._stop_loop = 0
        super(BalanceRuns, self).__init__(list(schedulers))

    def _next(self):
        try:
            scheduler = self._schedulers[self._next_scheduler]
            run = scheduler.next()
            self._next_scheduler = (self._next_scheduler + 1) % len(self._schedulers)
            return (run, scheduler)
        except StopIteration:
            del self._schedulers[self._next_scheduler]
            if self._next_scheduler == len(self._schedulers):
                self._next_scheduler = 0
            if self._schedulers:
                return self._next()
            else:
                raise StopIteration



class BalanceTime(Scheduler):
    r"""
    Run an iterable of schedulers, balancing how much time each scheduler is run.

    The iterable of schedulers must be finite.

    TODO: This should use a dynamic priority queue for asymptotically better
    complexity.
    """
    def __init__(self, schedulers):
        super(BalanceTime, self).__init__(list(schedulers))
        self._scheduler_issued         = { s : 0  for s in self._schedulers }
        self._scheduler_completed      = { s : 0  for s in self._schedulers }
        self._scheduler_completed_time = { s : 0. for s in self._schedulers }
        self._scheduler_expected_time  = { s : 0. for s in self._schedulers }

    def _update_expected_time(self, scheduler):
        avg = self._scheduler_completed_time[scheduler] / self._scheduler_completed[scheduler]
        self._scheduler_expected_time[scheduler] = avg * self._scheduler_issued[scheduler]

    def _next(self):
        try:
            #NOTE: self._scheduler contains those schedulers that are not yet
            #(known to be) finished. The dictionaries have mappings to also old
            #schedulers, so we should take care that we here only minimise over
            #the non-empty ones.
            scheduler = min(self._schedulers, key=lambda s: self._scheduler_timings[s])
            run = scheduler.next()
            self._scheduler_issued[scheduler] += 1
            self._update_expected_time(scheduler)
            return (run, scheduler)
        except StopIteration:
            del self._schedulers[index]
            del self._schedulers_timings[index]
            if self._schedulers:
                return self._next()
            else:
                raise StopIteration

    def _pass_register_completed(self, run, result):
        scheduler = self._which_scheduler[run]
        self._scheduler_completed[scheduler] += 1
        self._scheduler_completed_time[scheduler] += result.total_time()
        self._update_expected_time(scheduler)
        super(BalanceTime, self)._pass_register_completed(run, result)



class Benchmark(Scheduler):
    r"""
    A Benchmark is a pipeline of functions which can be run.
    As a Scheduler, it omits runs for itself indefinitely.
    """
    def __init__(self, pipeline, label):
        r"""
        Initializes the Benchmark

        INPUT:

        -  `pipeline` - the list of functions in the Benchmark.

        - `label` - A name or ID for this benchmark.

        """
        self._pipeline = tuple(pipeline)
        self._label = label
        super(Benchmark, self).__init__()
        self._runid = 0

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

    def set_label(self, label):
        self._label = label

    def pipeline(self):
        r"""
        Return the pipeline of functions of this benchmark.
        """
        return self._pipeline

    def _next(self):
        r"""
        For the Scheduler interface
        """
        return (self._label, self._next_runid())

    def _next_runid():
        runid = self._runid
        self._runid += 1
        return runid

    def run(self, runid):
        r"""
        Run the pipeline and return the timings and values produced.

        INPUT:

        - `runid` - An identifier numbering the run. This is intended only for
                    e.g. setting a random seed. Possibly ignore it.

        OUTPUT:

        - A Result object for the timing results.
        """
        result = Result(self, runid)
        data = runid
        for fun in self._pipeline:
            before = clock()
            data = fun(data)
            elapsed = clock() - before
            result.log_pipeline(data, elapsed)
        return result


def _make_autolabel(n):
    r"""
    Return a generic label for a benchmark with count n
    """
    return "[%s]" % n

_autolabel_regex = r"\[[0-9]*\]"



class Result(object):

    def __init__(self, benchmark, runid):
        self._benchmark = benchmark
        self._runid = runid
        self._data = [ ]
        self._timings = [ ]

    def log_pipeline(self, data, elapsed):
        self._data.append(data)
        self._timings.append(elapsed)

    def total_time(self):
        return sum(self._timings)





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
                b._set_label(_make_autolabel(n))  # WARNING!!! This could have side effects if the same benchmark is in more than one bleachermark!!!
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

        - ``**kwargs`` - the keyword arguments to be passed to the constructor of the runner.

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
            (1, (0, (3.999999999892978e-06, 0), (2.9999999999752447e-06, 0)))

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
        Runs the benchmarks in the collection and stores the returned data.

        TODO: Alias of `resume`?

        INPUT:

        - ``nruns`` - The number of times each
        """
        if self._current_runner is None:
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

         - ``transposed`` - (default = False), determines wether the data must
            be transposed. TODO: Explain what this means.

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

        TODO: Better name?
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
    It most likely will be the default one
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

class AdaptativeRunner:
    r"""
    Runner that decides the runs to make adaptitvely to produce
    the best possible overview of how does the timing of each benchmark
    deppend on its input.

    It is assumed that the runid will correspond to the size of the input

    For the moment, the strategy followed is the following: try the measures to
    expand aa range as big as possible, but making sure that two consecutive runid's
    have timings that don't differ by a factor bigger than 1.5
    (unless they are consecutive integers).
    """
    def __init__(self, bleachermark, totruns = 100):
        self._benchmarks=bleachermark._benchmarks
        self._timings = {i:[] for i in range(len(self._benchmarks))}
        self._curbm = 0
        self._pendingruns = totruns

    def repr(self):
        return "Adaptative Runner for {} benchmarks".format(len(self._benchmarks))

    def __iter__(self):
        return self

    def next(self):
        if self._pendingruns < 0:
            raise StopIteration
        self._pendingruns -= 1
        i = self._curbm
        # this part decides j, which is the next runid to run
        if not self._timings[i]:
            j = 1
            ii = 0
        elif len(self._timings[i]) == 1: # we need at least values for 1 and 2
            j = 2
            ii = 1
        else: #now we check if two consecutive runids have enough space between
            for ii in range(len(self._timings[i])):
                if ii == len(self._timings[i])-1:
                    j = 2 * self._timings[i][-1][0]
                else:
                    i0, t0 = self._timings[i][ii]
                    i1, t1 = self._timings[i][ii+1]
                    if (i1 > i0+1) and (t1/t0 > 1.5 or t0/t1 > 1.5):
                        j = int((i0 + i1)/2)
                        break
        # now we run, store and return it
        res = self._benchmarks[i].run(j)
        tottime = sum([r[0] for r in res[1:]])
        self._timings[i].insert(ii+1, (j, tottime))
        self._curbm = (i+1) % len(self._benchmarks)
        return (i, res)

    __next__ = next



if __name__ == "__main__":
    import doctest
    doctest.testmod()
