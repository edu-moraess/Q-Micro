"""
Performance utilities for Q-Micro.
Includes benchmarking, latency measurement, and throughput analysis.
"""

import time
import numpy as np
from typing import Callable, List, Dict, Optional, Any
from dataclasses import dataclass
import numba
import polars as pl

@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    function_name: str
    execution_time: float  # in seconds
    iterations: int
    avg_time: float        # average time per iteration (in seconds)
    std_time: float         # standard deviation of time per iteration
    throughput: float       # iterations per second

class PerformanceBenchmark:
    """Benchmark performance of functions."""

    @staticmethod
    def benchmark(
        func: Callable,
        iterations: int = 1000,
        warmup: int = 100,
        *args,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Benchmark a function by measuring its execution time.

        Args:
            func: Function to benchmark.
            iterations: Number of iterations to run.
            warmup: Number of warmup iterations (not counted).
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            BenchmarkResult with timing statistics.
        """
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)

        # Benchmark
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        times = np.array(times)

        return BenchmarkResult(
            function_name=func.__name__,
            execution_time=times.sum(),
            iterations=iterations,
            avg_time=times.mean(),
            std_time=times.std(),
            throughput=iterations / times.sum(),
        )

    @staticmethod
    def compare(
        funcs: List[Callable],
        iterations: int = 1000,
        warmup: int = 100,
        *args,
        **kwargs,
    ) -> List[BenchmarkResult]:
        """
        Compare performance of multiple functions.

        Args:
            funcs: List of functions to compare.
            iterations: Number of iterations to run.
            warmup: Number of warmup iterations.
            *args: Positional arguments to pass to the functions.
            **kwargs: Keyword arguments to pass to the functions.

        Returns:
            List of BenchmarkResult objects.
        """
        results = []
        for func in funcs:
            result = PerformanceBenchmark.benchmark(func, iterations, warmup, *args, **kwargs)
            results.append(result)
        return results

    @staticmethod
    def print_benchmark_results(results: List[BenchmarkResult]) -> None:
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 80)
        print(f"{'Function':<30} {'Avg Time (μs)':<15} {'Throughput (ops/s)':<20} {'Std Dev (μs)':<15}")
        print("-" * 80)
        for result in results:
            print(
                f"{result.function_name:<30} "
                f"{result.avg_time * 1e6:<15.2f} "
                f"{result.throughput:<20.0f} "
                f"{result.std_time * 1e6:<15.2f}"
            )
        print("=" * 80 + "\n")

@dataclass
class LatencyResult:
    """Result of a latency measurement."""
    function_name: str
    latency: float  # in seconds

class LatencyMeter:
    """Measure latency of functions."""

    @staticmethod
    def measure_latency(func: Callable, *args, **kwargs) -> LatencyResult:
        """
        Measure the latency of a single function call.

        Args:
            func: Function to measure.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            LatencyResult with the measured latency.
        """
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()

        return LatencyResult(
            function_name=func.__name__,
            latency=end_time - start_time,
        )

@dataclass
class ThroughputResult:
    """Result of a throughput measurement."""
    function_name: str
    total_time: float  # in seconds
    total_operations: int
    throughput: float   # operations per second

class ThroughputMeter:
    """Measure throughput of functions."""

    @staticmethod
    def measure_throughput(
        func: Callable,
        total_operations: int = 10000,
        *args,
        **kwargs,
    ) -> ThroughputResult:
        """
        Measure the throughput of a function.

        Args:
            func: Function to measure.
            total_operations: Number of operations to perform.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            ThroughputResult with the measured throughput.
        """
        start_time = time.perf_counter()
        for _ in range(total_operations):
            func(*args, **kwargs)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        throughput = total_operations / total_time

        return ThroughputResult(
            function_name=func.__name__,
            total_time=total_time,
            total_operations=total_operations,
            throughput=throughput,
        )

# Convenience functions
def benchmark(func: Callable, iterations: int = 1000, *args, **kwargs) -> BenchmarkResult:
    """Benchmark a function."""
    return PerformanceBenchmark.benchmark(func, iterations, *args, **kwargs)

def measure_latency(func: Callable, *args, **kwargs) -> LatencyResult:
    """Measure latency of a function."""
    return LatencyMeter.measure_latency(func, *args, **kwargs)

def measure_throughput(func: Callable, total_operations: int = 10000, *args, **kwargs) -> ThroughputResult:
    """Measure throughput of a function."""
    return ThroughputMeter.measure_throughput(func, total_operations, *args, **kwargs)
