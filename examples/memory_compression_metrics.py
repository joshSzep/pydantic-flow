"""Memory Compression - Metrics Collection and Monitoring.

This example demonstrates how to collect, analyze, and monitor compression
metrics for performance tracking, quality assurance, and optimization.

Key concepts:
1. Collecting metrics from MemoryCompressionComplete events
2. Aggregating metrics across multiple compressions
3. Generating performance reports
4. Monitoring compression quality
5. Detecting anomalies and optimization opportunities

Run with: uv run python examples/memory_compression_metrics.py
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from statistics import mean
from statistics import median
from statistics import stdev
from typing import Any

from pydantic import BaseModel

from pydantic_flow import CompressionMetrics
from pydantic_flow import Flow
from pydantic_flow import MemoryConfig
from pydantic_flow import SlidingWindowCompressor
from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.memory.events import MemoryCompressionComplete
from pydantic_flow.streaming.base import ProgressItem


# Helper to extract result from stream
async def extract_result_from_stream(stream):
    """Extract final result from async stream of progress items."""
    result = None
    async for item in stream:
        if hasattr(item, "result"):
            result = item.result
    return result


class ChatInput(BaseModel):
    """User message input."""

    message: str


class ChatOutput(BaseModel):
    """Agent response output."""

    response: str


# ============================================================================
# Metrics Collection Infrastructure
# ============================================================================


@dataclass
class CompressionStats:
    """Aggregated compression statistics.

    Tracks metrics across multiple compression operations for analysis.
    """

    total_compressions: int = 0
    total_messages_before: int = 0
    total_messages_after: int = 0
    total_tokens_saved: int = 0
    compression_times: list[float] = field(default_factory=list)
    compression_ratios: list[float] = field(default_factory=list)
    strategies_used: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    metrics_history: list[CompressionMetrics] = field(default_factory=list)

    def add_metrics(self, metrics: CompressionMetrics) -> None:
        """Add metrics from a compression operation.

        Args:
            metrics: Compression metrics to add to statistics.

        """
        self.total_compressions += 1
        self.total_messages_before += metrics.messages_before
        self.total_messages_after += metrics.messages_after
        self.total_tokens_saved += metrics.tokens_saved
        self.compression_times.append(metrics.compression_time_ms)
        self.compression_ratios.append(metrics.compression_ratio)
        self.strategies_used[metrics.compression_strategy] += 1
        self.metrics_history.append(metrics)

    @property
    def avg_compression_ratio(self) -> float:
        """Calculate average compression ratio."""
        return mean(self.compression_ratios) if self.compression_ratios else 0.0

    @property
    def median_compression_ratio(self) -> float:
        """Calculate median compression ratio."""
        return median(self.compression_ratios) if self.compression_ratios else 0.0

    @property
    def avg_compression_time(self) -> float:
        """Calculate average compression time in milliseconds."""
        return mean(self.compression_times) if self.compression_times else 0.0

    @property
    def median_compression_time(self) -> float:
        """Calculate median compression time in milliseconds."""
        return median(self.compression_times) if self.compression_times else 0.0

    @property
    def total_tokens_reduction(self) -> float:
        """Calculate total percentage reduction in tokens."""
        if self.total_messages_before == 0:
            return 0.0
        return (1.0 - (self.total_messages_after / self.total_messages_before)) * 100

    def generate_report(self) -> str:
        """Generate a formatted metrics report.

        Returns:
            Formatted string containing compression statistics.

        """
        lines = [
            "=" * 70,
            "Compression Metrics Report",
            "=" * 70,
            "",
            "Summary:",
            f"  Total Compressions: {self.total_compressions}",
            f"  Messages Before: {self.total_messages_before}",
            f"  Messages After: {self.total_messages_after}",
            f"  Total Tokens Saved: {self.total_tokens_saved}",
            f"  Overall Reduction: {self.total_tokens_reduction:.1f}%",
            "",
            "Compression Ratios:",
            f"  Average: {self.avg_compression_ratio:.3f}",
            f"  Median: {self.median_compression_ratio:.3f}",
            f"  Min: {min(self.compression_ratios):.3f}"
            if self.compression_ratios
            else "  Min: N/A",
            f"  Max: {max(self.compression_ratios):.3f}"
            if self.compression_ratios
            else "  Max: N/A",
        ]

        if len(self.compression_ratios) > 1:
            std = stdev(self.compression_ratios)
            lines.append(f"  Std Dev: {std:.3f}")

        lines.extend([
            "",
            "Performance:",
            f"  Average Time: {self.avg_compression_time:.2f}ms",
            f"  Median Time: {self.median_compression_time:.2f}ms",
            f"  Min Time: {min(self.compression_times):.2f}ms"
            if self.compression_times
            else "  Min Time: N/A",
            f"  Max Time: {max(self.compression_times):.2f}ms"
            if self.compression_times
            else "  Max Time: N/A",
        ])

        if len(self.compression_times) > 1:
            std = stdev(self.compression_times)
            lines.append(f"  Std Dev: {std:.2f}ms")

        lines.extend([
            "",
            "Strategies Used:",
        ])

        for strategy, count in sorted(
            self.strategies_used.items(), key=lambda x: x[1], reverse=True
        ):
            pct = (
                (count / self.total_compressions * 100)
                if self.total_compressions > 0
                else 0
            )
            lines.append(f"  {strategy}: {count} ({pct:.1f}%)")

        lines.append("=" * 70)

        return "\n".join(lines)

    def detect_anomalies(self) -> list[str]:
        """Detect potential issues or anomalies in compression metrics.

        Returns:
            List of warning messages about detected anomalies.

        """
        warnings = []

        # Check for low compression ratios
        if self.avg_compression_ratio > 0.8:
            warnings.append(
                f"⚠️  Low compression efficiency: {self.avg_compression_ratio:.2f} "
                "(consider different strategy)"
            )

        # Check for high compression times
        if self.avg_compression_time > 100:
            warnings.append(
                f"⚠️  Slow compression: {self.avg_compression_time:.1f}ms average "
                "(may impact performance)"
            )

        # Check for high variance in compression ratios
        if len(self.compression_ratios) > 3:
            std = stdev(self.compression_ratios)
            if std > 0.2:
                warnings.append(
                    f"⚠️  High variance in compression ratios: σ={std:.3f} "
                    "(inconsistent behavior)"
                )

        # Check for high variance in compression times
        if len(self.compression_times) > 3:
            std = stdev(self.compression_times)
            if std > 50:
                warnings.append(
                    f"⚠️  High variance in compression times: σ={std:.1f}ms "
                    "(unpredictable performance)"
                )

        return warnings


# ============================================================================
# Example Implementations
# ============================================================================


async def example_1_basic_metrics_collection():
    """Demonstrate basic metrics collection from compression events."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Metrics Collection")
    print("=" * 70)

    stats = CompressionStats()

    async def collect_metrics(item: ProgressItem) -> InterruptDecision:
        """Collect metrics from compression events."""
        if isinstance(item, MemoryCompressionComplete):
            stats.add_metrics(item.metrics)

            print(f"\n📊 Compression #{stats.total_compressions}:")
            print(f"   Strategy: {item.metrics.compression_strategy}")
            print(
                f"   Messages: {item.metrics.messages_before} → {item.metrics.messages_after}"
            )
            print(f"   Ratio: {item.metrics.compression_ratio:.3f}")
            print(f"   Time: {item.metrics.compression_time_ms:.2f}ms")
            print(f"   Tokens saved: {item.metrics.tokens_saved}")

        return InterruptDecision.proceed()

    compressor = SlidingWindowCompressor(window_size=5, max_tokens=100)

    _ = Flow[ChatInput, ChatOutput](
        input_type=ChatInput,
        output_type=ChatOutput,
        memory_config=MemoryConfig(
            enable_conversation_memory=True,
            compressor=compressor,
            emit_compression_events=True,
        ),
    )

    print("\nSetup: Collect metrics from each compression event")
    print("Handler: Tracks compression operations in CompressionStats object")

    print("\n" + "=" * 70)
    print("In production, you would:")
    print("  • Pass collect_metrics to flow.astream(interrupt=...)")
    print("  • Stats accumulate across multiple compressions")
    print("  • Generate reports periodically or on demand")
    print("=" * 70)


async def example_2_aggregated_reporting():
    """Demonstrate aggregated metrics reporting."""
    print("\n" + "=" * 70)
    print("Example 2: Aggregated Metrics Reporting")
    print("=" * 70)

    # Simulate multiple compressions with varying metrics
    stats = CompressionStats()

    print("\nSimulating 10 compression operations...")

    # Simulate varied compression results
    simulated_metrics = [
        CompressionMetrics(
            messages_before=50,
            messages_after=10,
            estimated_tokens_before=5000,
            estimated_tokens_after=1000,
            tokens_saved=4000,
            compression_ratio=0.2,
            compression_strategy="sliding_window_10",
            compression_time_ms=0.5,
        ),
        CompressionMetrics(
            messages_before=75,
            messages_after=15,
            estimated_tokens_before=7500,
            estimated_tokens_after=1500,
            tokens_saved=6000,
            compression_ratio=0.2,
            compression_strategy="sliding_window_10",
            compression_time_ms=0.7,
        ),
        CompressionMetrics(
            messages_before=100,
            messages_after=20,
            estimated_tokens_before=10000,
            estimated_tokens_after=2000,
            tokens_saved=8000,
            compression_ratio=0.2,
            compression_strategy="sliding_window_10",
            compression_time_ms=0.9,
        ),
        CompressionMetrics(
            messages_before=30,
            messages_after=10,
            estimated_tokens_before=3000,
            estimated_tokens_after=1000,
            tokens_saved=2000,
            compression_ratio=0.33,
            compression_strategy="summarization",
            compression_time_ms=150.0,
        ),
        CompressionMetrics(
            messages_before=45,
            messages_after=12,
            estimated_tokens_before=4500,
            estimated_tokens_after=1200,
            tokens_saved=3300,
            compression_ratio=0.27,
            compression_strategy="summarization",
            compression_time_ms=180.0,
        ),
    ]

    for i, metrics in enumerate(simulated_metrics, 1):
        stats.add_metrics(metrics)
        print(f"  ✓ Compression {i}: {metrics.compression_strategy}")

    print("\n" + stats.generate_report())

    # Check for anomalies
    warnings = stats.detect_anomalies()
    if warnings:
        print("\n⚠️  Anomalies Detected:")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("\n✅ No anomalies detected - compression performing well")


async def example_3_real_time_monitoring():
    """Demonstrate real-time monitoring with threshold alerts."""
    print("\n" + "=" * 70)
    print("Example 3: Real-Time Monitoring with Alerts")
    print("=" * 70)

    stats = CompressionStats()

    # Define thresholds
    COMPRESSION_RATIO_THRESHOLD = 0.5  # Alert if > 0.5 (poor compression)
    COMPRESSION_TIME_THRESHOLD = 100.0  # Alert if > 100ms
    TOKENS_SAVED_MIN = 1000  # Alert if < 1000 tokens saved

    async def monitor_compression(item: ProgressItem) -> InterruptDecision:
        """Monitor compression with real-time alerts."""
        if isinstance(item, MemoryCompressionComplete):
            metrics = item.metrics
            stats.add_metrics(metrics)

            print(f"\n🔍 Compression #{stats.total_compressions}:")
            print(f"   Strategy: {metrics.compression_strategy}")

            # Check compression ratio
            if metrics.compression_ratio > COMPRESSION_RATIO_THRESHOLD:
                print(f"   ⚠️  Ratio: {metrics.compression_ratio:.3f} (POOR)")
            else:
                print(f"   ✓ Ratio: {metrics.compression_ratio:.3f} (GOOD)")

            # Check compression time
            if metrics.compression_time_ms > COMPRESSION_TIME_THRESHOLD:
                print(f"   ⚠️  Time: {metrics.compression_time_ms:.2f}ms (SLOW)")
            else:
                print(f"   ✓ Time: {metrics.compression_time_ms:.2f}ms (FAST)")

            # Check tokens saved
            if metrics.tokens_saved < TOKENS_SAVED_MIN:
                print(f"   ⚠️  Saved: {metrics.tokens_saved} tokens (LOW)")
            else:
                print(f"   ✓ Saved: {metrics.tokens_saved} tokens (GOOD)")

        return InterruptDecision.proceed()

    print("\nMonitoring Configuration:")
    print(f"  Max Compression Ratio: {COMPRESSION_RATIO_THRESHOLD}")
    print(f"  Max Compression Time: {COMPRESSION_TIME_THRESHOLD}ms")
    print(f"  Min Tokens Saved: {TOKENS_SAVED_MIN}")

    print("\n" + "=" * 70)
    print("In production:")
    print("  • Alerts can trigger logging, notifications, or fallbacks")
    print("  • Thresholds configurable per environment")
    print("  • Metrics exported to monitoring systems (Prometheus, etc.)")
    print("=" * 70)


async def example_4_performance_optimization():
    """Demonstrate using metrics for performance optimization."""
    print("\n" + "=" * 70)
    print("Example 4: Performance Optimization Analysis")
    print("=" * 70)

    stats = CompressionStats()

    # Simulate mixed strategy performance
    strategies_data = {
        "sliding_window_5": [
            (40, 5, 4000, 500, 0.5, 0.3),
            (45, 5, 4500, 500, 0.5, 0.4),
            (50, 5, 5000, 500, 0.5, 0.3),
        ],
        "sliding_window_10": [
            (40, 10, 4000, 1000, 0.25, 0.6),
            (45, 10, 4500, 1000, 0.25, 0.5),
            (50, 10, 5000, 1000, 0.25, 0.7),
        ],
        "summarization": [
            (40, 15, 4000, 1500, 0.38, 120.0),
            (45, 18, 4500, 1800, 0.40, 150.0),
            (50, 20, 5000, 2000, 0.40, 180.0),
        ],
    }

    print("\nSimulating different compression strategies...")

    for strategy, data in strategies_data.items():
        print(f"\n  Testing: {strategy}")
        for msg_before, msg_after, tok_before, tok_after, ratio, time_ms in data:
            metrics = CompressionMetrics(
                messages_before=msg_before,
                messages_after=msg_after,
                estimated_tokens_before=tok_before,
                estimated_tokens_after=tok_after,
                tokens_saved=tok_before - tok_after,
                compression_ratio=ratio,
                compression_strategy=strategy,
                compression_time_ms=time_ms,
            )
            stats.add_metrics(metrics)

    # Analyze performance by strategy
    print("\n" + "=" * 70)
    print("Strategy Performance Analysis:")
    print("=" * 70)

    strategy_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"ratios": [], "times": [], "tokens_saved": []}
    )

    for metrics in stats.metrics_history:
        strategy_stats[metrics.compression_strategy]["ratios"].append(
            metrics.compression_ratio
        )
        strategy_stats[metrics.compression_strategy]["times"].append(
            metrics.compression_time_ms
        )
        strategy_stats[metrics.compression_strategy]["tokens_saved"].append(
            metrics.tokens_saved
        )

    for strategy, data in sorted(strategy_stats.items()):
        avg_ratio = mean(data["ratios"])
        avg_time = mean(data["times"])
        avg_saved = mean(data["tokens_saved"])

        print(f"\n{strategy}:")
        print(f"  Avg Compression Ratio: {avg_ratio:.3f}")
        print(f"  Avg Time: {avg_time:.2f}ms")
        print(f"  Avg Tokens Saved: {avg_saved:.0f}")

        # Recommendation
        if avg_ratio < 0.3 and avg_time < 10:
            print("  ✅ Recommended: Excellent balance of quality and speed")
        elif avg_ratio < 0.3:
            print(
                "  ⚠️  Good compression but slow - consider for quality-critical paths"
            )
        elif avg_time < 10:
            print("  ⚠️  Fast but poor compression - consider for speed-critical paths")
        else:
            print("  ❌ Not recommended: Poor compression and slow")

    print("\n" + "=" * 70)
    print("Optimization Insights:")
    print("  • sliding_window_5: Fast but saves fewer tokens")
    print("  • sliding_window_10: Good balance for most use cases")
    print("  • summarization: Best compression but slowest")
    print("=" * 70)


async def best_practices():
    """Display best practices for metrics collection."""
    print("\n" + "=" * 70)
    print("Metrics Collection Best Practices")
    print("=" * 70)

    print("\n1. Collection Strategy:")
    print("   • Use interrupt handlers to capture MemoryCompressionComplete events")
    print("   • Store metrics in persistent storage for long-term analysis")
    print("   • Aggregate metrics by time period, user, flow, etc.")
    print("   • Include context (user_id, flow_id) in metadata")

    print("\n2. Key Metrics to Track:")
    print("   • Compression ratio (quality indicator)")
    print("   • Compression time (performance indicator)")
    print("   • Tokens saved (efficiency indicator)")
    print("   • Messages before/after (context preservation)")
    print("   • Strategy used (for comparison)")

    print("\n3. Monitoring and Alerts:")
    print("   • Set thresholds for acceptable performance")
    print("   • Alert on anomalies (sudden changes, poor performance)")
    print("   • Track trends over time")
    print("   • Compare against baselines")

    print("\n4. Optimization Workflow:")
    print("   • Collect baseline metrics")
    print("   • Experiment with different strategies")
    print("   • Compare performance across strategies")
    print("   • Choose optimal strategy for use case")
    print("   • Continuously monitor and adjust")

    print("\n5. Integration with Monitoring Systems:")
    print("   • Export metrics to Prometheus, Datadog, etc.")
    print("   • Create dashboards for visualization")
    print("   • Set up automated alerts")
    print("   • Track metrics across deployments")

    print("\n6. Privacy and Compliance:")
    print("   • Avoid logging sensitive message content")
    print("   • Aggregate metrics to protect user privacy")
    print("   • Comply with data retention policies")
    print("   • Anonymize user identifiers where appropriate")


async def main():
    """Run all metrics collection examples."""
    print("\n" + "=" * 70)
    print("Memory Compression - Metrics Collection and Monitoring")
    print("=" * 70)
    print("\nThis demonstrates metrics collection and monitoring:")
    print("1. Basic metrics collection from events")
    print("2. Aggregated reporting across compressions")
    print("3. Real-time monitoring with alerts")
    print("4. Performance optimization analysis")

    await example_1_basic_metrics_collection()
    await example_2_aggregated_reporting()
    await example_3_real_time_monitoring()
    await example_4_performance_optimization()
    await best_practices()

    print("\n" + "=" * 70)
    print("For more examples, see:")
    print("  • memory_compression_basic.py - Basic compression strategies")
    print("  • memory_compression_approval.py - HITL approval workflows")
    print("  • memory_compression_custom.py - Custom compressor implementation")
    print("\nFor metrics documentation, see: src/pydantic_flow/memory/compression.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
