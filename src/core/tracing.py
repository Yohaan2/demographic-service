import os
from typing import Optional, TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:  # pragma: no cover
    from opentelemetry.sdk.trace import TracerProvider

_TRUTHY = {"true", "1", "yes"}


def tracing_disabled() -> bool:
    """OpenTelemetry is disabled when OTEL_SDK_DISABLED is "true", "1" or "yes" (case-insensitive)."""
    return os.getenv("OTEL_SDK_DISABLED", "").strip().lower() in _TRUTHY


def configure_tracing() -> "TracerProvider":
    # Imported lazily so that a disabled SDK never loads gRPC / the exporter.
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    service_name = os.getenv("OTEL_SERVICE_NAME", "demographic-service")
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=endpoint,
                insecure=not endpoint.startswith("https://"),
            )
        )
    )
    trace.set_tracer_provider(provider)
    return provider


tracer_provider: Optional["TracerProvider"] = None if tracing_disabled() else configure_tracing()


def instrument_fastapi(app: FastAPI) -> None:
    if tracer_provider is None:
        return  # no-op: OTEL_SDK_DISABLED
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)


def shutdown_tracing() -> None:
    if tracer_provider is None:
        return  # no-op: OTEL_SDK_DISABLED
    tracer_provider.shutdown()
