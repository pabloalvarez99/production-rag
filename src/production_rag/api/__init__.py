"""HTTP layer: schemas, dependencies, middleware and routers.

Nothing in this subpackage knows how retrieval works. Routes translate between
HTTP and the (future) service layer, which keeps the transport swappable — the
same services will be driven by a CLI and by the evaluation harness.
"""
