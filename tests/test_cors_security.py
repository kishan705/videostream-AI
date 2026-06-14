"""
test_cors_security.py — Comprehensive CORS configuration tests.

Validates that the fix for the insecure CORS configuration
(allow_origins=["*"] + allow_credentials=True) is working correctly.

Test matrix:
  1. Wildcard origin is NOT reflected
  2. Allowed origin IS reflected with credentials
  3. Disallowed origin is rejected
  4. Only permitted HTTP methods are allowed
  5. Only permitted headers are allowed
  6. Preflight (OPTIONS) requests behave correctly
  7. Multi-origin environment variable is parsed correctly
  8. Default fallback origin works when env var is unset
"""
import pytest
from httpx import AsyncClient, ASGITransport

pytestmark = pytest.mark.asyncio


# ============================================================================
# 1. Core Security — Wildcard is gone
# ============================================================================

class TestWildcardOriginBlocked:
    """Verify that the old `allow_origins=["*"]` behaviour is eliminated."""

    async def test_arbitrary_origin_not_reflected(self, default_app):
        """A random attacker origin must NOT be echoed back."""
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/",
                headers={"Origin": "https://evil-attacker.com"},
            )
        assert resp.status_code == 200
        assert "access-control-allow-origin" not in resp.headers

    async def test_wildcard_star_not_in_response(self, default_app):
        """Ensure the literal '*' value never appears in allow-origin."""
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/",
                headers={"Origin": "http://localhost:3000"},
            )
        assert resp.headers.get("access-control-allow-origin") != "*"


# ============================================================================
# 2. Allowed Origin — Happy Path
# ============================================================================

class TestAllowedOriginReflected:
    """Confirm legitimate origins receive proper CORS headers."""

    async def test_default_localhost_origin_allowed(self, default_app):
        """The default origin (http://localhost:3000) must be reflected."""
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/",
                headers={"Origin": "http://localhost:3000"},
            )
        assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"

    async def test_credentials_header_present(self, default_app):
        """allow_credentials=True must produce the header for allowed origins."""
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/",
                headers={"Origin": "http://localhost:3000"},
            )
        assert resp.headers.get("access-control-allow-credentials") == "true"

    async def test_origin_not_reflected_for_disallowed_origin(self, default_app):
        """The access-control-allow-origin header must NOT appear for disallowed origins.
        
        Note: Starlette's CORSMiddleware sets allow-credentials globally, but
        browsers enforce the CORS block based on the absence of allow-origin.
        Without allow-origin, the credentials header is inert.
        """
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/",
                headers={"Origin": "https://evil.com"},
            )
        assert "access-control-allow-origin" not in resp.headers


# ============================================================================
# 3. Multi-Origin Environment Variable
# ============================================================================

class TestMultiOriginEnvParsing:
    """Verify comma-separated ALLOWED_ORIGINS env var is parsed correctly."""

    async def test_first_origin_allowed(self, multi_origin_app):
        transport = ASGITransport(app=multi_origin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/",
                headers={"Origin": "https://prod.example.com"},
            )
        assert resp.headers["access-control-allow-origin"] == "https://prod.example.com"

    async def test_second_origin_allowed(self, multi_origin_app):
        transport = ASGITransport(app=multi_origin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/",
                headers={"Origin": "https://staging.example.com"},
            )
        assert resp.headers["access-control-allow-origin"] == "https://staging.example.com"

    async def test_unlisted_origin_rejected_in_multi(self, multi_origin_app):
        transport = ASGITransport(app=multi_origin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/",
                headers={"Origin": "https://unknown.example.com"},
            )
        assert "access-control-allow-origin" not in resp.headers


# ============================================================================
# 4. HTTP Method Restrictions
# ============================================================================

class TestAllowedMethods:
    """Only GET, POST, DELETE should pass preflight. Others must be refused."""

    async def test_preflight_allows_get(self, default_app):
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.options(
                "/",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                },
            )
        allowed = resp.headers.get("access-control-allow-methods", "")
        assert "GET" in allowed

    async def test_preflight_allows_post(self, default_app):
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.options(
                "/upload",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                },
            )
        allowed = resp.headers.get("access-control-allow-methods", "")
        assert "POST" in allowed

    async def test_preflight_allows_delete(self, default_app):
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.options(
                "/item/123",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "DELETE",
                },
            )
        allowed = resp.headers.get("access-control-allow-methods", "")
        assert "DELETE" in allowed

    async def test_preflight_rejects_put(self, default_app):
        """PUT is not in the allow list — preflight must not include it."""
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.options(
                "/",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "PUT",
                },
            )
        allowed = resp.headers.get("access-control-allow-methods", "")
        assert "PUT" not in allowed

    async def test_preflight_rejects_patch(self, default_app):
        """PATCH is not in the allow list — preflight must not include it."""
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.options(
                "/",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "PATCH",
                },
            )
        allowed = resp.headers.get("access-control-allow-methods", "")
        assert "PATCH" not in allowed


# ============================================================================
# 5. Header Restrictions
# ============================================================================

class TestAllowedHeaders:
    """Only Content-Type and Authorization should be allowed."""

    async def test_preflight_allows_content_type(self, default_app):
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.options(
                "/",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type",
                },
            )
        allowed = resp.headers.get("access-control-allow-headers", "").lower()
        assert "content-type" in allowed

    async def test_preflight_allows_authorization(self, default_app):
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.options(
                "/",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "Authorization",
                },
            )
        allowed = resp.headers.get("access-control-allow-headers", "").lower()
        assert "authorization" in allowed

    async def test_preflight_rejects_x_custom_header(self, default_app):
        """Arbitrary custom headers must not be reflected."""
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.options(
                "/",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "X-Custom-Evil-Header",
                },
            )
        allowed = resp.headers.get("access-control-allow-headers", "").lower()
        assert "x-custom-evil-header" not in allowed


# ============================================================================
# 6. Preflight (OPTIONS) Response Behaviour
# ============================================================================

class TestPreflightResponse:
    """Validate the OPTIONS preflight dance end-to-end."""

    async def test_preflight_returns_200_for_allowed_origin(self, default_app):
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.options(
                "/",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                },
            )
        assert resp.status_code == 200

    async def test_preflight_disallowed_origin_no_cors_headers(self, default_app):
        """Preflight from an untrusted origin must not include CORS headers."""
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.options(
                "/",
                headers={
                    "Origin": "https://malicious-site.com",
                    "Access-Control-Request-Method": "GET",
                },
            )
        assert "access-control-allow-origin" not in resp.headers

    async def test_simple_get_without_origin_has_no_cors(self, default_app):
        """Same-origin requests (no Origin header) should not get CORS headers."""
        transport = ASGITransport(app=default_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        assert "access-control-allow-origin" not in resp.headers


# ============================================================================
# 7. Single Custom Origin
# ============================================================================

class TestSingleCustomOrigin:
    """Test with a single custom production origin set via env var."""

    async def test_custom_origin_reflected(self, single_origin_app):
        transport = ASGITransport(app=single_origin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/",
                headers={"Origin": "https://myapp.example.com"},
            )
        assert resp.headers["access-control-allow-origin"] == "https://myapp.example.com"

    async def test_default_localhost_rejected_when_custom_set(self, single_origin_app):
        """If custom origin is set, the old default must no longer work."""
        transport = ASGITransport(app=single_origin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/",
                headers={"Origin": "http://localhost:3000"},
            )
        assert "access-control-allow-origin" not in resp.headers
