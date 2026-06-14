import pytest
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.main import app

def test_cors_wildcard_not_present_in_allowed_origins():
    for middleware in app.user_middleware:
        if isinstance(middleware.kwargs.get("allow_origins"), list):
            assert "*" not in middleware.kwargs["allow_origins"]

def test_cors_credentials_require_explicit_origin(client):
    response = client.options("/api/v1/videos/search?query=test", headers={"Origin": "https://evil.com", "Access-Control-Request-Method": "GET"})
    assert response.headers.get("access-control-allow-origin") != "*"

def test_raw_uploads_not_accessible_via_static_mount(client):
    response = client.get("/shared_storage/uploads/test.mp4")
    assert response.status_code == 404

@pytest.mark.parametrize("bad_id", ["../etc", "..%2Fetc", "../../passwd"])
def test_path_traversal_blocked_in_delete(client, auth_header, bad_id):
    response = client.delete(f"/api/v1/videos/{bad_id}", headers=auth_header)
    assert response.status_code in (400, 404)

def test_video_id_uuid_regex_rejects_traversal_strings():
    pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    assert not pattern.match("../etc")
    assert not pattern.match("..%2Fetc")
    assert not pattern.match("../../passwd")
    assert not pattern.match("evil-uuid")
    assert pattern.match("11111111-1111-1111-1111-111111111111")
