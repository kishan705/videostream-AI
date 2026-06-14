import pytest
import io

def test_upload_without_token_returns_401(client, sample_video_bytes):
    response = client.post("/api/v1/videos/upload", files={"file": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}, data={"title": "Test"})
    assert response.status_code == 401

def test_search_without_token_returns_401(client):
    response = client.get("/api/v1/videos/search?query=test")
    assert response.status_code == 401

def test_delete_without_token_returns_401(client, sample_video_id):
    response = client.delete(f"/api/v1/videos/{sample_video_id}")
    assert response.status_code == 401

def test_upload_with_invalid_token_returns_401(client, sample_video_bytes):
    headers = {"Authorization": "Bearer invalid-token"}
    response = client.post("/api/v1/videos/upload", headers=headers, files={"file": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}, data={"title": "Test"})
    assert response.status_code == 401

def test_upload_with_valid_token_passes_auth_check(client, auth_header, sample_video_bytes):
    response = client.post("/api/v1/videos/upload", headers=auth_header, files={"file": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}, data={"title": "Test"})
    assert response.status_code == 202

def test_malformed_bearer_header_returns_401(client, sample_video_bytes):
    headers = {"Authorization": "invalid-token-format"}
    response = client.post("/api/v1/videos/upload", headers=headers, files={"file": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}, data={"title": "Test"})
    assert response.status_code == 401
