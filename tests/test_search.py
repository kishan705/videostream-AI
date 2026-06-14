import pytest
import asyncio
from unittest.mock import patch

def test_search_with_valid_query_returns_results(client, auth_header, mock_qdrant):
    response = client.get("/api/v1/videos/search?query=test", headers=auth_header)
    assert response.status_code == 200
    assert "results" in response.json()

def test_search_top_k_above_100_returns_422(client, auth_header):
    response = client.get("/api/v1/videos/search?query=test&top_k=101", headers=auth_header)
    assert response.status_code == 422

def test_search_top_k_of_zero_returns_422(client, auth_header):
    response = client.get("/api/v1/videos/search?query=test&top_k=0", headers=auth_header)
    assert response.status_code == 422

def test_search_top_k_of_100_is_accepted(client, auth_header):
    response = client.get("/api/v1/videos/search?query=test&top_k=100", headers=auth_header)
    assert response.status_code == 200

def test_search_does_not_call_redis_keys(client, auth_header, fake_redis):
    # Overwrite fake_redis keys so it raises
    fake_redis.keys = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("redis.keys() should not be called"))
    response = client.get("/api/v1/videos/global-search?query=test", headers=auth_header)
    assert response.status_code == 200

def test_search_uses_run_in_executor_not_blocking_get(client, auth_header):
    with patch("asyncio.get_event_loop") as mock_get_event_loop:
        mock_loop = mock_get_event_loop.return_value
        
        async def dummy_coro(*args, **kwargs):
            return [0.1] * 1152
            
        mock_loop.run_in_executor.side_effect = dummy_coro
        
        response = client.get("/api/v1/videos/search?query=test", headers=auth_header)
        assert response.status_code == 200
        mock_loop.run_in_executor.assert_called()

def test_frame_search_top_k_above_100_returns_422(client, auth_header):
    response = client.get("/api/v1/videos/global-search?query=test&top_k=101", headers=auth_header)
    assert response.status_code == 422

def test_search_missing_query_param_returns_422(client, auth_header):
    response = client.get("/api/v1/videos/search", headers=auth_header)
    assert response.status_code == 422
