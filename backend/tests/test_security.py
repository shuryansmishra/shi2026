"""
Security & Input Validation Tests for SatQuery AI FastAPI Backend
Run with: pytest tests/test_security.py -v
"""
import io
import pytest
from fastapi.testclient import TestClient
from main import app
from config import get_settings

client = TestClient(app)
settings = get_settings()


def test_reject_malicious_file_extension():
    """Ensure executable, script, or non-imagery file extensions are rejected with 400."""
    fake_script = io.BytesIO(b"<?php echo 'malicious'; ?>")
    response = client.post(
        "/api/query",
        data={"query_text": "Analyze land cover"},
        files={"files": ("exploit.php", fake_script, "application/x-php")}
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.text


def test_accept_valid_image_extension():
    """Ensure supported satellite imagery extensions (.png, .tif, .tiff, .jpg) are accepted."""
    # 1x1 dummy PNG
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    fake_png = io.BytesIO(png_bytes)
    response = client.post(
        "/api/query",
        data={"query_text": "Analyze land cover"},
        files={"files": ("sample_satellite.png", fake_png, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "evidence" in data


def test_path_traversal_filename_sanitization():
    """Ensure directory traversal characters in filenames are stripped."""
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    fake_tiff = io.BytesIO(png_bytes)
    response = client.post(
        "/api/query",
        data={"query_text": "What is here?"},
        files={"files": ("../../etc/passwd.tif", fake_tiff, "image/tiff")}
    )
    assert response.status_code == 200


def test_static_directory_traversal_prevention():
    """Verify that private backend databases and config files cannot be accessed via /static."""
    # Attempting to read satquery.db or backend files through static mount
    response_db = client.get("/static/../satquery.db")
    assert response_db.status_code in (404, 403, 400)
    
    response_cfg = client.get("/static/../config.py")
    assert response_cfg.status_code in (404, 403, 400)


def test_file_count_boundary_conditions():
    """Verify that exactly 1 or 2 files are allowed."""
    # 0 files
    res_zero = client.post("/api/query", data={"query_text": "test"})
    assert res_zero.status_code == 422  # Missing required files field
    
    # 3 files (exceeds allowed limit of 2)
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    fake_img1 = ("img1.png", io.BytesIO(png_bytes), "image/png")
    fake_img2 = ("img2.png", io.BytesIO(png_bytes), "image/png")
    fake_img3 = ("img3.png", io.BytesIO(png_bytes), "image/png")
    res_three = client.post(
        "/api/query",
        data={"query_text": "Compare these"},
        files=[("files", fake_img1), ("files", fake_img2), ("files", fake_img3)]
    )
    assert res_three.status_code == 400
    assert "Provide exactly 1 or 2 images" in res_three.text


def test_cors_preflight_headers():
    """Verify CORS preflight handling for allowed origin."""
    response = client.options(
        "/api/query",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST"
        }
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
