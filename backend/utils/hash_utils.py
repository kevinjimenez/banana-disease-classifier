import hashlib

def compute_image_hash(image_bytes: bytes) -> str:
    """Calcula SHA256 hash de una imagen."""
    return hashlib.sha256(image_bytes).hexdigest()