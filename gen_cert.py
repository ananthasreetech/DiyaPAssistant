"""
Generates a self-signed SSL cert for local HTTPS.
Run once: python gen_cert.py
Requires: pip install cryptography
"""
import ipaddress, datetime, pathlib
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import socket

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

local_ip = get_local_ip()
print(f"Detected local IP: {local_ip}")

# Generate private key
key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

# Build cert
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COMMON_NAME, local_ip),
])
cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(datetime.datetime.utcnow())
    .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
    .add_extension(
        x509.SubjectAlternativeName([
            x509.IPAddress(ipaddress.IPv4Address(local_ip)),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]),
        critical=False,
    )
    .sign(key, hashes.SHA256())
)

pathlib.Path("key.pem").write_bytes(
    key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
)
pathlib.Path("cert.pem").write_bytes(
    cert.public_bytes(serialization.Encoding.PEM)
)

print(f"\nCert generated successfully.")
print(f"\nNow run:  run_https.bat")
print(f"On mobile Chrome open:  https://{local_ip}:8501")
print(f"(tap Advanced → Proceed when Chrome warns about the cert)")
