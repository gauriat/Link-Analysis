import socket

import pytest

from app.utils import validators
from app.utils.validators import (
    URLValidationError,
    validate_redirect,
    validate_url,
)


def fake_dns(monkeypatch, mapping: dict[str, list[str]]):
    """Make getaddrinfo return canned answers so tests never touch the network."""

    def _getaddrinfo(host, port, *args, **kwargs):
        if host not in mapping:
            raise socket.gaierror("not found")
        return [
            (socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))
            for ip in mapping[host]
        ]

    monkeypatch.setattr(validators.socket, "getaddrinfo", _getaddrinfo)


@pytest.fixture(autouse=True)
def public_example(monkeypatch):
    fake_dns(monkeypatch, {
        "example.com": ["93.184.216.34"],
        "www.example.com": ["93.184.216.34"],
        "evil.example.org": ["10.0.0.5"],
        "mixed.example.org": ["93.184.216.34", "127.0.0.1"],
    })


# ---------- accepted ----------
def test_valid_https_url():
    safe = validate_url("https://example.com/about?x=1#frag")
    assert safe.url == "https://example.com/about?x=1"
    assert safe.host == "example.com" and safe.port == 443
    assert safe.ips == ("93.184.216.34",)


def test_scheme_is_added_when_missing():
    assert validate_url("example.com").url == "https://example.com/"


def test_http_and_trailing_dot_host():
    assert validate_url("http://www.example.com./").host == "www.example.com"


def test_public_ip_literal_is_allowed():
    assert validate_url("https://8.8.8.8/").ips == ("8.8.8.8",)


# ---------- invalid input ----------
@pytest.mark.parametrize("bad", [
    "", "   ", "ftp://example.com", "file:///etc/passwd", "javascript:alert(1)",
    "https://", "https://exa mple.com", "http://example.com:notaport",
    "https://example.com/" + "a" * 2100,
])
def test_invalid_urls_rejected(bad):
    with pytest.raises(URLValidationError) as exc:
        validate_url(bad)
    assert exc.value.status_code == 400


def test_non_string_rejected():
    with pytest.raises(URLValidationError):
        validate_url(None)  # type: ignore[arg-type]


# ---------- SSRF ----------
@pytest.mark.parametrize("blocked", [
    "http://localhost/",
    "http://LOCALHOST./",
    "http://app.localhost/",
    "http://printer.local/",
    "http://metadata.google.internal/computeMetadata/v1/",
    "http://intranet/",                       # single-label host
    "http://127.0.0.1/",
    "http://127.1.2.3/",
    "http://0.0.0.0/",
    "http://10.0.0.1/",
    "http://172.16.5.4/",
    "http://192.168.1.1/",
    "http://169.254.169.254/latest/meta-data/",  # cloud metadata
    "http://100.64.0.1/",                     # CGNAT
    "http://224.0.0.1/",                      # multicast
    "http://[::1]/",
    "http://[fe80::1]/",
    "http://[fc00::1]/",
    "http://[::ffff:127.0.0.1]/",             # IPv4-mapped loopback
    "http://[::ffff:169.254.169.254]/",
    "http://[2002:7f00:1::]/",                # 6to4 wrapping 127.0.0.1
    "http://[64:ff9b::a00:1]/",               # NAT64 wrapping 10.0.0.1
])
def test_internal_targets_blocked(blocked):
    with pytest.raises(URLValidationError) as exc:
        validate_url(blocked)
    assert exc.value.code == "blocked_url"


@pytest.mark.parametrize("encoded", [
    "http://2130706433/",      # decimal 127.0.0.1
    "http://0x7f000001/",       # hex 127.0.0.1
    "http://0177.0.0.1/",       # octal 127.0.0.1
])
def test_encoded_ip_forms_blocked(encoded, monkeypatch):
    # Use the real resolver here: numeric forms resolve locally with no network.
    monkeypatch.undo()
    with pytest.raises(URLValidationError) as exc:
        validate_url(encoded)
    assert exc.value.code in {"blocked_url", "dns_error", "invalid_url"}


def test_credentials_blocked():
    with pytest.raises(URLValidationError) as exc:
        validate_url("https://user:pass@example.com/")
    assert exc.value.code == "blocked_url"


@pytest.mark.parametrize("url", ["http://example.com:22/", "https://example.com:8080/", "http://example.com:6379/"])
def test_non_web_ports_blocked(url):
    with pytest.raises(URLValidationError) as exc:
        validate_url(url)
    assert exc.value.code == "blocked_url"


def test_hostname_resolving_to_private_ip_blocked():
    with pytest.raises(URLValidationError) as exc:
        validate_url("https://evil.example.org/")
    assert exc.value.code == "blocked_url"


def test_any_private_answer_blocks_the_url():
    with pytest.raises(URLValidationError) as exc:
        validate_url("https://mixed.example.org/")
    assert exc.value.code == "blocked_url"


def test_unresolvable_host():
    with pytest.raises(URLValidationError) as exc:
        validate_url("https://does-not-exist.example.net/")
    assert exc.value.code == "dns_error"


# ---------- redirects ----------
def test_relative_redirect_is_resolved_and_checked():
    assert validate_redirect("https://example.com/a", "/b").url == "https://example.com/b"


def test_redirect_to_internal_address_blocked():
    with pytest.raises(URLValidationError):
        validate_redirect("https://example.com/", "http://169.254.169.254/")
