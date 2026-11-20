"""URL validation and SSRF protection.

`validate_url` is the single gate every outbound request must pass:

  1. normalize + parse the URL (scheme, credentials, port, length)
  2. reject obviously internal hostnames (localhost, *.internal, single-label)
  3. resolve DNS and reject the URL if ANY resolved address is not a public IP
     (loopback, private, link-local/cloud-metadata, CGNAT, multicast, reserved,
     plus IPv4 hidden inside IPv4-mapped / 6to4 / Teredo / NAT64 IPv6 addresses)

Redirects must be re-validated hop by hop with `validate_redirect`.

Known limitation: DNS is resolved here and again by the HTTP client when it
connects, so a hostile DNS server could in theory answer differently the second
time (DNS rebinding). For this project that residual risk is accepted and
documented in the README; production code should pin the connection to the
validated IP.
"""
import ipaddress
import socket
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit, urlunsplit

MAX_URL_LENGTH = 2048
ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {80, 443}
DEFAULT_PORTS = {"http": 80, "https": 443}

BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal", "metadata"}
BLOCKED_SUFFIXES = (
    ".localhost", ".local", ".internal", ".lan", ".home.arpa", ".intranet", ".corp",
)

_NAT64 = ipaddress.ip_network("64:ff9b::/96")


class URLValidationError(Exception):
    """Raised for any URL we refuse to fetch. Maps to a 4xx API error."""

    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class SafeURL:
    url: str
    scheme: str
    host: str
    port: int
    ips: tuple[str, ...] = field(default_factory=tuple)


def _invalid(message: str) -> URLValidationError:
    return URLValidationError("invalid_url", message, 400)


def _blocked(message: str) -> URLValidationError:
    return URLValidationError("blocked_url", message, 400)


def _embedded_addresses(ip: ipaddress._BaseAddress) -> list[ipaddress._BaseAddress]:
    """Return the IP itself plus any IPv4 address hidden inside an IPv6 one."""
    found = [ip]
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            found.append(ip.ipv4_mapped)
        if ip.sixtofour:
            found.append(ip.sixtofour)
        if ip.teredo:
            server, client = ip.teredo
            found.extend([server, client])
        if ip in _NAT64:
            found.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    return found


def is_public_ip(ip: ipaddress._BaseAddress) -> bool:
    """True only if the address (and anything embedded in it) is globally routable."""
    for candidate in _embedded_addresses(ip):
        if (
            not candidate.is_global
            or candidate.is_private
            or candidate.is_loopback
            or candidate.is_link_local
            or candidate.is_multicast
            or candidate.is_reserved
            or candidate.is_unspecified
        ):
            return False
    return True


def _check_hostname(host: str) -> None:
    if host in BLOCKED_HOSTNAMES or host.endswith(BLOCKED_SUFFIXES):
        raise _blocked("This host is not allowed.")
    if "." not in host:
        raise _blocked("Single-label (internal) hostnames are not allowed.")


def _resolve(host: str, port: int) -> tuple[str, ...]:
    """Resolve a hostname and make sure every address is public."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        raise URLValidationError(
            "dns_error", f"Could not resolve the host '{host}'.", 400
        ) from None

    ips: list[str] = []
    for info in infos:
        raw = info[4][0].split("%")[0]  # strip IPv6 zone id
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            continue
        if not is_public_ip(ip):
            raise _blocked("This URL resolves to a private or internal address.")
        ips.append(str(ip))

    if not ips:
        raise URLValidationError("dns_error", f"Could not resolve the host '{host}'.", 400)
    return tuple(dict.fromkeys(ips))


def validate_url(raw_url: str) -> SafeURL:
    """Validate a user-supplied URL and return a normalized, safe-to-fetch form."""
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise _invalid("URL must not be empty.")

    url = raw_url.strip()
    if len(url) > MAX_URL_LENGTH:
        raise _invalid(f"URL is longer than {MAX_URL_LENGTH} characters.")
    if any(ch.isspace() or ord(ch) < 32 for ch in url):
        raise _invalid("URL must not contain spaces or control characters.")
    if "://" not in url:
        url = "https://" + url  # allow 'example.com' without a scheme

    try:
        parts = urlsplit(url)
        host = parts.hostname
        explicit_port = parts.port
    except ValueError:
        raise _invalid("URL is malformed.") from None

    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise _invalid("Only http and https URLs are supported.")
    if not host:
        raise _invalid("URL has no hostname.")
    if parts.username is not None or parts.password is not None:
        raise _blocked("URLs containing credentials are not allowed.")

    port = explicit_port or DEFAULT_PORTS[scheme]
    if port not in ALLOWED_PORTS:
        raise _blocked(f"Port {port} is not allowed. Use 80 or 443.")

    host = host.rstrip(".").lower()
    try:
        host = host.encode("idna").decode("ascii")  # normalize unicode domains
    except UnicodeError:
        raise _invalid("Hostname is not valid.") from None

    # IP literal: check directly, no DNS needed.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        if not is_public_ip(literal):
            raise _blocked("Private or internal IP addresses are not allowed.")
        ips = (str(literal),)
    else:
        _check_hostname(host)
        ips = _resolve(host, port)  # also catches decimal/hex IPs like 2130706433

    netloc = f"[{host}]" if ":" in host else host
    if explicit_port:
        netloc += f":{explicit_port}"
    clean = urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))
    return SafeURL(url=clean, scheme=scheme, host=host, port=port, ips=ips)


def validate_redirect(current_url: str, location: str) -> SafeURL:
    """Validate a redirect target. Call this for EVERY hop; never follow blindly."""
    return validate_url(urljoin(current_url, location))
