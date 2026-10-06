"""Namespace policy (v0.11.0): requests about a restricted namespace stay LOCAL.

A platform team marks a namespace with the label `sovereign-selfheal.io/data-class`
(the key comes from `namespace_policy.label` in chain.yaml). Only the value `restricted`
changes the routing: a request about such a namespace goes to the local model and the
gates do not run. No label, `public` or any other value keeps the normal routing.

The router learns which namespaces a request is about in two ways:

  * hint (B): the agent sends the names in the request body, field `selfheal_namespaces`
    (`namespace_policy.hint_field`). The hook always removes this field, also when the
    hint is off, so LiteLLM never forwards it to a model.
  * scan (C): the router finds namespace names in the text of the request (PromQL label
    matchers, JSON and YAML `namespace` keys, `<name>.svc`, `-n <name>`, API paths).

A request is about a restricted namespace when the hint OR the scan names one: the scan
can only make the routing stricter. The labels are read from the Kubernetes API with the
ServiceAccount of the pod: the first read happens at start, then every `refresh_s`
seconds in a thread. A failed refresh keeps the last list. When the list was never read,
nothing is restricted (normal routing, with the privacy gate): the log line and the metric
`router_namespace_labels_loaded` show it.

No LiteLLM import: testable alone, like privacy_scoring.
"""

import os
import re
import ssl
import threading
import time

try:  # httpx is a hard LiteLLM dependency; guard so import never breaks the proxy
    import httpx
except Exception:  # pragma: no cover - defensive
    httpx = None

DEFAULT_LABEL = "sovereign-selfheal.io/data-class"
DEFAULT_HINT_FIELD = "selfheal_namespaces"
DEFAULT_REFRESH_S = 5.0
RESTRICTED = "restricted"
KNOWN_VALUES = ("restricted", "public")
SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
MAX_HINT_NAMES = 20
MAX_LOG_NAMES = 20

# A namespace name: an RFC 1123 label (lowercase letters, digits and '-', at most 63).
_NAME = r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?"
_NAME_RE = re.compile(rf"^{_NAME}$")
# Structured mentions of a namespace in agent text. Each alternative captures one name:
#   namespace="x" / namespace=~"x|y" / namespace!="x" / "namespace": "x" / namespace: x /
#   - `namespace`: x (the alert labels in the prompt of ogx-alert-translator)
#   (also exported_namespace=... of the GPU metrics)
#   x.svc (service DNS names), -n x / --namespace x / --namespace=x, /namespaces/x
_SCAN_RE = re.compile(
    r"(?<![a-z0-9])namespace[\"'`]?\s*(?:=~|!~|!=|==|=|:)\s*[\"'`]?(" + _NAME + r")"
    r"|(?<![-a-z0-9])(" + _NAME + r")\.svc(?![-a-z0-9])"
    r"|(?:^|\s)(?:-n|--namespace)(?:\s+|=)[\"']?(" + _NAME + r")"
    r"|/namespaces/(" + _NAME + r")",
    re.IGNORECASE,
)


def env_flag(name):
    """True/False from an env var (1, true, yes, on = True), or None when unset or empty."""
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return value.strip().lower() in ("1", "true", "yes", "on")


def scan_names(text):
    """Every namespace name that the text mentions in a structured form, lowercase."""
    names = set()
    for match in _SCAN_RE.finditer(text or ""):
        name = next((g for g in match.groups() if g), None)
        if name:
            names.add(name.lower())
    return names


def hint_names(value):
    """The valid names of a hint: a list of strings (or one string), lowercase, at most 20."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    names = []
    for item in value:
        if not isinstance(item, str):
            continue
        name = item.strip().lower()
        if _NAME_RE.match(name) and name not in names:
            names.append(name)
        if len(names) >= MAX_HINT_NAMES:
            break
    return names


def pop_hint(data, field):
    """Remove the hint field from the request (top level and `extra_body`) and return its value.

    Called for every request, also when the hint is off: LiteLLM must never forward the
    field to a model (an external provider would learn the namespace names).
    """
    value = None
    if isinstance(data, dict):
        if field in data:
            value = data.pop(field)
        extra = data.get("extra_body")
        if isinstance(extra, dict) and field in extra:
            nested = extra.pop(field)
            if value is None:
                value = nested
    return value


def _api_base():
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    if not host:
        raise RuntimeError("KUBERNETES_SERVICE_HOST is not set (not in a pod?)")
    port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
    if ":" in host:  # IPv6
        host = f"[{host}]"
    return f"https://{host}:{port}"


class NamespaceLabels:
    """The data-class label of every labelled namespace, read from the Kubernetes API."""

    def __init__(self, label, refresh_s=DEFAULT_REFRESH_S, fetch=None, on_change=None):
        self.label = label
        self.refresh_s = max(1.0, float(refresh_s))
        self._fetch = fetch or self._fetch_from_api
        self._on_change = on_change
        self._lock = threading.Lock()
        self._labels = {}
        self.loaded = False
        self.last_error = None
        self._thread = None

    # -- reading -----------------------------------------------------------------
    def labels(self):
        with self._lock:
            return dict(self._labels)

    def restricted(self):
        with self._lock:
            return {n for n, v in self._labels.items() if v == RESTRICTED}

    # -- refresh -----------------------------------------------------------------
    def refresh(self):
        """Read the labels once. True on success; on error keep the last list."""
        try:
            labels = dict(self._fetch())
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[policy-router] namespace labels: read error, keeping the last list "
                  f"(loaded={self.loaded}): {self.last_error}", flush=True)
            return False
        with self._lock:
            changed = labels != self._labels or not self.loaded
            self._labels = labels
            self.loaded = True
        self.last_error = None
        if changed:
            unknown = sorted(f"{n}={v}" for n, v in labels.items() if v not in KNOWN_VALUES)
            print(f"[policy-router] namespace labels: restricted={sorted(self.restricted())} "
                  f"unknown_values={unknown}", flush=True)
            if self._on_change is not None:
                try:
                    self._on_change(self)
                except Exception as exc:
                    print(f"[policy-router] namespace labels: callback error (ignored): {exc}", flush=True)
        return True

    def start(self):
        """First read now (before the first request), then a daemon thread refreshes."""
        self.refresh()
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="namespace-labels", daemon=True)
            self._thread.start()

    def _loop(self):
        while True:
            time.sleep(self.refresh_s)
            self.refresh()

    def _fetch_from_api(self):
        if httpx is None:
            raise RuntimeError("httpx unavailable")
        with open(os.path.join(SA_DIR, "token")) as fh:  # re-read: the token rotates
            token = fh.read().strip()
        ctx = ssl.create_default_context(cafile=os.path.join(SA_DIR, "ca.crt"))
        resp = httpx.get(
            f"{_api_base()}/api/v1/namespaces",
            params={"labelSelector": self.label},
            headers={"Authorization": f"Bearer {token}"},
            verify=ctx,
            timeout=5.0,
        )
        resp.raise_for_status()
        out = {}
        for item in resp.json().get("items") or []:
            meta = item.get("metadata") or {}
            name = meta.get("name")
            if name:
                out[name] = str((meta.get("labels") or {}).get(self.label, "")).strip().lower()
        return out


class NamespacePolicy:
    """Policy key `namespace_policy` of chain.yaml, with the env overrides."""

    def __init__(self, cfg, fetch=None, on_change=None, start=True):
        cfg = cfg or {}
        self.label = str(cfg.get("label") or DEFAULT_LABEL)
        self.hint_field = str(cfg.get("hint_field") or DEFAULT_HINT_FIELD)
        scan, hint = env_flag("NAMESPACE_SCAN_ENABLED"), env_flag("NAMESPACE_HINT_ENABLED")
        self.scan_enabled = bool(cfg.get("scan", False)) if scan is None else scan
        self.hint_enabled = bool(cfg.get("hint", False)) if hint is None else hint
        self.enabled = self.scan_enabled or self.hint_enabled
        self.labels = None
        if self.enabled:
            refresh_s = cfg.get("refresh_s", DEFAULT_REFRESH_S) or DEFAULT_REFRESH_S
            self.labels = NamespaceLabels(self.label, refresh_s, fetch=fetch, on_change=on_change)
            if start:
                self.labels.start()

    def evaluate(self, hint_value, text_fn):
        """The namespaces of a request and the restricted ones among them.

        `text_fn` returns the text to scan; it is called only when the scan is on.
        Returns None when the policy is off, else a dict with `restricted` (sorted names),
        `source` (`hint`, `scan`, `hint+scan` or `none`), `namespaces` (the names to log:
        the hint names and the labelled names that the scan found) and `loaded`.
        """
        if not self.enabled:
            return None
        labels = self.labels.labels()
        restricted = {n for n, v in labels.items() if v == RESTRICTED}
        hinted = hint_names(hint_value) if self.hint_enabled else []
        scanned = scan_names(text_fn()) if self.scan_enabled else set()
        r_hint = {n for n in hinted if n in restricted}
        r_scan = {n for n in scanned if n in restricted}
        sources = [s for s, hit in (("hint", r_hint), ("scan", r_scan)) if hit]
        if not sources:
            sources = [s for s, hit in (("hint", hinted), ("scan", scanned & set(labels))) if hit]
        names = list(hinted) + sorted(n for n in scanned if n in labels and n not in hinted)
        return {
            "restricted": sorted(r_hint | r_scan),
            "source": "+".join(sources) or "none",
            "namespaces": names[:MAX_LOG_NAMES],
            "loaded": self.labels.loaded,
        }
