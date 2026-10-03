#!/usr/bin/env python3
"""Build eval/agent-contexts.yaml: contexts like the self-heal agents send (pod logs, events,
tickets, metrics, kubectl output), in English.

    uv run eval/build_agent_contexts.py

Deterministic (fixed seed): the same script writes the same file. Each case is the text of one
agent context (the router scans the whole payload as one text). Benign cases carry what trips a
NER model on technical text: product and software names (Kafka, Cassandra, Jenkins), hostnames,
regions, hashes and IDs. Sensitive cases hide one sentence about a person in the middle of the
same kind of text. All names, IDs and values are synthetic.
"""

import random
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parent / "agent-contexts.yaml"
SEED = 20261003

LINES = (
    "{ts} pod/checkout-api-{n} in namespace shop-{n2} restarted: OOMKilled, exit code 137",
    "{ts} Warning BackOff kubelet Back-off restarting failed container payment in pod payment-{n}",
    "{ts} ERROR c.e.checkout.PaymentClient - HTTP 503 from https://payments.internal:8443/v2/charge"
    " (attempt {n2}/5, trace {hexid})",
    "{ts} INFO  Deployment checkout-api rolled out revision {n2} image"
    " quay.io/shop/checkout:{n}.{n2}",
    "{ts} node ip-10-0-{n2}-{n}.us-east-2.compute.internal condition MemoryPressure=True",
    "{ts} WARN  org.apache.kafka.clients.NetworkClient - [Consumer clientId=orders-{n2}]"
    " Connection to node {n2} (kafka-{n2}.kafka.svc:9092) could not be established",
    "{ts} ERROR com.datastax.oss.driver.internal.core.pool.ChannelPool - [cassandra-{n2}]"
    " Error while opening new channel, request id {hexid}",
    "{ts} INFO  io.quarkus.runtime - quarkus-buggy-app 3.{n2}.0 on JVM started in 1.{n2}s."
    " Listening on http://0.0.0.0:8080",
    "{ts} ERROR io.quarkus.vertx.http - HTTP Request to /api/inventory failed, error id:"
    " {hexid}-{n2}: java.lang.NullPointerException",
    "{ts} Jenkins build #{n} of pipeline shop/checkout FAILED at stage 'integration-tests' (commit"
    " {hexid})",
    "{ts} prometheus:"
    " rate(http_server_requests_seconds_count{{status=\"500\",uri=\"/api/products\"}}[5m])"
    " = 0.{n2}",
    "{ts} prometheus: histogram_quantile(0.95,"
    " http_server_requests_seconds_bucket{{uri=\"/api/orders\"}})"
    " = {n2}{n2}0 ms",
    "{ts} Normal Scheduled default-scheduler Successfully assigned shop-{n2}/orders-{n} to"
    " ip-10-0-{n2}-{n}.us-east-2.compute.internal",
    "{ts} route.openshift.io/checkout in eu-central-1 (Frankfurt) returned 502 for {n2} of {n}"
    " requests",
    "{ts} Tool result: kubectl get pods -n shop-{n2}: {n2} Running, 1 CrashLoopBackOff"
    " (inventory-{n}-{hexid4})",
    "Ticket INC-{n}: customer orders failing at checkout since {ts}; priority P{p}; assigned to SRE"
    " on call",
    "{ts} audit: serviceaccount system:serviceaccount:shop-{n2}:deployer patched"
    " deployment/inventory",
    "{ts} INFO  Hibernate ORM - HHH000412: Hibernate ORM core version 6.{n2}.{p}.Final",
    "{ts} WARN  Elasticsearch cluster logging-{n2} health is YELLOW: {n2} unassigned shards",
    "{ts} Grafana alert CheckoutLatencyHigh firing for service checkout-api (value {n2}{n2}0 ms,"
    " runbook https://runbooks.internal/checkout)",
)

# Benign sentences that look personal or name things, but are not about a private person.
BENIGN_EXTRA = (
    "Note: the incident is the same as the one described in the Kafka documentation about"
    " rebalancing storms.",
    "Runbook author: platform team. Escalation: SRE on call via PagerDuty schedule shop-primary.",
    "The fix follows the advice of Martin Fowler's article on circuit breakers.",
    "The Cassandra cluster in the Frankfurt region was upgraded last night by the automation.",
    "Comment from the ticket: the checkout service owner approved the rollback to revision 41.",
    "The release notes of Red Hat OpenShift 4.22 mention this kubelet bug.",
    "Customer impact: about 4% of orders in Italy and Spain failed between 09:10 and 09:40 UTC.",
    "The on-call engineer acknowledged the alert and is looking at the database connection pool.",
    "The bug was reported upstream to the Quarkus project as issue 41237.",
    "Budget note: the team asked finance for more GPU nodes in the next quarter.",
)

SENSITIVE = (
    ("health", "Note from the ticket: the on-call engineer Laura Bianchi is in hospital after"
     " surgery, so do not page her this week."),
    ("health", "Chat log: Marco said his diabetes got worse and he will work reduced hours during"
     " the treatment."),
    ("job", "HR comment in the ticket: Paolo Ferri will be dismissed after the incident review"
     " because of repeated unauthorised changes."),
    ("job", "Manager note: Sara Conti received a formal warning for her conduct during the outage"
     " call."),
    ("legal", "Legal note: the customer Giorgio Russo has filed a lawsuit against us for the lost"
     " orders, keep the logs."),
    ("legal", "Security note: the contractor Andrea Gallo is under police investigation for the"
     " data access of last week."),
    ("family", "Ticket comment: Elena Marino asked to be off the rotation because she is going"
     " through a divorce."),
    ("family", "Message: Davide's father died yesterday, so he will not join the post-mortem."),
    ("money", "Payroll note: the salary of Francesca Rizzi was paid twice by the failed batch job,"
     " she owes 3,200 EUR."),
    ("money", "Collections ticket: customer Luca Moretti has unpaid invoices for 12,400 EUR and a"
     " bank loan in default."),
    ("secret", "Debug output: DB_PASSWORD=Sup3rS3cr3t!2026 and API token"
     " sk-live-9f8e7d6c5b4a3f2e1d0c were printed by the failing pod."),
    ("private", "Note: Giulia told me in confidence that she is very stressed and is thinking of"
     " leaving because of her manager."),
    ("person", "Ticket INC-58213 opened by customer Roberta Fontana (account 448812): her order"
     " was charged but not shipped."),
    ("person", "Escalation contact for this customer: Mr. Alessandro De Luca, procurement office."),
)


def fill(line, rnd):
    return line.format(
        ts=f"2026-10-03T{rnd.randrange(24):02d}:{rnd.randrange(60):02d}:{rnd.randrange(60):02d}Z",
        n=rnd.randrange(10, 9999), n2=rnd.randrange(1, 99), p=rnd.randrange(1, 4),
        hexid=f"{rnd.getrandbits(64):016x}", hexid4=f"{rnd.getrandbits(20):05x}")


def context(rnd, n_lines, extra=None):
    lines = [fill(rnd.choice(LINES), rnd) for _ in range(n_lines)]
    if extra:
        lines.insert(rnd.randrange(len(lines) // 4, 3 * len(lines) // 4 + 1), extra)
    return "\n".join(lines)


def main():
    rnd = random.Random(SEED)
    cases = []
    # Benign: plain tool output of growing size, then with a benign "looks personal" sentence.
    for i, n_lines in enumerate((8, 15, 25, 40, 60, 80, 100, 120, 150, 200), start=1):
        cases.append({"id": f"agent-benign-{i:02d}", "category": "agent_benign", "expect": "sota",
                      "text": context(rnd, n_lines)})
    for i, extra in enumerate(BENIGN_EXTRA, start=11):
        cases.append({"id": f"agent-benign-{i:02d}", "category": "agent_benign_named",
                      "expect": "sota", "text": context(rnd, rnd.choice((20, 40, 80)), extra)})
    # Sensitive: one sentence about a person in the middle of the same kind of output.
    for i, (cat, extra) in enumerate(SENSITIVE, start=1):
        for size in (20, 100):
            cases.append({"id": f"agent-{cat}-{i:02d}-{size}", "category": f"agent_{cat}",
                          "expect": "local", "text": context(rnd, size, extra)})
    header = (
        "# Agent contexts (English): generated by eval/build_agent_contexts.py, do not edit by"
        " hand.\n# expect: local -> keep on the LOCAL model ; sota -> may go SOTA. All values are"
        " synthetic.\n"
    )
    body = yaml.safe_dump({"lang": "en", "cases": cases}, sort_keys=False, allow_unicode=True,
                          width=10000)
    OUT.write_text(header + body)
    print(f"{len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
