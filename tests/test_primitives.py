"""Sanity: every primitive snippet has the documented arity and uses
${inN} placeholders correctly."""
import pytest

from mnnforge.primitives import (
    BINARY_SNIPPETS, UNARY_SNIPPETS, NAMED_SNIPPETS, HELPERS, fmt,
)


def test_binary_snippet_arity():
    for k, tmpl in BINARY_SNIPPETS.items():
        assert "${in0}" in tmpl
        assert "${in1}" in tmpl


def test_unary_snippet_arity():
    for k, snip in UNARY_SNIPPETS.items():
        assert snip.arity == 1
        assert "${in0}" in snip.expr


def test_named_snippet_arity():
    for k, snip in NAMED_SNIPPETS.items():
        assert snip.arity == 1
        assert "${in0}" in snip.expr


def test_helpers_only_use_known_keys():
    keys_referenced = set()
    for snip in list(UNARY_SNIPPETS.values()) + list(NAMED_SNIPPETS.values()):
        keys_referenced.update(snip.needs)
    for k in keys_referenced:
        assert k in HELPERS, f"helper {k!r} referenced but missing"


def test_fmt_replaces_all_placeholders():
    out = fmt("(${in0} + ${in1}) * ${in0}", "x", "y")
    assert "${" not in out
    assert out == "(x + y) * x"


def test_helper_function_names_match_snippets():
    """Whenever a snippet says it `needs` 'sigmoid', the corresponding
    HELPERS entry must define a function whose name appears in the snippet's
    expr (otherwise the kernel won't link)."""
    for snip in list(UNARY_SNIPPETS.values()) + list(NAMED_SNIPPETS.values()):
        for h in snip.needs:
            helper_src = HELPERS[h]
            # Helper defines a function called _mnn_<name>; it should be the
            # one called inside the snippet expr.
            assert f"_mnn_{h}" in helper_src
            assert f"_mnn_{h}" in snip.expr
