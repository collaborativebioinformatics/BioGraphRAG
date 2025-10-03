"""Identifier normalization helpers."""

from __future__ import annotations

import re


HGNC_PATTERN = re.compile(r"^HGNC:\d+$")
UNIPROT_PATTERN = re.compile(r"^[A-NR-Z0-9]{6,10}$")


def normalize_gene_symbol(symbol: str | None) -> str | None:
    if symbol:
        return symbol.upper()
    return None


def normalize_gene_id(hgnc_id: str | None, symbol: str | None) -> str | None:
    if hgnc_id and HGNC_PATTERN.match(hgnc_id):
        return hgnc_id
    return normalize_gene_symbol(symbol)


def normalize_uniprot(uniprot_id: str | None) -> str | None:
    if uniprot_id and UNIPROT_PATTERN.match(uniprot_id):
        return uniprot_id
    return None
"""Identifier normalization helpers."""

from __future__ import annotations

import re


HGNC_PATTERN = re.compile(r"^HGNC:\d+$")
ENSEMBL_PATTERN = re.compile(r"^ENSG\d{11}$")
UNIPROT_PATTERN = re.compile(r"^[A-NR-Z0-9]{6,10}$")


def normalize_gene_id(hgnc_id: str | None, symbol: str | None) -> str | None:
    if hgnc_id and HGNC_PATTERN.match(hgnc_id):
        return hgnc_id
    if symbol:
        return symbol.upper()
    return None


def normalize_ensembl_id(ensembl_id: str | None) -> str | None:
    if ensembl_id and ENSEMBL_PATTERN.match(ensembl_id):
        return ensembl_id
    return None


def normalize_uniprot_id(uniprot_id: str | None) -> str | None:
    if uniprot_id and UNIPROT_PATTERN.match(uniprot_id):
        return uniprot_id
    return None




