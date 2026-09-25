"""Detector A validators: they turn loose regex matches into verified entities."""

import pytest
from privacy_scoring import codice_fiscale, iban_mod97, luhn


@pytest.mark.parametrize("number, valid", [
    ("4111 1111 1111 1111", True),     # test Visa number
    ("5555555555554444", True),        # test Mastercard number
    ("4111 1111 1111 1112", False),    # wrong check digit
    ("1111111111111111", False),       # all the same digit: placeholder
    ("411111111111", False),           # too short
])
def test_luhn(number, valid):
    assert luhn(number) is valid


@pytest.mark.parametrize("cf, valid", [
    ("RSSMRA80A01H501U", True),
    ("rssmra80a01h501u", True),        # case does not matter
    ("RSSMRA80A01H501X", False),       # wrong check letter
    ("BNCGLI85M41F205X", False),       # invented code used in the e2e prompt R7
    ("RSSMRA80A01H501", False),        # 15 characters
])
def test_codice_fiscale(cf, valid):
    assert codice_fiscale(cf) is valid


@pytest.mark.parametrize("iban, valid", [
    ("IT60X0542811101000000123456", True),
    ("IT60 X054 2811 1010 0000 0123 456", True),   # spaces are ignored
    ("GB82WEST12345698765432", True),
    ("IT61X0542811101000000123456", False),        # wrong check digits
    ("IT60X05428", False),                         # too short
])
def test_iban_mod97(iban, valid):
    assert iban_mod97(iban) is valid
