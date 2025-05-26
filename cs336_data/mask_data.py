import re

def mask_emails(text):
    email_re = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    matches = email_re.findall(text)
    count = len(matches)
    return email_re.sub("|||EMAIL_ADDRESS|||", text), count

def mask_phone_numbers(text):
    phone_re = re.compile(r"(?:\(\d{3}\)|\d{3})[ \-.]?\d{3}[ \-.]?\d{4}")
    matches = phone_re.findall(text)
    count = len(matches)
    return phone_re.sub("|||PHONE_NUMBER|||", text), count

def mask_ips(text):
    ip_re = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    matches = ip_re.findall(text)
    count = len(matches)
    return ip_re.sub("|||IP_ADDRESS|||", text), count