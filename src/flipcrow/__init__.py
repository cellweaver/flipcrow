import logging

# Libraries should not configure logging; callers opt in via
# scp1644_mothership.configure_logging().
logging.getLogger(__name__).addHandler(logging.NullHandler())
