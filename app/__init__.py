# Shared Config for Logging
import logging

logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
  )