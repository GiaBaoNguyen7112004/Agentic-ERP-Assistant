"""Allow running the package as: python -m agentic_erp_assistant <args>"""

import sys

from agentic_erp_assistant import main

if __name__ == "__main__":
    main(sys.argv[1:])
