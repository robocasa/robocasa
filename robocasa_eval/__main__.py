"""python -m robocasa_eval: no eager simulator or model imports."""
import sys
sys.dont_write_bytecode = True
from .cli import main

raise SystemExit(main())
