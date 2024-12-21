# Packages
import os, sys
import importlib

# Appending src dir. for module import
sys.path.append(os.path.dirname(os.getcwd()))

# Modules
import models.irpd as irpd
importlib.reload(irpd)


# Base testing instance
base_tests = irpd.IRPD(dir_path="/Users/fogellmcmuffin/Dropbox/ai_irpd_coding/")