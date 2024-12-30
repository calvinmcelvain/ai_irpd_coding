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

# Testing args
instance = 'uni'
ras = 'eli'
stages = ['1', '1r', '1c', '2', '3']
treatments = 'noise'
sub_test_args = {'instance': instance, 'ras': ras, 'stages': stages, 'treatments': treatments, 'test_type': 'subtest'}
test_test_args = {'instance': instance, 'ras': ras, 'stages': stages, 'treatments': treatments, 'test_type': 'test'}