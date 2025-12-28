import numpy as np

fnc_path = r'd:\Hackathons  & Competitions\Synaptix\Model-Physcosis\psychosis-classification-with-rsfmri\data\train\SZ\sub004\fnc.npy'
icn_path = r'd:\Hackathons  & Competitions\Synaptix\Model-Physcosis\psychosis-classification-with-rsfmri\data\train\SZ\sub004\icn_tc.npy'

fnc = np.load(fnc_path)
icn = np.load(icn_path)

print(f"FNC Shape: {fnc.shape}")
print(f"ICN Timecourses Shape: {icn.shape}")
print(f"FNC Data Type: {fnc.dtype}")
print(f"ICN Data Type: {icn.dtype}")
