import scipy.io

mat_path = r'd:\Hackathons  & Competitions\Synaptix\Model-Physcosis\psychosis-classification-with-rsfmri\data\train\SZ\sub004\fnc.mat'
mat = scipy.io.loadmat(mat_path)
print("FNC keys:", mat.keys())

icn_mat = scipy.io.loadmat(r'd:\Hackathons  & Competitions\Synaptix\Model-Physcosis\psychosis-classification-with-rsfmri\data\train\SZ\sub004\icn_tc.mat')
print("ICN keys:", icn_mat.keys())
