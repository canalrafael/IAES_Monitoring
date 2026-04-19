import numpy as np
benign = ['data0_clean.csv', 'data10_clean.csv', 'data12_clean.csv', 'data15_clean.csv', 'data18_clean.csv', 'data19_clean.csv', 'data21_clean.csv', 'data23_clean.csv', 'data24_clean.csv', 'data7_clean.csv']
attack = ['data13_clean.csv', 'data14_clean.csv', 'data16_clean.csv', 'data17_clean.csv', 'data20_clean.csv', 'data22_clean.csv', 'data25_clean.csv', 'data26_clean.csv', 'data27_clean.csv', 'data3_clean.csv', 'data4_clean.csv', 'data5_clean.csv', 'data6_clean.csv']
np.random.seed(42)
print('Test Benign:', list(np.random.choice(benign, 3, replace=False)))
print('Test Attack:', list(np.random.choice(attack, 3, replace=False)))
