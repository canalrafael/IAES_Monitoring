import re
import numpy as np

def parse_weights(h_path):
    with open(h_path, 'r') as f:
        content = f.read()
    
    # Strip comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    def extract_float_list(pattern):
        match = re.search(pattern, content, re.DOTALL)
        if not match: return None
        return np.array([float(x.replace('f', '')) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', match.group(1))])

    def extract_float_matrix(pattern, rows, cols):
        match = re.search(pattern, content, re.DOTALL)
        if not match: return None
        data = np.array([float(x.replace('f', '')) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', match.group(1))])
        return data.reshape(rows, cols)

    def get_define(name):
        match = re.search(fr'#define\s+{name}\s+(\d+)', content)
        return int(match.group(1)) if match else None

    n_feat = get_define('MDL_N_FEATURES') or 12
    n_h1 = get_define('MDL_N_H1') or 16
    n_h2 = get_define('MDL_N_H2') or 16

    mean = extract_float_list(fr'MDL_FEAT_MEAN\[{n_feat}\] = \{{(.*?)\}};')
    std = extract_float_list(fr'MDL_FEAT_STD\[{n_feat}\] = \{{(.*?)\}};')
    w1 = extract_float_matrix(fr'MDL_W1\[{n_h1}\]\[{n_feat}\] = \{{(.*?)\}};', n_h1, n_feat)
    b1 = extract_float_list(fr'MDL_B1\[{n_h1}\] = \{{(.*?)\}};')
    w2 = extract_float_matrix(fr'MDL_W2\[{n_h2}\]\[{n_h1}\] = \{{(.*?)\}};', n_h2, n_h1)
    b2 = extract_float_list(fr'MDL_B2\[{n_h2}\] = \{{(.*?)\}};')
    w3 = extract_float_matrix(fr'MDL_W3\[1\]\[{n_h2}\] = \{{(.*?)\}};', 1, n_h2)
    b3 = extract_float_list(fr'MDL_B3\[1\] = \{{(.*?)\}};')
    
    return mean, std, w1, b1, w2, b2, w3, b3
